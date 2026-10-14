"""Run independent frozen outer folds in separate CPU processes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import msvcrt
import psutil
from worker_lifecycle import CheckpointRestart, RESTART_EXIT_CODE, install

HERE = Path(__file__).resolve().parent


def memory_action(available_gb, active_workers):
    if available_gb < 5 and active_workers > 1:
        return 'reduce'
    if available_gb < 2 and active_workers == 1:
        return 'stop'
    return None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def jobs(output):
    protocol = read(output / 'protocol.json')
    return [(seed, fold) for seed in protocol['outer_seeds']
            for fold in range(1, protocol['outer_folds'] + 1)
            if not (output / f'seed_{seed}/fold_{fold}/complete.json').exists()]


def verify(output):
    protocol = read(output / 'protocol.json')
    if protocol.get('mode') == 'retry3':
        from retry_policy import verify_retry
        verify_retry(output)
        return
    for key, path in [('code_sha256', HERE / 'run_nested.py'),
                      ('data_sha256', HERE.parents[1] / 'data/data.csv')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise RuntimeError(f'Frozen {key} differs')
    if protocol['smoke'] or protocol['hpo_trials_per_branch'] != 100:
        raise RuntimeError('This launcher requires the full frozen evaluation')
    for package, version in protocol['versions'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Package changed: {package}')
    if sys.version != protocol['python']:
        raise RuntimeError('Python version changed')


def worker(output, seed, fold, max_trials=10):
    if read(output / 'protocol.json').get('mode') == 'retry3':
        from retry_policy import retry_worker
        return retry_worker(output, seed, fold, max_trials)
    verify(output)
    import run_nested as audit
    data = audit.pd.read_csv(audit.ROOT / 'data/data.csv')
    data = data.loc[data.cycle_added == 0].reset_index(drop=True)
    authoritative = audit.pd.read_csv(audit.ROOT / 'data/authoritative_cycle1_45.csv')
    assert len(data) == 45 and data.sample_id.is_unique
    audit.np.testing.assert_allclose(
        data.set_index('sample_id')[audit.C + audit.Y].sort_index(),
        authoritative.set_index('sample_id')[audit.C + audit.Y].sort_index(), rtol=0, atol=1e-10)
    tr, te = list(audit.KFold(5, shuffle=True, random_state=seed).split(data))[fold - 1]
    manifest = {'train_ids': data.iloc[tr].sample_id.tolist(), 'test_ids': data.iloc[te].sample_id.tolist()}
    if manifest != read(output / 'splits' / f'seed_{seed}_fold_{fold}.json'):
        raise RuntimeError('Frozen split differs')
    destination = output / f'seed_{seed}/fold_{fold}'
    destination.mkdir(parents=True, exist_ok=True)
    audit.ACTIVE_STATUS = destination / 'worker_status.json'
    audit.optuna.logging.set_verbosity(audit.optuna.logging.WARNING)
    audit.tf.get_logger().setLevel('ERROR')
    audit.event('Parallel outer fold started', root_seed=seed, fold=fold, pid=os.getpid())
    install(audit, max_trials=max_trials)
    try:
        audit.run_fold(data, output, seed, fold, tr, te, False)
    except CheckpointRestart as exc:
        record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid(),
                  'reason': str(exc), 'completed_work_preserved': True}
        with (destination / 'process_restarts.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record) + '\n')
        audit.event('Worker checkpoint saved; requesting fresh process', reason=str(exc))
        raise SystemExit(RESTART_EXIT_CODE)


def supervise(output, count):
    verify(output)
    # The OS releases this lock even if the supervisor crashes.
    lock = (output / 'parallel.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # Refuse to overlap any evaluator already writing into this output tree.
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid == os.getpid():
            continue
        command = process.info['cmdline'] or []
        if any(Path(arg).name in ('run_nested.py', 'parallel_run.py') for arg in command):
            if '--output' in command and Path(command[command.index('--output') + 1]).resolve() == output:
                raise RuntimeError(f'An evaluator is already active: PID {process.pid}')
    pending = jobs(output)
    running = {}
    failures = []
    memory_events = read(output / 'memory_events.json') if (output / 'memory_events.json').exists() else []
    retiring = None
    effective_count = count
    metadata = {'pid': os.getpid(), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'script': str(Path(__file__).resolve()), 'python': sys.executable,
                'output': str(output), 'workers': count, 'mode': 'parallel_outer_folds',
                'evaluation_mode': read(output / 'protocol.json').get('mode', 'single_attempt'),
                'process_recycling': {'max_new_hpo_trials': 10, 'after_new_reference_fit': True,
                                      'after_gan_result': True, 'restart_exit_code': RESTART_EXIT_CODE},
                'memory_guard': {'reduce_below_available_gb': 5, 'emergency_stop_below_available_gb': 2,
                                 'max_wait_for_saved_boundary_seconds': 60}}
    save(output / 'process.json', metadata)
    save(output / 'parallel_process.json', metadata)
    last_completed = -1
    try:
        while pending or running:
            available_gb = psutil.virtual_memory().available / 2**30
            # Do not launch more workers when the system has little headroom.
            while pending and len(running) < effective_count and not failures and not retiring and available_gb >= 6:
                seed, fold = pending.pop(0)
                destination = output / f'seed_{seed}/fold_{fold}'
                destination.mkdir(parents=True, exist_ok=True)
                log = (destination / 'worker.log').open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--output', str(output), '--worker', '--seed', str(seed), '--fold', str(fold)],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                running[process.pid] = (process, seed, fold, log)
            for pid, (process, seed, fold, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[pid]
                complete = (output / f'seed_{seed}/fold_{fold}/complete.json').exists()
                if code == RESTART_EXIT_CODE and not complete:
                    pending.insert(0, (seed, fold))
                elif code or not complete:
                    failures.append({'seed': seed, 'fold': fold, 'exit_code': code})
            action = memory_action(available_gb, len(running))
            if action == 'stop':
                raise MemoryError('Less than 2 GiB RAM available with one worker; preserved completed trials and stopped safely.')
            if action == 'reduce' and retiring is None:
                # Prefer a worker in HPO, which saves progress after every trial.
                candidates = []
                for pid, (_, seed, fold, _) in running.items():
                    path = output / f'seed_{seed}/fold_{fold}/worker_status.json'
                    state = read(path) if path.exists() else {}
                    try:
                        rss = psutil.Process(pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                    candidates.append((state.get('message') == 'HPO trial finished', rss, pid, path))
                if candidates:
                    _, _, pid, path = max(candidates)
                    retiring = {'pid': pid, 'since': time.monotonic(),
                                'stamp': path.stat().st_mtime_ns if path.exists() else None}
            if retiring:
                pid = retiring['pid']
                if pid not in running:
                    retiring = None
                else:
                    process, seed, fold, log = running[pid]
                    path = output / f'seed_{seed}/fold_{fold}/worker_status.json'
                    boundary = path.exists() and path.stat().st_mtime_ns != retiring['stamp']
                    if boundary or available_gb < 2 or time.monotonic() - retiring['since'] >= 60:
                        process.terminate()
                        process.wait()
                        log.close()
                        del running[pid]
                        if not (output / f'seed_{seed}/fold_{fold}/complete.json').exists():
                            pending.insert(0, (seed, fold))
                        effective_count = max(1, min(effective_count - 1, len(running)))
                        memory_events.append({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'seed': seed, 'fold': fold, 'available_gb': round(available_gb, 2),
                            'new_worker_limit': effective_count, 'saved_boundary_observed': boundary,
                            'action': 'Requeued incomplete fold; saved trials retained; unfinished stage may repeat'})
                        save(output / 'memory_events.json', memory_events)
                        retiring = None
            completed = len(list(output.glob('seed_*/fold_*/complete.json')))
            active = []
            for pid, (_, seed, fold, _) in running.items():
                status_path = output / f'seed_{seed}/fold_{fold}/worker_status.json'
                active.append({'pid': pid, 'seed': seed, 'fold': fold,
                               'status': read(status_path) if status_path.exists() else None})
            save(output / 'status.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 'message': 'Parallel full evaluation', 'workers': active,
                 'completed_outer_folds': completed, 'pending_outer_folds': len(pending),
                 'configured_workers': count, 'effective_worker_limit': effective_count,
                 'available_ram_gb': round(available_gb, 2), 'memory_events': memory_events,
                 'failures': failures})
            if completed != last_completed:
                # Only this process publishes shared reports. Training workers never do.
                subprocess.run([sys.executable, str(HERE / 'make_report.py'), '--output', str(output)], check=True)
                last_completed = completed
            if failures and not running:
                raise RuntimeError(f'Worker failed; remaining jobs preserved: {failures}')
            if running or pending:
                time.sleep(10)
        subprocess.run([sys.executable, str(HERE / 'run_nested.py'), '--output', str(output), '--summarize-only'], check=True)
        save(output / 'status.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'message': 'Full evaluation complete', 'completed_outer_folds': 15, 'failures': failures})
    except BaseException as exc:
        save(output / 'status.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'message': 'Parallel evaluation stopped', 'error': str(exc),
             'completed_outer_folds': len(list(output.glob('seed_*/fold_*/complete.json'))),
             'memory_events': memory_events, 'failures': failures})
        raise
    finally:
        for process, _, _, log in running.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--trials-per-process', type=int, default=10)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.plan_only:
        verify(output)
        print(json.dumps({'workers': args.workers, 'pending_jobs': jobs(output)}))
    elif args.worker:
        worker(output, args.seed, args.fold, args.trials_per_process)
    else:
        supervise(output, args.workers)


if __name__ == '__main__':
    main()
