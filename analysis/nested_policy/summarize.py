"""Recompute the revised SI's two-branch nested MAE/R2 tables without training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICIES = ('real_only', 'always_wgan_with_fallback')
TARGETS = (('SANE', 'all', 'sane', 'uV/K'),
           ('kappa', 'measured', 'kappa', 'W/(m K)'),
           ('kappa', 'all', 'kappa', 'W/(m K)'),
           ('FoM', 'measured', 'fom', 'um/A'),
           ('FoM', 'all', 'fom', 'um/A'))
SEEDS = (42, 2025, 314159)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def near(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError(f'Metric mismatch: {a} != {b}')

def metrics(rows, target):
    y = [r['true_'+target] for r in rows]
    p = [r['pred_'+target] for r in rows]
    if len(y) < 2 or not all(math.isfinite(v) for v in y+p):
        raise ValueError('Metric requires at least two finite observations')
    sst = math.fsum((v-statistics.mean(y))**2 for v in y)
    if sst <= 0:
        raise ValueError('R2 undefined for constant reference values')
    return {'n': len(y), 'mae': math.fsum(abs(a-b) for a,b in zip(y,p))/len(y),
            'r2': 1-math.fsum((a-b)**2 for a,b in zip(y,p))/sst}

def verify_hashes():
    for name, expected in read(HERE/'provenance.json')['sha256'].items():
        actual = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source hash mismatch: {name}')

def compute(raw=HERE/'raw'):
    raw = Path(raw)
    protocol = read(raw/'protocol.json')
    assert not protocol['smoke'] and protocol['hpo_trials_per_branch'] == 100
    assert protocol['outer_seeds'] == list(SEEDS) and protocol['outer_folds'] == 5
    assert protocol['inner_folds'] == 5 and protocol['synthetic_samples'] == 200
    markers = sorted(raw.glob('seed_*/fold_*/complete.json'))
    assert len(markers) == 15, 'Expected 15 complete folds, including generation failures'
    initial = {r['sample_id']: r for r in read_csv(raw/'input_45.csv')}
    assert len(initial) == 45
    records, generation, success, fold_rows = [], [], {}, []
    for marker in markers:
        fold = marker.parent
        result = read(marker)
        seed = result['summaries'][0]['root_seed']
        number = result['summaries'][0]['outer_fold']
        key = (seed, number)
        assert key not in success and seed in SEEDS and number in range(1, 6)
        membership = read(fold/'membership.json')
        train, test = set(membership['outer_train']), set(membership['outer_test'])
        assert len(train) == 36 and len(test) == 9 and not train & test
        assert train | test == set(initial)
        split = read(raw/'splits'/f'seed_{seed}_fold_{number}.json')
        assert train == set(split['train_ids']) and test == set(split['test_ids'])
        for m in sorted((fold/'inner').glob('inner_*/membership.json')):
            inner = read(m)
            assert set(inner['train']) | set(inner['validation']) == train
            assert not set(inner['train']) & set(inner['validation'])
        assert len(list((fold/'inner').glob('inner_*/membership.json'))) == 5
        statuses = sorted(fold.glob('**/generation_status.json'))
        assert len(statuses) == 6 and all('attempts' not in p.parts for p in statuses)
        for path in statuses:
            g = read(path)
            ids = set(g['training_ids'])
            assert ids <= train and not ids & test, 'Generation saw outer held-out rows'
            allowed = train if path.parent.name == 'full_gan' else set(read(path.parent.parent/'membership.json')['train'])
            assert ids == allowed, 'Generation must use its corresponding inner training fold'
            generation.append({'seed': seed, 'fold': number,
                               'task': path.parent.relative_to(fold).as_posix(),
                               'success': g['success'], 'training_rows': len(ids)})
        success[key] = bool(read(fold/'full_gan/generation_status.json')['success'])
        assert success[key] == result['decision']['full_generation_success']
        local = []
        source = read_csv(fold/'outer_predictions.csv')
        assert len(source) == 27  # Original provenance retains all three policies.
        for policy in POLICIES:
            rr = [r for r in source if r['policy'] == policy]
            assert len(rr) == 9 and {r['sample_id'] for r in rr} == test
            for row in rr:
                assert int(row['root_seed']) == seed and int(row['outer_fold']) == number
                r = {k: row[k] for k in ('policy','sample_id','kappa_source')}
                r.update(seed=seed, fold=number, generation_success=success[key])
                for c in ('true_kappa','true_sane','pred_kappa','pred_sane'):
                    r[c] = float(row[c])
                    assert math.isfinite(r[c])
                assert r['true_kappa'] > 0 and r['pred_kappa'] > 0 and r['pred_sane'] >= 0
                ref = initial[r['sample_id']]
                near(r['true_kappa'],ref['kxx']); near(r['true_sane'],ref['S_ANE'])
                assert r['kappa_source'] == ref['kappa_source']
                r['true_fom'] = r['true_sane']/r['true_kappa']
                r['pred_fom'] = r['pred_sane']/r['pred_kappa']
                local.append(r)
            data = [r for r in local if r['policy'] == policy]
            saved = next(x for x in result['summaries'] if x['policy'] == policy)
            for target,subset,field,unit in TARGETS:
                values = [r for r in data if subset == 'all' or r['kappa_source'] == 'measured']
                m = metrics(values,field)
                col = ('mae_fom_measured' if subset == 'measured' else 'mae_fom_all') if field == 'fom' else 'mae_'+field
                if not (field == 'kappa' and subset == 'measured'):
                    near(m['mae'],saved[col])
                fold_rows.append(dict(seed=seed,fold=number,policy=policy,
                    generation_success=success[key],target=target,subset=subset,unit=unit,**m))
        if not success[key]:
            real = {r['sample_id']:r for r in local if r['policy']=='real_only'}
            for r in local:
                assert all(r[c] == real[r['sample_id']][c] for c in ('pred_kappa','pred_sane')), 'Fallback must match Real-only predictions'
        records.extend(local)
    assert len(records) == 270 and len(generation) == 90
    seed_rows, summary = [], []
    for scope in ('all_15_folds','generation_success_only'):
        selected = [r for r in records if scope=='all_15_folds' or r['generation_success']]
        for policy in POLICIES:
            for target,subset,field,unit in TARGETS:
                stats=[]
                for seed in SEEDS:
                    rr=[r for r in selected if r['seed']==seed and r['policy']==policy and (subset=='all' or r['kappa_source']=='measured')]
                    if scope=='all_15_folds':
                        assert len(rr)==(45 if subset=='all' else 32)
                        assert len({r['sample_id'] for r in rr})==len(rr)
                    m=metrics(rr,field);stats.append(m)
                    seed_rows.append(dict(scope=scope,seed=seed,policy=policy,target=target,subset=subset,unit=unit,**m))
                summary.append(dict(scope=scope,policy=policy,target=target,subset=subset,unit=unit,
                    prediction_rows=sum(m['n'] for m in stats),
                    mae_mean=statistics.mean(m['mae'] for m in stats),mae_sd=statistics.stdev(m['mae'] for m in stats),
                    r2_mean=statistics.mean(m['r2'] for m in stats),r2_sd=statistics.stdev(m['r2'] for m in stats)))
    paired={}
    for r in fold_rows:
        if r['target']=='FoM' and r['subset']=='measured':paired.setdefault((r['seed'],r['fold']),{})[r['policy']]=r['mae']
    deltas=[r[POLICIES[1]]-r[POLICIES[0]] for r in paired.values()]
    verification={'folds':15,'prediction_rows':270,'final_generation_successes':sum(success.values()),
        'generation_tasks':90,'generation_rejections':sum(not r['success'] for r in generation),
        'FoM_measured_wins':sum(d < -1e-10 for d in deltas),'FoM_measured_losses':sum(d > 1e-10 for d in deltas),
        'FoM_measured_ties':sum(abs(d)<=1e-10 for d in deltas)}
    return {'summary_metrics.csv':summary,'seed_metrics.csv':seed_rows,'fold_metrics.csv':fold_rows,
            'predictions.csv':records,'generation_tasks.csv':generation},verification

def write_report(output,tables,verification):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    lines=['# Single-attempt nested evaluation for the revised SI','',
        'All 15 folds, including generation failures. Values are mean +/- sample SD across three seed-level pooled OOF metrics.','',
        '| Target | Reference subset | Policy | MAE | R2 |','|---|---|---|---:|---:|']
    for r in tables['summary_metrics.csv']:
        if r['scope']=='all_15_folds':
            lines.append(f"| {r['target']} ({r['unit']}) | {r['subset']} | {r['policy']} | {r['mae_mean']:.5f} +/- {r['mae_sd']:.5f} | {r['r2_mean']:.5f} +/- {r['r2_sd']:.5f} |")
    lines += ['',f"Final generation succeeded in {verification['final_generation_successes']}/15 folds. Measured-kappa FoM: {verification['FoM_measured_wins']} wins, {verification['FoM_measured_losses']} losses, {verification['FoM_measured_ties']} fallback ties.",
        '', 'The primary FoM endpoint uses directly measured kappa. Reconstructed kappa is shown separately. The CSV also contains a matched generation-success-only summary; it is conditional, not the headline result.',
        '', 'This is a retrospective single-DNN evaluation. Repeated partitions are not independent datasets, and seed SD is not a confidence interval. It does not demonstrate general WGAN superiority or acquisition efficiency.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def verify_report(tables,verification):
    for name,expected in tables.items():
        actual=read_csv(HERE/'results'/name)
        assert len(actual)==len(expected),(name,len(actual),len(expected))
        for a,e in zip(actual,expected):
            assert set(a)==set(e)
            for key,value in e.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):near(a[key],value)
                else:assert a[key]==str(value),(name,key,a[key],value)
    assert read(HERE/'results/verification.json')==verification

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    verify_hashes()
    tables,verification=compute()
    if args.verify:verify_report(tables,verification)
    if args.output:write_report(args.output,tables,verification)
    if not args.verify and not args.output:p.error('Specify --verify or --output')
    print(json.dumps(verification))

if __name__=='__main__':main()
