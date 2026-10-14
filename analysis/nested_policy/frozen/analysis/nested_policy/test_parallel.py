"""Resource thresholds only; scientific evaluation is unchanged."""
import json
from parallel_run import memory_action, jobs


def test_reduce_concurrency_before_ram_is_exhausted():
    assert memory_action(4.9, 3) == 'reduce'
    assert memory_action(4.9, 2) == 'reduce'
    assert memory_action(5.1, 3) is None


def test_single_worker_emergency_and_no_spurious_stop():
    assert memory_action(1.9, 1) == 'stop'
    assert memory_action(4.9, 1) is None
    assert memory_action(1.9, 0) is None


def test_queue_preserves_partial_work_and_skips_complete(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'outer_seeds': [42], 'outer_folds': 3}))
    done = tmp_path / 'seed_42/fold_1'
    done.mkdir(parents=True)
    (done / 'complete.json').write_text('{}')
    partial = tmp_path / 'seed_42/fold_2'
    partial.mkdir(parents=True)
    (partial / 'worker_status.json').write_text('{}')
    assert jobs(tmp_path) == [(42, 2), (42, 3)]
