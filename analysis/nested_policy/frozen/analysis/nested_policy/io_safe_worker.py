"""Apply JSON I/O resilience, then execute the frozen evaluator unchanged."""
import sys
from pathlib import Path
from runtime_io import install

if __name__ == '__main__':
    root = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    install([root])
    import parallel_run
    if parallel_run.read(root / 'protocol.json').get('mode') == 'retry5':
        from retry5_policy import retry_worker
        seed = int(sys.argv[sys.argv.index('--seed') + 1])
        fold = int(sys.argv[sys.argv.index('--fold') + 1])
        retry_worker(root, seed, fold)
    else:
        parallel_run.main()
