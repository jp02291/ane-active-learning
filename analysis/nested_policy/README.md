# Completed single-attempt nested evaluation

This is the reference retrospective evaluation for Supplementary Note S1. It
replaces the earlier simplified nested comparison formerly deposited under
`analysis/robustness_checks/`. The old files remain in Git/release history only.
This replacement changes the evaluation protocol, not just the metric label:
generation is local to each inner training fold and final-generation failures
are retained through Real-only fallback. Retry experiments are not included.

## Published comparison

- Initial 45 compositions; three outer seeds (42, 2025, 314159), five folds each.
- 36 outer training and nine test compositions per fold; no top-FoM retention.
- Five inner folds, 100 HPO trials per branch, 200 synthetic rows, one GAN
  training attempt per generation task, at most 10,000 GAN epochs with early stopping.
- Real-only versus WGAN with explicit Real-only fallback after generation failure.
- Primary endpoint: FoM MAE where kappa was directly measured (32 compositions).
  Reconstructed kappa (13 compositions) is included only in separately labeled metrics.
- Final generation succeeded in 12/15 folds; all 15 folds remain in the headline result.
- Metrics are calculated by pooling predictions within each seed, then taking
  the mean and sample SD across seeds. SD is not a confidence interval; repeated
  folds and predictions are not independent datasets. R2 is 1-SSE/SST, including negatives.

## Recompute the SI table without training

From the repository root, using Python 3.10 or newer (standard library only):

```bash
python analysis/nested_policy/summarize.py --verify
```

This verifies frozen/raw SHA-256 hashes, all 15 complete folds, memberships,
generation records, preserved fallback predictions and the deposited metrics.
It independently recalculates MAE/R2 and compares them to `results/`, without
writing or fitting models. To create a fresh report in another directory:

```bash
python analysis/nested_policy/summarize.py --output artifacts/nested_report
```

`results/summary_metrics.csv` is the SI table source. `seed_metrics.csv` and
`fold_metrics.csv` provide seed/fold MAE and R2 for both models and all target
subsets. `predictions.csv` contains the paired outer predictions (270 rows).
`RESULTS.md` is the compact human-readable report. The generation-success-only
summary is conditional and never replaces the full 15-fold summary.

## Retrain from the frozen scientific implementation

The evaluator imports its sibling `frozen/src/ane` package, not the corrected
current `src/ane` campaign code. The snapshot preserves the exact bytes used
for the reported evaluation. Comments or automatic diagnostic branches in the
snapshot describe that retrospective implementation, not the historical
campaign's manual scenario choice. `requirements-frozen.txt` records the
scientific library versions and identifies supporting dependencies whose
historical versions were not recorded. The original environment was Windows,
Python 3.12.3; full bitwise reproducibility across hardware is not guaranteed.

Use a NEW output directory. The deposited `raw/` records are read-only evidence,
not a portable training checkpoint: model binaries, sampler states and SQLite
files are not included. Fresh retraining creates those checkpoints itself.

```bash
pip install -r analysis/nested_policy/requirements-frozen.txt
python analysis/nested_policy/frozen/analysis/nested_policy/run_nested.py --output artifacts/nested_new --prepare-only
```

On Windows, the checkpoint-recycling launcher limits memory use without reducing
the 100-trial budget (normal checkpoint exits use code 75):

```bash
python analysis/nested_policy/frozen/analysis/nested_policy/parallel_run.py --output artifacts/nested_new --workers 1
```

For a serial run on other platforms, run `run_nested.py --output artifacts/nested_new`
without `--prepare-only`; this does not recycle processes and may need more memory.
The original Windows operational helpers are included unchanged. Their retry-mode
dispatch paths are unused; retry modules/results are deliberately not packaged.

The raw evaluator retains its original three branches, including an ancillary
Adaptive branch. The SI report selects the two prespecified comparison branches
by name, never by performance. Its metric-recomputation code has no TensorFlow
dependency. This audit evaluates single DNNs and does not reproduce campaign
ensemble choices or establish acquisition efficiency.

## Verification tests

```bash
python -m pytest tests/test_scenario_ranking.py tests/test_cycle_configs.py tests/test_recovered_splits.py tests/test_nested_deposit.py -q
python -m pytest analysis/nested_policy/frozen/analysis/nested_policy/test_nested.py analysis/nested_policy/frozen/analysis/nested_policy/test_worker_lifecycle.py -q
```

Run the frozen evaluator tests in a separate process, as shown, because they
intentionally load and patch the frozen ane modules. No network training is
performed by these tests.
