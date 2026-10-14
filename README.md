# Closed-loop active learning for anomalous Nernst heat-flux sensor alloys

Code and data accompanying *Discovery of high-sensitivity heat-flux sensor
materials via active learning*.

The framework searches an eight-element alloy space for compositions that
maximize the material-level heat-flux sensitivity |*S*<sub>ANE</sub>| / κ. At
each cycle it trains real-only and augmented surrogates. The researchers
manually chose a scenario after reviewing the held-out MAEs of both properties.
The selected surrogate ranks candidate compositions,
and hands ten of them to synthesis. Measurements return and the loop repeats.

---

## Installation

Download and unpack the archive at
[https://doi.org/10.5281/zenodo.22764249](https://doi.org/10.5281/zenodo.22764249), which
resolves to the latest version, then

```bash
cd ane-active-learning
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.10 or later. TensorFlow (generator, surrogate, ensemble), Optuna
(stage 2) and xgboost (`analysis/benchmark` only) are ordinary dependencies,
not extras, so `pip install -e .` is enough to run every stage below. A GPU
shortens generator training but is not required.

Verify the installation:

```bash
pytest -q
```

What that checks, and what it does not. The suite is a regression test on the
deterministic parts: featurization, the physics filters, generated-sample
filtering and k-center selection, ensemble pruning, Pareto ranking, the
scenario-summary behavior and recorded campaign choices, the per-cycle data cutoffs, the Fig. 4(d)
projection, the Fig. 6(d) disorder parameters, the input-space distances the
Discussion quotes, the seed-variability table of Supplementary Note S2, and the
two comparisons under `analysis/` that the supplement reports. Network training
itself is not covered -- no TensorFlow model is fitted anywhere in the suite --
so a green run means the surrounding logic is unchanged, not that the published
results have been reproduced.

---

## Repository layout

```
src/ane/
├── elements.py     element inventory and atomic property table
├── features.py     composition -> 7 ILR coordinates + 8 descriptors
├── physics.py      Miedema mixing enthalpy; Callaway lattice conductivity
│                   defaults follow the supplementary information: negative
│                   Umklapp exponent, gamma = 3.12. The campaign's own
│                   positive-exponent pair (gamma = 2.0) is kept as
│                   LEGACY_UMKLAPP_SIGN / LEGACY_GAMMA and gives the same
│                   kappa_L at 300 K -- see `CallawayParams`
├── config.py       all pipeline parameters, YAML-serializable
├── data.py         dataset loading, validation, train-test partition
├── augment.py      WGAN-GP generation and filtering
│                   note: its ILR transform is invertible and is deliberately
│                   not the one in features.py -- see the module docstring
├── surrogate.py    hyperparameter search and deep ensemble
└── select.py       candidate enumeration and Pareto ranking

scripts/
├── 0_split_data.py         partition the measured data
├── 1_train_gan.py          train WGAN-GP, emit augmented datasets
├── 2_tune_hyperparams.py   Optuna search per augmentation scenario
├── 3_train_ensemble.py     train and prune the deep ensemble
└── 4_select_candidates.py  enumerate, rank, and export the next batch

configs/
├── default.yaml    every parameter at its default; the parameter reference
├── cycle1.yaml     the three campaign cycles: data cutoff, held-out size,
├── cycle2.yaml     selected scenario, artifact root, and the hyperparameters
└── cycle3.yaml     each cycle actually used
data/        measured compositions and properties; the reconstruction inputs
             for the 13 literature entries; per-run seed-variability output
tests/       numerical parity against the original implementation

analysis/    the published figures that are not produced by a pipeline stage.
             Each directory holds its own README, the script, and the numbers
             the script emits, so a reader can check a figure without re-running
             a cycle.
├── pca/          Fig. 4(d), the biplot of the 30 nominated compositions
├── bootstrap/    Fig. 3, the paired percentile bootstrap of the two branches
├── benchmark/    Supplementary Note S5 and Fig. S5, five regressors compared
│                 under one protocol on the cycle-1 partition
├── acquisition/  Supplementary Fig. S2, the four acquisition rules replayed
│                 on the measured pool, its repeat without the 13
│                 reconstructed literature entries, and the hypervolume that
│                 its legend summarizes
├── input_space/  the distances in the model input space quoted in the
│                 Discussion
├── baseline/     conditional random and diversity-only acquisition baselines
├── disorder/     Fig. 6(d) disorder parameters and the Fig. 6 source data
└── robustness_checks/  nested branch comparison and conditional baselines
```

`augment.py`, `surrogate.py` and `select.py` implement the Methods sections on
data augmentation, the ensemble surrogate and candidate selection respectively;
`data.py` implements the partition described under dataset and data
partitioning.

Every stage imports its featurization from `ane.features`. In the original
notebooks that code was duplicated in four places; consolidating it removes a
whole class of silent failure, where a surrogate is applied to features
computed differently from those it was trained on.

---

## Running a cycle

The stages are sequential and communicate through files.

```bash
python scripts/0_split_data.py       --config configs/cycle3.yaml
python scripts/2_tune_hyperparams.py --config configs/cycle3.yaml --scenario base
python scripts/1_train_gan.py        --config configs/cycle3.yaml
python scripts/2_tune_hyperparams.py --config configs/cycle3.yaml
python scripts/3_train_ensemble.py   --config configs/cycle3.yaml
python scripts/4_select_candidates.py --config configs/cycle3.yaml
```

Stage 2 appears twice: once to build the reference surrogate stage 1 filters
against, once to search every scenario. See below.

Every path below is relative to that cycle's roots: `data/split/cycleN/` for
the partition and `artifacts/cycleN/` for everything else.

| stage | reads | writes |
|---|---|---|
| 0 | `data/data.csv` | `split/{train,test}.csv`, `split_manifest.json` |
| 1 | train split + `dnn_base/` from stage 2 of the same cycle | `augmented_data_n{100..500}.csv`, `gan_manifest.json` |
| 2 | train split + augmented sets | `best_params.json`, `final_best_model.h5`, `metrics_test.json` per scenario; `scenario_summary.csv` |
| 3 | train split + best params of the selected scenario | `<selected>/ensemble_trained/` |
| 4 | `<selected>/ensemble_trained/` | `selection/pareto_3objective_top5_{performance,exploration}.csv` |

The selected scenario is pinned in each cycle configuration -- `dnn_gan_n200`
for cycles 1 and 3, `dnn_base` for cycle 2 -- so stages 3 and 4 build the
ensemble the campaign used. Stage 2 still evaluates every scenario and writes
`scenario_summary.csv` in execution order, without an automatic ranking or
winner. The campaign choices were manual: the researchers reviewed the held-out
MAEs of kappa and |S_ANE| together. Both targets favored the same branch in each
reported cycle. Stage 3 reads the configured artifact directory, not the first
row of the summary. A fresh experiment requires an explicit choice of that
directory after reviewing its validation results.

**Stage 1 depends on stage 2 of the same cycle.** Its consistency filter needs
`f_ref`, a surrogate trained on the real data of the cycle being run
and nothing else, so stage 2 has to produce that model first:

```bash
python scripts/2_tune_hyperparams.py --config configs/cycle3.yaml --scenario base
python scripts/1_train_gan.py        --config configs/cycle3.yaml
python scripts/2_tune_hyperparams.py --config configs/cycle3.yaml   # every scenario
```

`gan.surrogate_model_dir` points at `<artifact_root>/dnn_base` of the same
cycle, never at a previous one. Running stage 1 before that directory exists
fails rather than silently filtering against a stale model.

Stage 1 refuses to emit anything if no checkpoint improved the proxy
comparison. That is deliberate -- a generator that never helped has not earned
a place in the training set -- so a failed stage 1 is a result, not a bug.

Stage 4 emits the ten compositions for the next round of synthesis. Their
measured properties are appended to `data/data.csv` and the cycle repeats.

**Two models, two jobs.** Stage 2 ends by refitting a single model on the whole
training split for a cross-validated number of epochs; that model's held-out
validation accuracy is what Fig. 3 reports. Stage 3 trains a separate sixty-member
ensemble from the same hyperparameters and prunes it. The ensemble is not a
more accurate predictor and is not reported as one -- it exists so that stage 4
has a measure of disagreement to explore along. Stage 2 records the results
for each augmentation scenario without selecting one. Stage 3 reads
`best_params.json` from the scenario explicitly configured by the researcher.

---

## Re-running a campaign cycle

`data/data.csv` holds all seventy compositions -- the forty-five the campaign
started from and the twenty-five it went on to measure. Splitting that file
whole reproduces no cycle: it would train cycle 1 on the compositions cycle 1
was supposed to find.

`split.up_to_cycle` is what prevents that. It keeps only rows with
`cycle_added` below the given cycle, so the three configuration files select
45, 55 and 65 compositions respectively, and stage 0 produces test sets of
9, 10 and 10 -- the sizes the manuscript reports.

The cycle-1 membership was recovered and reconciled against Supplementary
Table S1, and the cycle-2 and cycle-3 memberships were recovered from the
Fig. 3 prediction record (see `recovery_manifest.json` in
`data/reported_splits/cycle2/` and `cycle3/`). The three configuration files
therefore validate and copy the pinned 36/9, 45/10 and 55/10 splits in
`data/reported_splits/`. TensorFlow arithmetic and missing run artifacts still
prevent bit-for-bit recovery of the reported MAEs.

```bash
python scripts/0_split_data.py --config configs/cycle1.yaml   # 36 train /  9 held out
python scripts/0_split_data.py --config configs/cycle2.yaml   # 45 train / 10 held out
python scripts/0_split_data.py --config configs/cycle3.yaml   # 55 train / 10 held out
```

Stage 0 refuses to run on a file containing post-campaign rows when
`up_to_cycle` is unset, rather than silently splitting all seventy. To use
every row on purpose, set it past the last cycle.

Alongside the two CSVs, stage 0 writes `split_manifest.json` naming every
composition on each side and recording whether the membership was fixed or
reconstructed. All three cycles report `fixed_reported_membership`; a
configuration without `fixed_train_csv` and `fixed_test_csv` draws a new
partition and reports `reconstructed`.

---

## Four things to know before reading the numbers

**The train-test partition is target informed.** Compositions in the top 15% by
|*S*<sub>ANE</sub>| / κ are placed in the training set, and the test set is
drawn at random from the remainder. At each cycle only about ten
high-performance compositions existed; assigning any of them to the test set
would have removed the target region from the training data. Reported test
metrics therefore describe accuracy over the bulk of the composition space
rather than within the high-performance region. It is the retention rule that
does this, not the draw. See `ane.data`.

**Generated data is filtered, not trusted.** A sample from the generator is
kept only if it satisfies the physical constraints, states properties inside
the measured range, and agrees with a surrogate trained on measured data alone
to within that surrogate's calibrated error. When no separate validation file
is available the error is calibrated on a fifth of the training data and
multiplied by a safety factor, which is an in-sample estimate and is why the
factor is there. Checkpoints are accepted only when adding the synthetic data
lowers a proxy model's cross-validated error *and* beats the same comparison
with the synthetic labels shuffled. See `ane.augment`.

**Campaign scenario selection was manual and used held-out validation
performance.** As clarified by the campaign author, the researchers reviewed
the MAEs of kappa and |S_ANE| together rather than applying an automatic CV,
mean-R2, or FoM ranking. Both property MAEs favored augmentation in cycles 1
and 3 and real-only in cycle 2. The selected augmented scenarios used 200
synthetic samples. These recorded choices remain pinned in the cycle configs.
Earlier package documentation incorrectly described them as CV-selected, and
stage 2 sorted its summary by CV score; that automatic sorting has been removed.

`optuna_cv_score` remains the objective of the best hyperparameter trial
within each scenario: the mean over folds of each fold's minimum validation
loss. `final_epoch_fold_loss` is the separate diagnostic used to choose the
refit epoch count. Neither quantity selects a scenario. The summary preserves
execution order and leaves the decision to the researcher; no scalar rule or
tie-breaking rule has been reconstructed from the historical manual choices.

**The held-out campaign set is a scenario-selection validation set, not an
independent test set.** Its target values informed the manual scenario choice,
although they did not enter generator training, filtering, or the internal
hyperparameter search. Legacy file names (`test.csv`, `metrics_test.json`,
`test_predictions.csv`) and `test_*` summary fields are retained for artifact
compatibility; in the campaign comparison they refer to this validation set.
Bootstrap intervals from these data do not correct for scenario selection.
The held-out compositions are also read for duplicate exclusion in
`ane.augment`; target properties are not used in that exclusion step.

**The internal HPO cross-validation is not strictly out-of-fold with respect
to generation.** The generator is trained once on the full cycle-training
split, and the resulting synthetic samples are shared across HPO folds. Thus,
inner validation rows contributed indirectly to generation. These internal
scores are tuning diagnostics, not independent estimates of augmentation
benefit. The separately deposited retrospective nested comparison has its own
scope and limitations, described in `analysis/robustness_checks/README.md`.

---

## Data

`data/data.csv` holds composition, thermal conductivity κ and anomalous Nernst
coefficient |*S*<sub>ANE</sub>| at 300 K.

Thermal conductivity was not reported for the literature entries. For those
rows κ was reconstructed as κ<sub>L</sub> + κ<sub>e</sub>, with the electronic
term from the reported electrical conductivity through the Wiedemann-Franz law
and the lattice term from the Callaway model in `ane.physics`. Those values are
marked in the `kappa_source` column and are model-derived, not measured.

---

## Reproducibility

The campaign configuration records distinct stochastic roots: 42 for the
split, GAN training, Optuna/CV and candidate enumeration; 2027 for the final
GAN pool; and `2025 + m` for ensemble member `m`. These historical values are
kept because changing them retroactively can change the candidate ranking.

For a standardized seed-42 reanalysis, pass `--master-seed 42` to **every**
stage. This sets every component root to 42; ensemble member `m` uses `42 + m`
so that the 60 members remain distinct. Every stage records its effective seed
in its manifest or metadata. Example:

```bash
python scripts/0_split_data.py        --config configs/cycle1.yaml --master-seed 42
python scripts/2_tune_hyperparams.py  --config configs/cycle1.yaml --scenario base --master-seed 42
python scripts/1_train_gan.py         --config configs/cycle1.yaml --master-seed 42
python scripts/2_tune_hyperparams.py  --config configs/cycle1.yaml --master-seed 42
python scripts/3_train_ensemble.py    --config configs/cycle1.yaml --master-seed 42
python scripts/4_select_candidates.py --config configs/cycle1.yaml --master-seed 42
```

Python, NumPy and TensorFlow are seeded at the start of each relevant stage or
ensemble member. Results should be reproducible on the same platform;
TensorFlow does not guarantee bitwise identity across hardware or library
versions, so small differences in the last digits are expected elsewhere.

The mechanism has a name. TensorFlow enables oneDNN custom operations by
default and warns at import that they "may see slightly different numerical
results due to floating-point round-off errors from different computation
orders". The campaign ran with that default. Setting `TF_ENABLE_ONEDNN_OPTS=0`
removes the ordering dependence but does not reproduce the campaign's
arithmetic either; it is a different, more deterministic run.

Each cycle's configuration file is the record of what that cycle was run with.
Keep them.

Models are written as HDF5 (`.h5`). Keras 3 treats that format as legacy and
warns on save; it is kept because it is the format the released artifacts use
and the format stage 4 discovers members by.

The dependency bounds in `pyproject.toml` are lower bounds, which is enough to
install a working pipeline but not enough to pin numerical behavior: TensorFlow
and scikit-learn both change results across minor versions.

`requirements-lock.txt` pins the campaign environment: Python 3.10 with
TensorFlow 2.20.0, Keras 3.12.0, Optuna 4.7.0, NumPy 1.26.4, SciPy 1.15.2,
scikit-learn 1.7.2, pandas 2.3.3, joblib 1.5.2, PyYAML 6.0.3, h5py 3.15.1 and
xgboost 2.1.4.
The environment is identified rather than assumed -- a surviving model file
records Keras 3.12.0 in its root attributes, the environment reports the same,
and that model's hyperparameters match Supplementary Table S8.

---

## What is not in this repository

The four Jupyter notebooks the pipeline was originally written as. The copies
that survive were edited after the campaign finished, during exploratory
testing, and several of their settings no longer match the runs behind the
reported results. Publishing them would mean publishing code that does not
reproduce the paper. The values that do are in `configs/default.yaml`, and the
test suite pins both sides so that neither can drift silently. The two sources
do not disagree uniformly: the DNN search bounds are a case where the
configuration deliberately departs from the surviving notebook
(`test_search_bounds_superseding_the_surviving_notebook`), while the GAN block
follows it throughout and is pinned value by value
(`test_gan_config_matches_the_notebook_everywhere`).

The complete hyperparameter-search and per-scenario records from the
campaign's own runs were not retained. In particular, `best_params.json` and
`scenario_summary.csv` were not archived for cycles 2 and 3. The deposited
Fig. 3 prediction records support the reported held-out branch comparisons,
but they do not reconstruct every candidate scenario or a historical CV ranking.
The scenario choices were manual, as described above.

`RELEASE_VERIFICATION.json` and `SHA256_MANIFEST.csv` describe the original
v1.1.0 deposit. They are retained as historical records, not regenerated to
make the correction appear part of that release. New working-tree edits will
therefore differ from that original manifest. See `CHANGELOG.md` for the
unreleased correction; this change does not republish the Zenodo deposit.

The partitions themselves are deposited for all three cycles. Cycle 1's was
retained. The cycle 2 and cycle 3 membership was recovered from the Fig. 3
prediction record: its `y_true` values are the measured properties of the
held-out rows, and the (kappa, |S_ANE|) pairs are unique across `data/data.csv`,
so each row identifies one composition.

All three are pinned under `data/reported_splits/` and copied out by stage 0
rather than redrawn, because a fresh draw does not reproduce them -- on the
same pool and the same seed, only three of the ten cycle-2 held-out compositions
coincide. `recovery_manifest.json` records the source files, their
hashes, how far the nearest alternative match sat, and the protocol checks the
recovered set passes.

---

## Citation

This version is archived at [10.5281/zenodo.23027635](https://doi.org/10.5281/zenodo.23027635); the
concept DOI [10.5281/zenodo.22764249](https://doi.org/10.5281/zenodo.22764249) always resolves to
the latest version. See `CITATION.cff`.

```bibtex
@software{ane_active_learning,
  title  = {Code and data for: Discovery of high-sensitivity heat-flux sensor materials via active learning},
  author = {Park, Jinho and Jang, Byungkwan},
  year   = {2026},
  version = {1.1.0},
  doi    = {10.5281/zenodo.23027635},
  url    = {https://doi.org/10.5281/zenodo.23027635}
}
```

The article this accompanies is named in the title above; cite it from the
journal record once it appears.

---

## License

MIT.
