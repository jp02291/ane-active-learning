# Closed-loop active learning for anomalous Nernst heat-flux sensor alloys

Code and data accompanying *Discovery of High-Sensitivity Heat-Flux Sensor
Materials via Active Learning*.

The framework searches an eight-element alloy space for compositions that
maximize the material-level heat-flux sensitivity |*S*<sub>ANE</sub>| / κ. At
each cycle it trains a surrogate on the measured data, decides whether
generatively augmented data improve prediction, ranks candidate compositions,
and hands ten of them to synthesis. Measurements return and the loop repeats.

---

## Installation

Download and unpack the archive at
[https://doi.org/10.5281/zenodo.21913047](https://doi.org/10.5281/zenodo.21913047), then

```bash
cd ane-active-learning
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.10 or later. TensorFlow (generator, surrogate, ensemble) and Optuna
(stage 2) are ordinary dependencies, not extras, so `pip install -e .` is
enough to run every stage below. A GPU shortens generator training but is not
required.

Verify the installation:

```bash
pytest -q          # 197 passed
```

What that checks, and what it does not. The suite is a regression test on the
deterministic parts: featurization, the physics filters, generated-sample
filtering and k-center selection, ensemble pruning, Pareto ranking, the
scenario-ranking criterion, the per-cycle data cutoffs, the Fig. 3(d)
projection, and the Fig. 5(d) disorder parameters. Network training
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
│                   note: the Umklapp exponent here carries the opposite sign
│                   to the form quoted in the supplementary information, with a
│                   compensating Grueneisen parameter. The two are the same
│                   calculation at 300 K -- see `CallawayParams`
├── config.py       all pipeline parameters, YAML-serializable
├── data.py         dataset loading, validation, train-test partition
├── augment.py      WGAN-GP generation and filtering        (Algorithm S2)
│                   note: its ILR transform is invertible and is deliberately
│                   not the one in features.py -- see the module docstring
├── surrogate.py    hyperparameter search and deep ensemble (Algorithms S3, S4)
└── select.py       candidate enumeration and Pareto ranking (Algorithm S5)

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
```

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
`scenario_summary.csv`; a fresh search may rank them differently, which is why
the campaign's choice is pinned rather than read back from that file.

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
training split for a cross-validated number of epochs; that model's test
accuracy is what the manuscript reports. Stage 3 trains a separate sixty-member
ensemble from the same hyperparameters and prunes it. The ensemble is not a
more accurate predictor and is not reported as one -- it exists so that stage 4
has a measure of disagreement to explore along. Stage 2 also ranks the
augmentation scenarios, and the winner's `best_params.json` is what stage 3
reads.

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

The original cycle-1 membership has now been recovered and reconciled against
Supplementary Table S1. `configs/cycle1.yaml` therefore validates and copies
the authoritative 36/9 split in `data/reported_splits/cycle1/`. The cycle-2 and
cycle-3 memberships were not recovered; their sizes match, but their membership
is reconstructed and need not match the campaign. TensorFlow arithmetic and
missing run artifacts still prevent bit-for-bit recovery of the reported MAEs.

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
reconstructed. Cycle 1 reports `fixed_reported_membership`; cycles 2 and 3
report `reconstructed`.

---

## Four things to know before reading the numbers

**The train-test partition is not random.** Compositions in the top 15% by
|*S*<sub>ANE</sub>| / κ are placed in the training set, and the test set is
drawn from the remainder by cluster-stratified sampling. At each cycle only
about ten high-performance compositions existed; assigning any of them to the
test set would have removed the target region from the training data. Reported
test metrics therefore describe accuracy over the bulk of the composition space
rather than within the high-performance region. See `ane.data`.

**Generated data is filtered, not trusted.** A sample from the generator is
kept only if it satisfies the physical constraints, states properties inside
the measured range, and agrees with a surrogate trained on measured data alone
to within that surrogate's calibrated error. When no separate validation file
is available the error is calibrated on a fifth of the training data and
multiplied by a safety factor, which is an in-sample estimate and is why the
factor is there. Checkpoints are accepted only when adding the synthetic data
lowers a proxy model's cross-validated error *and* beats the same comparison
with the synthetic labels shuffled. See `ane.augment`.

**Scenario selection uses the five-fold score, and the campaign's own scores
were not kept.** Scenarios are ranked on `optuna_cv_score` in
`scenario_summary.csv` -- the objective of the winning Optuna trial, which is
the mean over folds of each fold's minimum validation loss. That is the
criterion the manuscript defines. `final_epoch_fold_loss` sits beside it in the
same file and is a minimum over folds taken while choosing the epoch count; it
is a diagnostic, and ranking on it selects scenarios on whichever fold fit
best. An earlier revision of this repository did rank on it. The correction
brings the code in line with the manuscript, but it cannot confirm what the
campaign itself ranked on: the per-cycle cross-validation scores were not
retained, so the recorded branch choices cannot be re-derived from them.

**The cross-validation used for scenario selection is not strictly
out-of-fold.** The generator is trained once per cycle on the full training
split, and the resulting synthetic samples are shared across the folds used for
hyperparameter and augmentation-size selection, so fold-validation samples
contributed indirectly to the generative model. The bias acts in favour of the
augmented branch and therefore does not inflate the central observation, which
is that the augmented branch was *not* always preferred. All performance values
reported in the manuscript were computed on the untouched test set, which is
excluded from generator training, scenario selection and hyperparameter search.

---

---

## Data

`data/data.csv` holds composition, thermal conductivity κ and anomalous Nernst
coefficient |*S*<sub>ANE</sub>| at 300 K.

Thermal conductivity was not reported for the literature entries. For those
rows κ was reconstructed as κ<sub>L</sub> + κ<sub>e</sub>, with the electronic
term from the reported electrical conductivity through the Wiedemann–Franz law
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
install a working pipeline but not enough to pin numerical behaviour: TensorFlow
and scikit-learn both change results across minor versions.

`requirements-lock.txt` pins the campaign environment: Python 3.10 with
TensorFlow 2.20.0, Keras 3.12.0, Optuna 4.7.0, NumPy 1.26.4, SciPy 1.15.2,
scikit-learn 1.7.2, pandas 2.3.3, joblib 1.5.2, PyYAML 6.0.3 and h5py 3.15.1.
The environment is identified rather than assumed -- a surviving model file
records Keras 3.12.0 in its root attributes, the environment reports the same,
and that model's hyperparameters match Supplementary Table S5.

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

The campaign's own run outputs for cycles 2 and 3. `split_manifest.json`,
`best_params.json` and `scenario_summary.csv` were not archived for those
cycles, so a reader can re-run the workflow but cannot check the reported
numbers against records of the runs that produced them. Cycle 1 is the
exception: its 36/9 partition was recovered and is deposited under
`data/split/cycle1/`.

The electrical conductivities as reported by Refs [1] and [2]. Those values
were used during the campaign but were not carried into this repository, so
`data/literature_reconstruction.csv` recovers the column from the deposited
totals as `sigma = (kappa_total - kappa_L) / (L0 T)`. The recovered values,
0.88-1.92e6 S/m, lie in the same range as the measured conductivities of
Supplementary Table S7, but the column is not independent of the reconstruction
it feeds. Substituting the reported conductivities would turn
`tests/test_callaway_reconstruction.py` from a consistency check into an
end-to-end one.

---

## Citation

This deposit is archived at [10.5281/zenodo.21913047](https://doi.org/10.5281/zenodo.21913047); see `CITATION.cff`.

```bibtex
@software{ane_active_learning,
  title  = {Code and data for: Discovery of High-Sensitivity Heat-Flux Sensor Materials via Active Learning},
  author = {Park, Jinho and Jang, Byungkwan and Yu, Hyun and Jin, Hyungyu},
  year   = {2026},
  doi    = {10.5281/zenodo.21913047},
  url    = {https://doi.org/10.5281/zenodo.21913047}
}
```

The article this accompanies is named in the title above; cite it from the
journal record once it appears.

---

## License

MIT.
