# Changelog

## 1.2.3 - 2026-08-25

- Added `analysis/acquisition/`, which reproduces the acquisition-rule
  comparison of Supplementary Fig. S5 and the repeat reported in Supplementary
  Note S1. This calculation was previously outside the repository.
- Added `analysis/benchmark/`, which reproduces the surrogate-model comparison
  of Supplementary Note S2 and Fig. S6. It too was previously outside the
  repository.
- Both take their pool from `data/data.csv` and their descriptors from
  `ane.features`, so the two comparisons and the campaign surrogate see
  identical inputs.
- Added `tests/test_acquisition_reproduction.py` and
  `tests/test_benchmark_reproduction.py`, which pin the descriptor source, the
  data each comparison runs on, and the published numbers. The complete suite
  contains 223 tests.
- `xgboost` is now a declared dependency. `analysis/benchmark` is the only
  module that needs it.
- `README.md` documents `analysis/`.
- `README.md` and `ane.surrogate` describe the held-out split without calling
  it untouched. No held-out target value enters generator training, filtering,
  scenario selection or hyperparameter search, but the held-out compositions
  are read in the duplicate-exclusion step of `ane.augment` so that a generated
  sample cannot coincide with one of them. The manuscript states the same
  qualification.
- `analysis/pca/run_pca.py` reports 31.0% for the first component in its module
  docstring, matching the value the script computes.
- `analysis/acquisition/README.md` and `analysis/benchmark/README.md` record the
  environment each set of deposited results was produced in. Neither matches
  `requirements-lock.txt`, which pins the pipeline stages rather than these two
  analyses.
- The comments and markdown of
  `analysis/bootstrap/single_model_bootstrap_selected_models_v2_seed42.ipynb`
  are in English.
- Every figure an analysis script writes is now byte-reproducible. matplotlib
  stamps a creation date into PDF metadata; suppressing it means re-running the
  scripts leaves the SHA-256 manifest intact.
- Added `.gitattributes`, which stops git from converting line endings. The
  manifest hashes the deposited bytes, and the tree is a deliberate mix: source
  and documentation are LF, while the CSV and JSON files are written by pandas
  and json on Windows and carry CRLF. Any conversion on checkout would break
  verification. Checked against a clone with `core.autocrlf` set to true.

## 1.2.2 - 2026-08-24

- `ane.physics._fluctuation_parameters` no longer applies a per-element
  `(M_i / M_bar)^2` weight. The single-lattice Klemens form without that factor
  is what Supplementary Note S1 describes and what reproduces the archived
  `kappa_L` of the campaign; the weighted form did not, missing the Fe-Ga
  binaries by more than 30 percent. The mass-ratio weight of the Abeles
  multi-sublattice treatment is a per-sublattice factor and does not apply
  here.
- `analysis/fig5/compute_disorder.py` and `analysis/fig5/README.md` no longer
  state that Note S1 carries the mass-ratio weights. Both files predate the
  change above and were still describing the v1.2.1 reconstruction. The numbers
  they produce are unaffected; the prose was the only thing that was wrong.
- `data/literature_reconstruction.csv` deposits the composition, reference,
  `S_ANE`, `kappa_L`, and `kappa_total` used for the 13 reconstructed literature
  entries.
- The 13 `kappa_total` values agree with the training labels in `data.csv` and
  the values in Supplementary Table S1.
- `tests/test_callaway_reconstruction.py` asserts that `kappa_lattice`
  reproduces the deposited `kappa_L` for all 13 entries; the complete suite
  contains 196 tests.

## 1.2.1 - 2026-08-24

- `CallawayParams` now defaults to the standard negative Umklapp exponent with
  gamma = 3.1192, the pair reported in Supplementary Note S1 and Table S6. The
  campaign ran the algebraically equivalent positive-exponent form with
  gamma = 2.0; that pair is retained as `LEGACY_UMKLAPP_SIGN` and
  `LEGACY_GAMMA` so the historical calculation stays runnable. Both give the
  same kappa_L at 300 K, and the released code no longer differs in sign
  convention from the supplementary text.
- `tests/test_callaway_reconstruction.py` pins the new defaults, the derivation
  of gamma from the legacy pair, and the equivalence of the two forms. The
  complete suite contains 198 tests.

## 1.2.0 - 2026-08-20

- Added `analysis/pca/`, which rebuilds the Fig. 3(d) biplot from
  `data/candidates.csv`. The projection was previously produced outside this
  repository and could not be reproduced from the released material.
- Recorded the two conventions the projection depends on: it runs on the eight
  nominal atomic fractions rather than the ILR coordinates, and the fractions
  are standardized to unit variance. Mean-centering alone gives 73.5% / 15.9%
  instead of the reported split.
- Added `tests/test_pca_reproduction.py`, which pins the published variance
  percentages, checks that every nominated composition lies on the 0.01
  enumeration grid, and fails if standardization stops mattering.
- Added `analysis/fig5/`, which computes the site-resolved disorder parameters
  of Fig. 5(d) and Supplementary Fig. S4 and writes `data/fig5_source_data.csv`.
  This calculation was also previously outside the repository.
- Recorded the two conventions that calculation depends on: the site sums carry
  no (M_i/M_s)^2 weight, and the elemental data are those of Supplementary
  Table S8(a) rather than the Callaway-Klemens set of Table S8(b). Swapping the
  radius table moves Gamma_V by an order of magnitude.
- Added `data/fig5_source_data.csv`, the numerical source data behind all four
  Fig. 5 panels, including the (220) FWHM values behind Fig. 5(c).
- Added `tests/test_disorder_reproduction.py`. The complete suite contains
  197 tests.

## 1.1.0 - 2026-08-14

- Reconciled the initial dataset to the authoritative 45 compositions and
  linked every initial row to Supplementary Table S1.
- Added the recovered cycle-1 36/9 train-test membership and made stage 0
  validate and reuse it instead of reconstructing a different partition.
- Added `--master-seed 42` to every pipeline stage. The standardized reanalysis
  uses stochastic root 42 throughout while preserving ensemble diversity as
  `42 + member_id`.
- Preserved the distinct historical campaign seeds in the cycle configuration
  files so that the prospective record is not rewritten retroactively.
- Added and executed the paired-bootstrap notebook with bootstrap root 42.
- Added compact nested-augmentation and conditional-baseline audit results.
- Added regression tests for the authoritative data, fixed split, and seed
  policy. The complete suite contains 185 tests.
