# Changelog

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
  enumeration grid, and fails if standardization stops mattering. The complete
  suite contains 190 tests.

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
