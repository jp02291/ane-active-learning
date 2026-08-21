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
