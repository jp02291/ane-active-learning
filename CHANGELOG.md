# Changelog

## Unreleased - campaign selection correction and completed nested evaluation

- Correct the account of the historical campaign: researchers manually
  reviewed held-out kappa and |S_ANE| MAEs together; both targets favored the
  same branch in each reported cycle. Internal CV tuned model parameters but
  did not automatically choose the campaign scenario.
- Remove CV sorting from the stage-2 scenario summary. Keep execution order,
  all tuning diagnostics, and the explicit cycle-1/3 augmented and cycle-2
  real-only configuration paths. No new scalar selection rule is introduced.
- Explain that legacy `test` files/fields refer to scenario-selection
  validation data in the campaign, and that bootstrap intervals do not correct
  for selection.
- Preserve campaign data, splits, metrics, model settings and the historical
  release verification. Replace the obsolete simplified nested outputs with
  the completed single-attempt, inner-fold-local evaluation, its frozen source,
  raw results and independently reproducible MAE/R2 summaries. Include all 15
  folds and failure fallback; exclude retry experiments. Retain the original
  ancillary Adaptive records only as raw provenance, not SI comparison rows.
- Regenerate the package manifest and add update verification. No training is
  rerun for packaging, and the archived Zenodo release is not modified.
- Fig. 6(d): `analysis/disorder/compute_disorder.py` now computes Gamma_M
  with Eq. (S4) of Supplementary Note S7, including the (M_s / M_avg)^2 weight
  on the site combination. The previous script omitted that weight and gave
  2.46, 4.71 and 6.04 x 1e-2 for x = 0.0625, 0.12 and 0.1875 instead of the
  plotted 2.67, 3.88 and 3.25 x 1e-2. The volume parameter Gamma_V, which the
  manuscript does not show, is removed. The source data, results, README and
  `tests/test_disorder_reproduction.py` are updated accordingly.
- `configs/cycle3.yaml`: `selection.diversity_distance` corrected from 0.1 to
  0.05, the value stated in Methods and consistent with the cycle-3
  nominations (smallest within-mode distance 0.051).

## 1.1.0 - 2026-09-29

Aligns the archive with the revised manuscript and supplementary information.
No campaign data, split or model result changes.

- Supplementary numbering follows the revised supplement throughout:
  acquisition comparison Fig. S2, surrogate benchmark Note S5 and Fig. S5,
  branch comparison Note S1, seed variability Note S2, literature
  thermal conductivity Note S4, transport data Table S3, elemental data
  Table S5, hyperparameters Table S8.
- `analysis/acquisition/plot_fig_s2.py` draws the two panels of Fig. S2. The
  dominated hypervolume of the same runs, which the Fig. S2 legend summarizes,
  is drawn separately as `results/hypervolume_robust.png`.
- `analysis/benchmark/plot_fig_s5.py` draws Fig. S5 as two panels, the MAE of
  kappa and of |S_ANE|, each with the cross-validation splits, their mean and
  standard deviation, and the held-out result.
- New `analysis/input_space/` computes the distances in the model input space
  quoted in the Discussion, with `tests/test_input_space_distance.py`.
- `analysis/disorder/` evaluates Gamma_M and Gamma_V for the Fe-Ga-Al series
  only, as stated in Methods. The Gamma cells of the Fe-Ga-Pt rows of
  `data/alloy_series_source_data.csv` are now empty, and the dependence of
  the Pt result on the unresolved site assignment is written to
  `results/pt_site_assignment_sensitivity.csv`.
- `tests/test_seed_variability.py` pins every entry of the Note S2 table for
  both targets. The cycle-2 real-only kappa MAE it uses is corrected from 2.141
  to 2.137, the value in the bootstrap summary and in Note S1, which moves the
  observed cycle-2 kappa difference from 28% to 29%.

## 1.0.0 - 2026-09-15

- First public release, accompanying the article "Discovery of
  high-sensitivity heat-flux sensor materials via active learning".
