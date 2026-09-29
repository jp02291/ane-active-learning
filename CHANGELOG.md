# Changelog

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
