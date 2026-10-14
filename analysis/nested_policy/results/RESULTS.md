# Single-attempt nested evaluation for the revised SI

All 15 folds, including generation failures. Values are mean +/- sample SD across three seed-level pooled OOF metrics.

| Target | Reference subset | Policy | MAE | R2 |
|---|---|---|---:|---:|
| SANE (uV/K) | all | real_only | 0.70002 +/- 0.11902 | 0.60744 +/- 0.10625 |
| kappa (W/(m K)) | measured | real_only | 3.11034 +/- 0.57981 | -0.04689 +/- 0.40177 |
| kappa (W/(m K)) | all | real_only | 2.69165 +/- 0.47694 | 0.07895 +/- 0.34510 |
| FoM (um/A) | measured | real_only | 0.05238 +/- 0.00791 | 0.41900 +/- 0.16879 |
| FoM (um/A) | all | real_only | 0.04568 +/- 0.00780 | 0.50021 +/- 0.14850 |
| SANE (uV/K) | all | always_wgan_with_fallback | 0.68872 +/- 0.01164 | 0.64108 +/- 0.01495 |
| kappa (W/(m K)) | measured | always_wgan_with_fallback | 2.90954 +/- 0.18483 | 0.12917 +/- 0.06112 |
| kappa (W/(m K)) | all | always_wgan_with_fallback | 2.64186 +/- 0.17480 | 0.20130 +/- 0.06140 |
| FoM (um/A) | measured | always_wgan_with_fallback | 0.04771 +/- 0.00525 | 0.50056 +/- 0.09890 |
| FoM (um/A) | all | always_wgan_with_fallback | 0.04437 +/- 0.00263 | 0.53890 +/- 0.06098 |

Final generation succeeded in 12/15 folds. Measured-kappa FoM: 9 wins, 3 losses, 3 fallback ties.

The primary FoM endpoint uses directly measured kappa. Reconstructed kappa is shown separately. The CSV also contains a matched generation-success-only summary; it is conditional, not the headline result.

This is a retrospective single-DNN evaluation. Repeated partitions are not independent datasets, and seed SD is not a confidence interval. It does not demonstrate general WGAN superiority or acquisition efficiency.
