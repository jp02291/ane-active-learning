# Seed policy

The prospective campaign and the standardized post-audit reanalysis must not be
described as if they were the same run.

| Component | Campaign record | Standardized reanalysis |
|---|---:|---:|
| target-informed split | 42 | 42 |
| WGAN-GP training | 42 | 42 |
| final filtered GAN pool | 2027 | 42 |
| Optuna TPE and five-fold CV | 42 | 42 |
| deep-ensemble member `m` | `2025 + m` | `42 + m` |
| candidate-grid sampling | 42 | 42 |
| paired-bootstrap root | 314159 in the earlier notebook | 42 in the remediated notebook |

The campaign configuration files retain the recorded values because changing
the generator pool or ensemble members can change the selected candidates.
Passing `--master-seed 42` to every pipeline stage creates the standardized
reanalysis. It does not rewrite the historical campaign.

Every ensemble member must not receive the identical literal seed 42. Members
need distinct initializations to define ensemble disagreement, so the
standardized rule is `42 + member_id`.
