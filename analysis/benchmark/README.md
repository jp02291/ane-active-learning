# Surrogate-model benchmark (Supplementary Note S2, Fig. S6)

Five regressors under one protocol, to show that the DNN surrogate is not an
arbitrary choice.

```bash
python run_benchmark.py                    # full search, hours
python run_benchmark.py --reuse-params results   # refit the deposited choice, minutes
python plot_fig_s6.py
```

## Protocol

Everything except the regressor is held fixed.

| | |
|---|---|
| models | DNN, kernel ridge (KRR), support vector (SVR), extreme gradient boosting (XGB), Gaussian process (GPR) |
| inputs | 15 dimensions from `ane.features`: 7 ILR coordinates plus 8 descriptors |
| data | `data/split/cycle1/`, the partition the campaign reported: 36 training, 9 held out |
| cross-validation | `RepeatedKFold`, 3 folds and 3 repeats, the same 9 splits for every model |
| search | Optuna TPE, 150 trials each, objective = target-scaled MSE over the 9 splits |
| scaling | `MinMaxScaler` fitted inside each training fold only |
| seed | 42 throughout |

The DNN search dominates the runtime: 150 trials take hours, the other four
take minutes each. `--reuse-params results` reads the deposited hyperparameters
and re-runs only the evaluation, which reproduces every number in the table
below in about two minutes.

## Results

MAE, mean over the nine cross-validation splits with one standard deviation,
and on the nine held-out samples.

**Thermal conductivity, W m⁻¹ K⁻¹**

| model | cross-validation | held-out |
|---|---|---|
| DNN | 2.371 ± 0.670 | 3.069 |
| KRR | 2.671 ± 0.342 | 2.260 |
| SVR | **2.198 ± 0.620** | 2.540 |
| XGB | 2.243 ± 0.560 | 2.953 |
| GPR | 2.566 ± 0.349 | **2.183** |

**|S_ANE|, μV K⁻¹**

| model | cross-validation | held-out |
|---|---|---|
| DNN | 0.836 ± 0.437 | 0.478 |
| KRR | 0.746 ± 0.235 | 0.436 |
| SVR | **0.657 ± 0.144** | **0.419** |
| XGB | 0.811 ± 0.119 | 0.628 |
| GPR | 0.691 ± 0.240 | 0.552 |

## Reading the table

**No model leads on both evaluations.** SVR has the lowest cross-validation
error for both targets. On the held-out set GPR is lowest for kappa and SVR for
|S_ANE|.

**The DNN is here for what it provides, not for its error.** It predicts both
targets jointly and supplies the deep ensemble whose spread becomes the
disagreement score U of the three-objective selection; none of the other four
does that without a separate construction. Its entries also come from a single
seeded model rather than the pruned ensemble, so they carry more run-to-run
variation than the other four.

**Neither column carries much weight on its own.** Cross-validation averages
nine partitions of 36 samples and the standard deviations span most of the gaps
between models; the held-out column is nine samples scored once. The partition
is also target informed (Algorithm S1), so these errors describe the bulk of
the composition space rather than the high-performance region -- equally for
all five models.

## Outputs

```
results/
├── best_params_<MODEL>.json      the selected hyperparameters and CV objective
├── all_best_params.json          the five collected into one file
├── comparison_protocol.json      the settings above, machine readable
├── cv_fold_metrics.csv           MAE, RMSE, R2 per model, split and target
├── cv_metrics_summary.csv        mean and SD over the nine splits
├── held_out_test_metrics.csv     MAE, RMSE, R2 on the nine held-out samples
├── held_out_test_predictions.csv per-sample predictions from all five models
├── fig_S6.png / fig_S6.pdf       Supplementary Fig. S6
└── model_comparison_cv.png       a diagnostic emitted by the run: the same
                                  cross-validation MAEs plus R2. Not Fig. S6.
```

## Environment

The deposited results were produced with Python 3.9.21, TensorFlow 2.20.0,
Optuna 4.9.0, xgboost 2.1.4, scikit-learn 1.6.1 and NumPy 2.0.2. That is not
the environment pinned in `requirements-lock.txt`, which fixes the pipeline
stages rather than this analysis, and every model here is sensitive to its
library versions at this sample size. Expect the MAEs to shift in the last
digits elsewhere, and the DNN entries -- a single seeded model -- to shift most.

`tests/test_benchmark_reproduction.py` pins the descriptor source, the
partition, and the published numbers.
