# Acquisition-rule comparison (Supplementary Fig. S5)

This script replays the campaign on the measured pool and asks how quickly each
of four acquisition rules would have reached the best composition in it. It
produces the numbers behind Supplementary Fig. S5 and the repeat reported in
Supplementary Note S1.

```bash
python run_benchmark.py
python run_benchmark.py --only batch1
```

## The rules

All four run on the same Gaussian-process surrogate and the same
15-dimensional inputs, so the comparison isolates the selection rule.

| key | rule |
|---|---|
| `gp_pareto_unc` | the rule used in this work: a non-dominated filter in (\|S_ANE\|, 1/kappa, U), then ranking on \|S_ANE\|/kappa for the exploitation half and on U for the exploration half |
| `gp_ratio_ucb` | upper-confidence bound on the scalar \|S_ANE\|/kappa, the closest single-objective analogue of Algorithm S5 |
| `gp_ehvi` | expected hypervolume improvement over the observed front |
| `random` | uniform sampling without replacement |

kappa is modeled on a log scale so the posterior cannot place mass on
non-positive thermal conductivity, which would make 1/kappa diverge.

## The three configurations

| name | starting data | batch | cycles | reported in |
|---|---|---|---|---|
| `batch1` | all 45 cycle-0 rows | 1 | 25 | Fig. S5(a) |
| `robust` | a fresh random 80% of the cycle-0 rows per repetition | 5 | 5 | Fig. S5(b, c) |
| `measured_only` | the 32 measured cycle-0 rows | 1 | 25 | Note S1 |

Each runs 50 repetitions. Two of the rules are deterministic once the starting
set is fixed: `gp_ratio_ucb` and `gp_pareto_unc` rank the pool from the
posterior alone, so at `cycle0_subsample = 1.0` all 50 repetitions follow the
same path. `gp_ehvi` draws Monte-Carlo samples from the posterior and `random`
samples the pool, so both vary even from a fixed start. Lowering the fraction
to 0.8 in `robust` makes all four vary through the data as well, which is what
produces the bands in panel (c). The same subset is used by every rule at a given repetition, so
the comparison stays paired, and compositions acquired during the campaign are
never placed in the starting set.

## Acquisitions needed to reach the pool optimum

Median over the 50 repetitions, counted from the starting set.

| rule | `batch1` | `robust` | `measured_only` |
|---|---|---|---|
| `gp_pareto_unc` | **2** | **10** | **4** |
| `gp_ratio_ucb` | 15 | 20 | 9 |
| `gp_ehvi` | 15 | 25 | 10 |
| `random` | 8.5 | 15 | 8.5 |

Under `robust`, the repetitions that reached the pool optimum at all were
50/50, 43/50, 30/50 and 35/50 in the order above.

Random is identical in `batch1` and `measured_only` by construction: it does
not use the surrogate, so removing the 13 reconstructed entries from the
training data cannot affect it. That the two columns agree exactly is a
consistency check on the harness, not a result.

## Two things this does not establish

**The pool is not neutral ground.** It consists of the compositions this
campaign measured, and the campaign selected them with `gp_pareto_unc`. The
comparison therefore reports how the rules ordered a fixed set that one of them
assembled. It is not a general ranking of acquisition functions, and
Supplementary Fig. S5 states that bound explicitly.

**The pool contains the best composition among only 25 candidates.** With a
target that dense, the number of acquisitions needed is a coarse measure and
small differences between rules should not be read as a performance gap.

## Inputs

The pool is `data/data.csv`, the 70-composition dataset, with `cycle_added`
marking the 45 initial entries as cycle 0 and `kappa_source` marking the 13
entries whose kappa was reconstructed rather than measured.

Features come from `ane.features.featurize`, so this baseline and the campaign
surrogate see identical inputs: the same ILR construction, the same element
order, and the element properties of Supplementary Table S8(a). That shared
source is load-bearing: a separate element table here would change the
surrogate and every number above without changing anything a reader can see, so
`tests/test_acquisition_reproduction.py` pins where the descriptors come from.

## Outputs

```
results/
├── benchmark_batch1.csv         one row per rule, repetition and budget
├── benchmark_robust.csv
├── benchmark_measured_only.csv
└── acquisition_summary.json     inputs, SHA-256, configurations, and the
                                 acquisitions-to-optimum table above
```

Each CSV carries `strategy`, `seed`, `n_exp`, `best_ratio` (the best
\|S_ANE\|/kappa characterized so far) and `hv` (the dominated hypervolume of the
characterized set), which is everything the three panels plot.

## Environment

The deposited results were produced with Python 3.12.3, scikit-learn 1.7.2,
NumPy 1.26.4 and SciPy 1.13.1. That is not the environment pinned in
`requirements-lock.txt`, which fixes the pipeline stages rather than this
analysis. The Gaussian process is fitted by scikit-learn, so a different
scikit-learn can move the acquisition counts by an experiment or two; the
ordering of the rules is not that fragile.

`tests/test_acquisition_reproduction.py` pins the descriptor source, the pool,
and the published acquisition counts.
