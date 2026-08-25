# Independent pre-submission audit results

These files are the compact, machine-readable outputs of the authoritative-45
remediation audit. The complete run journals, cached Optuna studies, generation
artifacts, and scripts are distributed separately in
`ANE_round2_45_nested_baseline_package.zip`.

## Nested augmentation comparison

The outer validation rows were sealed from generator training, filtering, HPO,
and branch selection. Three root seeds and five outer folds were attempted.
Fourteen of fifteen fold jobs produced a valid paired comparison; seed 42,
fold 4 failed the predefined requirement of 200 unique accepted synthetic
samples (112 were produced) and was retained as a generation failure.

Across the 14 paired folds, real-only had the lower scaled MSE in 10 and
augmentation in 4. The mean paired delta,
`MSE_augmented - MSE_real`, was +0.00456. Seed 42 alone had a negative mean
delta over its four successful folds, but that mean was dominated by fold 2
and did not establish fold-wise or seed-wise robustness.

## Conditional random/diversity baselines

The baseline comparison operates only on the 25 candidates that the campaign
actually measured. It therefore tests prioritization within a campaign-selected
pool, not full-space superiority. At budgets 10 and 20, the observed acquisition
order did not outperform the random distribution on the prespecified best-FOM
or hypervolume summaries. The result supports a conditional sensitivity
statement, not a causal claim that random search would have found the same
materials in the full composition space.

## Claim boundary

These results support describing WGAN-GP augmentation as an optional,
cycle/fold-dependent modeling component. They do not support claiming robust
augmentation superiority. They also do not support claiming that the observed
candidate order was significantly more sample-efficient than random or
diversity-only search.
