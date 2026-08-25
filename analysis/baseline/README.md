# Conditional random and diversity-only baseline

This script compares the observed acquisition order with random sampling
without replacement and sequential maximin diversity on the same 25 candidates
that were experimentally measured. The randomization root is 42.

```bash
python run_baselines.py
```

Because the 25-candidate universe was itself selected by the active-learning
campaign, this is a conditional prioritization audit. It cannot establish what
random search would have found in the full unlabeled composition space.
