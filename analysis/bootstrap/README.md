# Paired bootstrap for the reported single-model comparison

`single_model_bootstrap_selected_models_seed42.ipynb` reproduces the
single-model MAEs and the paired percentile-bootstrap comparison used for
Fig. 3. The six deposited `test_predictions.csv` files preserve the paired
held-out sample order for the real-only and augmented branches. These files
retain their legacy names; the campaign used this held-out set for manual
scenario selection, so it is a validation set rather than an independent test.

The point estimates do not depend on the bootstrap seed. The run uses 5,000
replicates with `BOOTSTRAP_SEED = 42`; the effective replicate seeds are
deterministic offsets of that root by cycle, target, and comparison.

Run the notebook from this directory so that its relative paths resolve:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  single_model_bootstrap_selected_models_seed42.ipynb
```

The bootstrap describes resampling variability on the finite held-out
scenario-selection validation set. It does not account for scenario selection,
model-initialization, data-partition, generator, or measurement uncertainty. Those limitations must remain in the figure caption
and supplementary text.
