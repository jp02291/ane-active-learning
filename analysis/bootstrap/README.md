# Paired bootstrap for the reported single-model comparison

`single_model_bootstrap_selected_models_v2_seed42.ipynb` reproduces the
single-model MAEs and the paired percentile-bootstrap comparison used for
Fig. 2. The six deposited `test_predictions.csv` files preserve the paired
test-sample order for the real-only and augmented branches.

The point estimates do not depend on the bootstrap seed. This remediated run
uses 5,000 replicates with `BOOTSTRAP_SEED = 42`; the effective replicate seeds
are deterministic offsets of that root by cycle, target, and comparison.

Run the notebook from this directory so that its relative paths resolve:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  single_model_bootstrap_selected_models_v2_seed42.ipynb
```

The bootstrap quantifies uncertainty from the finite held-out test samples. It
does not quantify model-initialization, data-partition, generator, or
measurement uncertainty. Those limitations must remain in the figure caption
and supplementary text.
