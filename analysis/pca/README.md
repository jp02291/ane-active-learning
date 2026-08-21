# Principal component analysis of the nominated compositions

This script rebuilds the biplot of Fig. 3(d): the 30 compositions nominated
over the three active-learning cycles, projected onto their first two principal
components.

```bash
python run_pca.py
```

Input is `data/candidates.csv`, which carries the cycle index and the
exploitation/exploration label alongside the eight atomic fractions, so no
separate per-cycle file is needed.

Two conventions decide what the figure shows.

The analysis runs on the eight **nominal** atomic fractions rather than on the
ILR coordinates used as surrogate inputs, so that the loading directions read
directly as element contents. The fractions are then **standardized to zero mean
and unit variance**. Mean-centering alone gives a different projection — Fe and
Ga carry the largest raw spread and would dominate — so the reported variance
split is specific to the standardized run. `tests/test_pca_reproduction.py` pins
both the published percentages and the fact that the two preprocessing choices
disagree.

The loading arrows are drawn with a common display scaling. Their directions are
meaningful; their absolute lengths are not. `results/pca_loadings.csv` stores the
unscaled `components_`.

## Outputs

| File | Contents |
| --- | --- |
| `results/pca_scores.csv` | PC1/PC2 per composition, with cycle and selection type |
| `results/pca_loadings.csv` | unscaled PC1/PC2 loadings for the eight elements |
| `results/pca_biplot.png` | reference rendering of Fig. 3(d) |
| `results/pca_summary.json` | variance split, preprocessing, input SHA256 |

The published figure was redrawn in external plotting software from
`pca_scores.csv` and `pca_loadings.csv`; `pca_biplot.png` is the reference
rendering produced directly by this script.
