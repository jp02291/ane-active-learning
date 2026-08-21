# Disorder parameters and Fig. 5 source data

This script computes the site-resolved alloy-disorder parameters plotted in
Fig. 5(d) and in the lower panel of Supplementary Fig. S4, and writes the
numerical source data for Fig. 5.

```bash
python compute_disorder.py
```

## The model

Fe3X is treated as two sublattices with fixed site fractions n_Fe = 0.75 and
n_X = 0.25. Within a site,

    Gamma_s(P) = sum_{i in s} f_{i,s} (1 - P_i / P_s)^2

with f_{i,s} normalised inside the site, and the sites combine as

    Gamma = n_Fe Gamma_Fe-site + n_X Gamma_X-site

Gamma_M takes P = M, the atomic weight. Gamma_V takes the atomic volume, P = r^3
(the 4/3 pi prefactor cancels in the ratio).

For Fe0.74Ga0.24Pt0.02 the site preference of Pt is not resolved by the
measurements, so it is split as (Fe0.74Pt0.01)(Ga0.24Pt0.01). That composition
gives the largest Gamma_M and Gamma_V of either series.

## Two conventions that decide the numbers

This calculation is **not** the Callaway-Klemens reconstruction of
Supplementary Note S1, and differs from it in two ways.

There is no `(M_i / M_s)^2` weight inside the site sum, and no `(M_s / M)^2`
weight on the site combination. Note S1 carries both. The two analyses are
therefore reported on their own scales and are not compared numerically.

The radii and masses are those of **Supplementary Table S8(a)**, the descriptor
set — not the Callaway-Klemens set of Table S8(b). This matters more than it
looks: Ga and Al are nearly the same size in S8(a) (1.408 and 1.429 A) and
differ markedly in S8(b) (1.53 and 1.43 A), which moves Gamma_V by an order of
magnitude and shifts where it peaks.

`tests/test_disorder_reproduction.py` pins both conventions, so an edit that
silently reintroduces the weights or swaps the radius table will fail the suite.

## Outputs

| File | Contents |
| --- | --- |
| `../../data/fig5_source_data.csv` | numerical source data for all four Fig. 5 panels |
| `results/disorder_parameters.csv` | Gamma_M and Gamma_V per composition |
| `results/disorder_summary.json` | conventions used, extrema, input hash |

`fig5_source_data.csv` collects kappa, kappa_e, kappa_L and |S_ANE| (as in
Supplementary Table S7), the (220) FWHM behind Fig. 5(c), and the disorder
parameters computed here. FWHM was measured for six of the eight compositions;
the two Fe-Ga-Al intermediates are left blank.
