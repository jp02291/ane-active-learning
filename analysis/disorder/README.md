# Disorder parameters and Fig. 6 source data

This script computes the site-resolved alloy-disorder parameters plotted in
Fig. 6(d) for the Fe0.75Ga0.25-xAlx series, and writes the numerical source
data for all four panels of Fig. 6.

```bash
python compute_disorder.py
```

## The model

Fe3X is treated as two sublattices with fixed site fractions n_Fe = 0.75 and
n_X = 0.25. Within a site,

    Gamma_s(P) = sum_{i in s} f_{i,s} (1 - P_i / P_s)^2

with f_{i,s} normalized inside the site, and the sites combine as

    Gamma = n_Fe Gamma_Fe-site + n_X Gamma_X-site

Gamma_M takes P = M, the atomic weight. Gamma_V takes the atomic volume, P = r^3
(the 4/3 pi prefactor cancels in the ratio). Ga and Al share the X sublattice.

## Why the Fe-Ga-Pt series is not evaluated

As stated in Methods, the Fe-Ga-Pt series is not evaluated. Fe0.74Ga0.24Pt0.02
has an Fe fraction of 0.74, so 0.01 of the Fe sublattice must be filled by Pt,
by Ga, or left vacant, and the measurements do not distinguish these cases. The
difference is comparable to the compositional accuracy of the arc-melted
specimens (Supplementary Note S3), but it decides the answer:

| assignment of the 0.01 Fe-site deficit | Gamma_M | Gamma_V |
|---|---|---|
| Pt fills it | 0.0845 | 1.16e-3 |
| Ga fills it | 0.0461 | 1.55e-3 |
| left vacant | 0.0443 | 2.56e-5 |

Gamma_V changes by a factor of about 60 across assignments that the data cannot
separate. The Gamma columns of the Fe-Ga-Pt rows of the source data are
therefore left empty, and the table above is written to
`results/pt_site_assignment_sensitivity.csv`.

## Two conventions that decide the numbers

This calculation is **not** the Callaway-Klemens reconstruction of Methods and
Supplementary Note S4.

There is no `(M_i / M_s)^2` weight inside the site sum, and no `(M_s / M)^2`
weight on the site combination. The Callaway-Klemens reconstruction carries
neither: it is the single-lattice Klemens form
`Gamma = sum_i c_i (1 - P_i / P_bar)^2` over the whole composition, with no
sublattice structure for a site weight to attach to. The two analyses differ in
the site resolution and the within-site normalization, and in the radius set
below, so they are reported on their own scales and are not compared
numerically.

The radii and masses are those of **Supplementary Table S5(a)**, the descriptor
set -- not the Callaway-Klemens set of Table S5(b). This matters more than it
looks: Ga and Al are nearly the same size in S5(a) (1.408 and 1.429 A) and
differ markedly in S5(b) (1.53 and 1.43 A), which moves Gamma_V by an order of
magnitude and shifts where it peaks.

`tests/test_disorder_reproduction.py` pins both conventions, the omission of the
Fe-Ga-Pt series and its reason, and the extrema shown in Fig. 6(d): Gamma_V
peaks at Fe0.75Ga0.13Al0.12, the composition with the lowest kappa_L, and
Gamma_M at x = 0.1875.

## Outputs

| File | Contents |
| --- | --- |
| `../../data/alloy_series_source_data.csv` | numerical source data for all four Fig. 6 panels |
| `results/disorder_parameters.csv` | Gamma_M and Gamma_V for the evaluated compositions |
| `results/pt_site_assignment_sensitivity.csv` | Gamma of Fe0.74Ga0.24Pt0.02 under the three assignments above |
| `results/disorder_summary.json` | conventions used, extrema, input hash |

`alloy_series_source_data.csv` collects kappa, kappa_e, kappa_L and |S_ANE|
(as in Supplementary Table S3), the (220) FWHM behind Fig. 6(c), and the
disorder parameters computed here. FWHM was measured for six of the eight
compositions; the two Fe-Ga-Al intermediates are left blank.
