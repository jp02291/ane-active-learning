# Mass-fluctuation parameter and Fig. 6 source data

This script computes the mass-fluctuation parameter Gamma_M plotted in
Fig. 6(d) for the Fe0.75Ga0.25-xAlx series, and writes the numerical source
data for all four panels of Fig. 6.

```bash
python compute_disorder.py
```

## The model

Gamma_M follows Eq. (S4) of Supplementary Note S7, the multi-sublattice
expression of Yang et al. Fe3X is treated as two sublattices with relative
degeneracies c_Fe = 3 and c_X = 1, used as the normalized site fractions
n_Fe = 0.75 and n_X = 0.25. Within a site,

    Gamma_s = sum_{k in s} f_{k,s} (1 - M_k / M_s)^2,   M_s = sum_k f_{k,s} M_k

with f_{k,s} normalized inside the site, and the sites combine as

    Gamma_M = sum_s n_s (M_s / M_avg)^2 Gamma_s,   M_avg = sum_s n_s M_s

Ga and Al share the X sublattice, so only the X sublattice contributes for the
Fe0.75Ga0.25-xAlx series. The atomic masses are those of Supplementary
Table S5(a).

| composition | Gamma_M (1e-2) |
|---|---|
| Fe0.75Ga0.25 | 0 |
| Fe0.75Ga0.1875Al0.0625 | 2.67 |
| Fe0.75Ga0.13Al0.12 | 3.88 |
| Fe0.75Ga0.0625Al0.1875 | 3.25 |
| Fe0.75Al0.25 | 0 |

Gamma_M peaks at Fe0.75Ga0.13Al0.12, the composition with the lowest kappa_L,
and the two largest values (x = 0.12 and 0.1875) coincide with the two lowest
kappa_L values of the series.

## Why the Fe-Ga-Pt series is not evaluated

As stated in Supplementary Note S7, the Fe-Ga-Pt series is not evaluated.
Fe0.74Ga0.24Pt0.02 has an Fe fraction of 0.74, so 0.01 of the Fe sublattice
must be filled by Pt, by Ga, or left vacant, and the measurements do not
distinguish these cases. The amount is comparable to the compositional accuracy
of the arc-melted specimens (Supplementary Note S3). For reference, Eq. (S4)
gives

| assignment of the 0.01 Fe-site deficit | Gamma_M |
|---|---|
| Pt fills it | 0.0891 |
| Ga fills it | 0.0758 |
| left vacant | 0.0732 |

The Gamma_M cells of the Fe-Ga-Pt rows of the source data are left empty, and
the table above is written to `results/pt_site_assignment_sensitivity.csv`.

## Relation to the Callaway-Klemens reconstruction

This calculation is **not** the Callaway-Klemens reconstruction of Methods and
Supplementary Note S4. That reconstruction uses the single-lattice Klemens form
`Gamma = sum_i c_i (1 - M_i / M_bar)^2` over the whole composition, with no
sublattice structure, and the atomic data of Table S5(b). The two analyses are
reported on their own scales and are not compared numerically.

`tests/test_disorder_reproduction.py` pins the Eq. (S4) weighting, the Table
S5(a) masses, the values plotted in Fig. 6(d), and the omission of the Fe-Ga-Pt
series.

## Outputs

| File | Contents |
| --- | --- |
| `../../data/alloy_series_source_data.csv` | numerical source data for all four Fig. 6 panels |
| `results/disorder_parameters.csv` | Gamma_M for the evaluated compositions |
| `results/pt_site_assignment_sensitivity.csv` | Gamma_M of Fe0.74Ga0.24Pt0.02 under the three assignments above |
| `results/disorder_summary.json` | conventions used, maximum, input hash |

`alloy_series_source_data.csv` collects kappa, kappa_e, kappa_L and |S_ANE|
(as in Supplementary Table S3), the (220) FWHM behind Fig. 6(c), and Gamma_M
computed here. FWHM was measured for six of the eight compositions; the two
Fe-Ga-Al intermediates are left blank.
