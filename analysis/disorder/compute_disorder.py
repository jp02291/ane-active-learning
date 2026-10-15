"""Mass-fluctuation parameter for Fig. 6(d), and the Fig. 6 source data.

Fig. 6(d) plots the mass-fluctuation parameter Gamma_M for the
Fe0.75Ga0.25-xAlx series. This script is the calculation behind that panel, and
it also emits the source-data table for all four panels of Fig. 6.

Gamma_M follows Eq. (S4) of Supplementary Note S7, the multi-sublattice
expression of Yang et al. Fe3X is treated as two sublattices with relative
degeneracies c_Fe = 3 and c_X = 1, used here as the normalized site fractions
n_Fe = 0.75 and n_X = 0.25. Within a site,

    Gamma_s = sum_{k in s} f_{k,s} (1 - M_k / M_s)^2,   M_s = sum_k f_{k,s} M_k

with f_{k,s} normalized inside the site, and the sites combine as

    Gamma_M = sum_s n_s (M_s / M_avg)^2 Gamma_s,   M_avg = sum_s n_s M_s.

The (M_s / M_avg)^2 weight on the site combination is part of Eq. (S4). There
is no further weight inside the site sum.

The Fe-Ga-Pt series is not evaluated, as stated in Supplementary Note S7. With
an Fe fraction of 0.74, the result for Fe0.74Ga0.24Pt0.02 depends on which
element fills the remaining 0.01 of the Fe sublattice, an amount comparable to
the compositional accuracy of the arc-melted specimens. The script writes
Gamma_M under three assignments to `results/pt_site_assignment_sensitivity.csv`
and leaves the Gamma_M cells of the Fe-Ga-Pt rows empty.

This calculation is distinct from the Callaway-Klemens reconstruction of
Methods and Supplementary Note S4, which uses the single-lattice Klemens form
Gamma = sum_i c_i (1 - M_i / M_bar)^2 over the whole composition. The two are
reported on their own scales rather than compared numerically.

The atomic masses are those of Supplementary Table S5(a).
`tests/test_disorder_reproduction.py` pins the convention and the plotted values.

    python compute_disorder.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results"

#: Supplementary Table S5(a)
ATOMIC_MASS = {"Fe": 55.845, "Ga": 69.723, "Al": 26.9815385, "Pt": 195.084}

N_FE_SITE, N_X_SITE = 0.75, 0.25

#: Site occupancies of the compositions that are evaluated: the Fe-Ga-Al series
#: and the shared Fe0.75Ga0.25 endpoint. Ga and Al share the X sublattice.
SITE_MODELS: dict[str, dict[str, dict[str, float]]] = {
    "Fe0.75Ga0.25": {"Fe_site": {"Fe": 0.75}, "X_site": {"Ga": 0.25}},
    "Fe0.75Ga0.1875Al0.0625": {"Fe_site": {"Fe": 0.75},
                               "X_site": {"Ga": 0.1875, "Al": 0.0625}},
    "Fe0.75Ga0.13Al0.12": {"Fe_site": {"Fe": 0.75},
                           "X_site": {"Ga": 0.13, "Al": 0.12}},
    "Fe0.75Ga0.0625Al0.1875": {"Fe_site": {"Fe": 0.75},
                               "X_site": {"Ga": 0.0625, "Al": 0.1875}},
    "Fe0.75Al0.25": {"Fe_site": {"Fe": 0.75}, "X_site": {"Al": 0.25}},
}

#: Three ways to place the 0.01 Fe-site deficit of Fe0.74Ga0.24Pt0.02. The
#: measurements do not distinguish them, which is why the series is not
#: evaluated; Gamma_M under each assignment is written out for reference.
PT_ASSIGNMENTS: dict[str, dict[str, dict[str, float]]] = {
    "Pt fills the Fe-site deficit": {"Fe_site": {"Fe": 0.74, "Pt": 0.01},
                                     "X_site": {"Ga": 0.24, "Pt": 0.01}},
    "Ga fills the Fe-site deficit": {"Fe_site": {"Fe": 0.74, "Ga": 0.01},
                                     "X_site": {"Ga": 0.23, "Pt": 0.02}},
    "Fe-site deficit left vacant": {"Fe_site": {"Fe": 0.74},
                                    "X_site": {"Ga": 0.24, "Pt": 0.02}},
}

#: Supplementary Table S3, and the (220) FWHM behind Fig. 6(c). FWHM was
#: measured for six of the eight compositions; None marks the two that were not.
TRANSPORT = {
    #                          kappa   kappa_e kappa_L S_ANE  FWHM(220) deg
    "Fe0.75Ga0.25":           (19.48,  6.90,  12.58,  5.88,  0.51572),
    "Fe0.75Ga0.1875Al0.0625": (17.97,  6.56,  11.41,  5.60,  None),
    "Fe0.75Ga0.13Al0.12":     (15.04,  7.47,   7.57,  5.40,  0.43471),
    "Fe0.75Ga0.0625Al0.1875": (15.52,  7.13,   8.39,  4.82,  None),
    "Fe0.75Al0.25":           (18.30,  7.64,  10.66,  4.00,  0.66229),
    "Fe0.74Ga0.24Pt0.02":     (14.98, 10.39,   4.59,  5.29,  0.73459),
    "Fe0.75Ga0.125Pt0.125":   (17.63, 12.13,   5.50,  2.96,  0.59070),
    "Fe0.75Pt0.25":           (20.43, 11.00,   9.43,  2.98,  0.59156),
}

SERIES = {
    "Fe0.75Ga0.25": ("both", 0.0),
    "Fe0.75Ga0.1875Al0.0625": ("Fe-Ga-Al", 6.25),
    "Fe0.75Ga0.13Al0.12": ("Fe-Ga-Al", 12.0),
    "Fe0.75Ga0.0625Al0.1875": ("Fe-Ga-Al", 18.75),
    "Fe0.75Al0.25": ("Fe-Ga-Al", 25.0),
    "Fe0.74Ga0.24Pt0.02": ("Fe-Ga-Pt", 2.0),
    "Fe0.75Ga0.125Pt0.125": ("Fe-Ga-Pt", 12.5),
    "Fe0.75Pt0.25": ("Fe-Ga-Pt", 25.0),
}


def site_mean(site: dict[str, float], prop: dict[str, float]) -> float:
    """Occupancy-weighted mean of `prop` on one sublattice."""
    total = sum(site.values())
    if total <= 0:
        raise ValueError("site composition must be positive")
    return sum(v / total * prop[el] for el, v in site.items())


def site_gamma(site: dict[str, float], prop: dict[str, float]) -> float:
    """Gamma_s for one sublattice; the site composition is normalized here."""
    total = sum(site.values())
    frac = {el: v / total for el, v in site.items()}
    p_bar = site_mean(site, prop)
    return sum(f * (1.0 - prop[el] / p_bar) ** 2 for el, f in frac.items())


def effective_gamma(sites: dict[str, dict[str, float]],
                    prop: dict[str, float]) -> float:
    """Eq. (S4): site terms weighted by the site fractions and (M_s / M_avg)^2."""
    weights = {"Fe_site": N_FE_SITE, "X_site": N_X_SITE}
    means = {s: site_mean(sites[s], prop) for s in weights}
    p_avg = sum(weights[s] * means[s] for s in weights)
    return sum(weights[s] * (means[s] / p_avg) ** 2 * site_gamma(sites[s], prop)
               for s in weights)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in TRANSPORT:
        series, at_pct = SERIES[name]
        kappa, kappa_e, kappa_L, s_ane, fwhm = TRANSPORT[name]
        sites = SITE_MODELS.get(name)
        rows.append({
            "composition": name,
            "series": series,
            "substituent_at_percent": at_pct,
            "kappa_W_per_mK": kappa,
            "kappa_e_W_per_mK": kappa_e,
            "kappa_L_W_per_mK": kappa_L,
            "S_ANE_uV_per_K": s_ane,
            "FWHM_220_deg": "" if fwhm is None else fwhm,
            "Gamma_M": "" if sites is None else round(effective_gamma(sites, ATOMIC_MASS), 8),
        })

    src = REPO / "data" / "alloy_series_source_data.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "disorder_parameters.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["composition", "Gamma_M"],
            extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(r for r in rows if r["Gamma_M"] != "")

    sensitivity = [
        {"assignment": label,
         "Gamma_M": round(effective_gamma(sites, ATOMIC_MASS), 8)}
        for label, sites in PT_ASSIGNMENTS.items()
    ]
    with open(OUT / "pt_site_assignment_sensitivity.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["assignment", "Gamma_M"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(sensitivity)

    evaluated = [r for r in rows if r["Gamma_M"] != ""]
    summary = {
        "site_fractions": {"Fe_site": N_FE_SITE, "X_site": N_X_SITE},
        "elemental_data": "Supplementary Table S5(a)",
        "evaluated_series": "Fe0.75Ga0.25-xAlx",
        "fe_ga_pt_series": "not evaluated; see pt_site_assignment_sensitivity.csv",
        "formula": "Supplementary Note S7, Eq. (S4)",
        "mass_weight_inside_site": False,
        "mass_weight_on_site_combination": True,
        "max_Gamma_M": max(r["Gamma_M"] for r in evaluated),
        "argmax_Gamma_M": max(evaluated, key=lambda r: r["Gamma_M"])["composition"],
        "source_data_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
    (OUT / "disorder_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print(f"{'composition':<24}{'Gamma_M':>12}{'FWHM(220)':>11}")
    for r in rows:
        print(f"{r['composition']:<24}{str(r['Gamma_M']):>12}"
              f"{str(r['FWHM_220_deg']):>11}")
    print(f"\nwrote {src.relative_to(REPO)} and {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
