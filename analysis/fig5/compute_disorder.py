"""Site-resolved alloy-disorder parameters for Fig. 5(d) and Supplementary Fig. S4.

Both figures plot the mass-disorder parameter Gamma_M and the volume-mismatch
parameter Gamma_V for the two model-selected alloy series. This script is the
calculation behind them, and it also emits the Fig. 5 source-data table.

The model treats Fe3X as two sublattices with fixed site fractions n_Fe = 0.75
and n_X = 0.25. Within a site,

    Gamma_s(P) = sum_{i in s} f_{i,s} (1 - P_i / P_s)^2

with f_{i,s} normalized inside the site, and the two sites combine as

    Gamma = n_Fe Gamma_Fe-site + n_X Gamma_X-site.

Two conventions are load-bearing, and they keep this calculation distinct from
the Callaway-Klemens reconstruction of Supplementary Note S1.

The site sums carry no (M_i / M_s)^2 weight and the site combination carries no
(M_s / M)^2 weight. Note S1 carries neither: it is the single-lattice
Klemens form Gamma = sum_i c_i (1 - P_i / P_bar)^2 evaluated over
the whole composition, with no sublattice structure for a site weight to attach
to. What separates the two calculations is the site resolution and the
within-site normalization of f_{i,s}, together with the radius set below, so
they are reported on their own scales rather than compared numerically.

The radii and masses are those of Supplementary Table S8(a), the descriptor
set, not the Callaway-Klemens set of Table S8(b). The choice matters: Ga and Al
are nearly the same size in S8(a) (1.408 and 1.429 A) and differ markedly in
S8(b) (1.53 and 1.43 A), which changes Gamma_V by an order of magnitude.
`tests/test_disorder_reproduction.py` pins both conventions.

Gamma_V uses r^3 as the volume proxy. The 4/3 pi prefactor cancels in the ratio
P_i / P_s, so it is omitted.

    python compute_disorder.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results"

#: Supplementary Table S8(a) -- the descriptor set
ATOMIC_MASS = {"Fe": 55.845, "Ga": 69.723, "Al": 26.9815385, "Pt": 195.084}
METALLIC_RADIUS_A = {"Fe": 1.26, "Ga": 1.408, "Al": 1.429, "Pt": 1.39}

N_FE_SITE, N_X_SITE = 0.75, 0.25

#: Site occupancies. For Fe0.74Ga0.24Pt0.02 the site preference of Pt is not
#: resolved by the measurements, so it is split across both sublattices.
SITE_MODELS: dict[str, dict[str, dict[str, float]]] = {
    "Fe0.75Ga0.25": {"Fe_site": {"Fe": 0.75}, "X_site": {"Ga": 0.25}},
    "Fe0.75Ga0.1875Al0.0625": {"Fe_site": {"Fe": 0.75},
                               "X_site": {"Ga": 0.1875, "Al": 0.0625}},
    "Fe0.75Ga0.13Al0.12": {"Fe_site": {"Fe": 0.75},
                           "X_site": {"Ga": 0.13, "Al": 0.12}},
    "Fe0.75Ga0.0625Al0.1875": {"Fe_site": {"Fe": 0.75},
                               "X_site": {"Ga": 0.0625, "Al": 0.1875}},
    "Fe0.75Al0.25": {"Fe_site": {"Fe": 0.75}, "X_site": {"Al": 0.25}},
    "Fe0.74Ga0.24Pt0.02": {"Fe_site": {"Fe": 0.74, "Pt": 0.01},
                           "X_site": {"Ga": 0.24, "Pt": 0.01}},
    "Fe0.75Ga0.125Pt0.125": {"Fe_site": {"Fe": 0.75},
                             "X_site": {"Ga": 0.125, "Pt": 0.125}},
    "Fe0.75Pt0.25": {"Fe_site": {"Fe": 0.75}, "X_site": {"Pt": 0.25}},
}

#: Supplementary Table S7, and the (220) FWHM behind Fig. 5(c). FWHM was
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


def site_gamma(site: dict[str, float], prop: dict[str, float]) -> float:
    """Gamma_s(P) for one sublattice; the site composition is normalized here."""
    total = sum(site.values())
    if total <= 0:
        raise ValueError("site composition must be positive")
    frac = {el: v / total for el, v in site.items()}
    p_bar = sum(f * prop[el] for el, f in frac.items())
    return sum(f * (1.0 - prop[el] / p_bar) ** 2 for el, f in frac.items())


def effective_gamma(sites: dict[str, dict[str, float]],
                    prop: dict[str, float]) -> float:
    return (N_FE_SITE * site_gamma(sites["Fe_site"], prop)
            + N_X_SITE * site_gamma(sites["X_site"], prop))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    volume = {el: r ** 3 for el, r in METALLIC_RADIUS_A.items()}

    rows = []
    for name, sites in SITE_MODELS.items():
        series, at_pct = SERIES[name]
        kappa, kappa_e, kappa_L, s_ane, fwhm = TRANSPORT[name]
        rows.append({
            "composition": name,
            "series": series,
            "substituent_at_percent": at_pct,
            "kappa_W_per_mK": kappa,
            "kappa_e_W_per_mK": kappa_e,
            "kappa_L_W_per_mK": kappa_L,
            "S_ANE_uV_per_K": s_ane,
            "FWHM_220_deg": "" if fwhm is None else fwhm,
            "Gamma_M": round(effective_gamma(sites, ATOMIC_MASS), 8),
            "Gamma_V": round(effective_gamma(sites, volume), 10),
        })

    src = REPO / "data" / "fig5_source_data.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "disorder_parameters.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["composition", "Gamma_M", "Gamma_V"],
            extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "site_fractions": {"Fe_site": N_FE_SITE, "X_site": N_X_SITE},
        "elemental_data": "Supplementary Table S8(a)",
        "mass_weight_inside_site": False,
        "mass_weight_on_site_combination": False,
        "volume_proxy": "r^3",
        "max_Gamma_M": max(r["Gamma_M"] for r in rows),
        "max_Gamma_V": max(r["Gamma_V"] for r in rows),
        "source_data_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
    (OUT / "disorder_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print(f"{'composition':<24}{'Gamma_M':>12}{'Gamma_V':>14}{'FWHM(220)':>11}")
    for r in rows:
        print(f"{r['composition']:<24}{r['Gamma_M']:12.6f}"
              f"{r['Gamma_V']:14.8f}{str(r['FWHM_220_deg']):>11}")
    print(f"\nwrote {src.relative_to(REPO)} and {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
