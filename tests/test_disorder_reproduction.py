"""The Fig. 6(d) mass-fluctuation parameter must stay reproducible, and stay
distinct from the Callaway-Klemens reconstruction of Methods and Supplementary
Note S4.

Gamma_M follows Eq. (S4) of Supplementary Note S7: the site sums carry no mass
weight, the site combination carries the (M_s / M_avg)^2 weight, and the atomic
masses come from Supplementary Table S5(a). The values plotted in Fig. 6(d) are
pinned here alongside the qualitative correspondence the main text draws
between Gamma_M, the (220) FWHM and the lattice thermal conductivity. The
Fe-Ga-Pt series is not evaluated (Supplementary Note S7), and the tests pin that
omission.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "disorder" / "compute_disorder.py"
SOURCE_DATA = ROOT / "data" / "alloy_series_source_data.csv"

BEST_PT = "Fe0.74Ga0.24Pt0.02"
BEST_AL = "Fe0.75Ga0.13Al0.12"
REFERENCE = "Fe0.75Ga0.25"

#: Gamma_M of Fig. 6(d), in units of 1e-2
FIG6D_GAMMA_M = {
    "Fe0.75Ga0.1875Al0.0625": 2.67,
    "Fe0.75Ga0.13Al0.12": 3.88,
    "Fe0.75Ga0.0625Al0.1875": 3.25,
}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compute_disorder", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rows() -> dict[str, dict[str, str]]:
    with open(SOURCE_DATA, newline="", encoding="utf-8") as f:
        return {r["composition"]: r for r in csv.DictReader(f)}


def test_uses_the_table_s5a_masses(mod) -> None:
    """Supplementary Table S5(a), to the three decimals the table prints."""
    expected = {"Fe": 55.845, "Ga": 69.723, "Al": 26.982, "Pt": 195.084}
    for el, m in expected.items():
        assert mod.ATOMIC_MASS[el] == pytest.approx(m, abs=5e-4)


def test_site_weight_follows_eq_s4(mod) -> None:
    """No weight inside a site, (M_s / M_avg)^2 on the site combination."""
    m = mod.ATOMIC_MASS
    site = {"Ga": 0.5, "Al": 0.5}
    m_bar = 0.5 * m["Ga"] + 0.5 * m["Al"]
    unweighted = sum(0.5 * (1 - m[e] / m_bar) ** 2 for e in site)
    assert mod.site_gamma(site, m) == pytest.approx(unweighted)

    sites = {"Fe_site": {"Fe": 0.75}, "X_site": {"Ga": 0.125, "Al": 0.125}}
    m_avg = 0.75 * m["Fe"] + 0.25 * m_bar
    eq_s4 = 0.25 * (m_bar / m_avg) ** 2 * unweighted
    assert mod.effective_gamma(sites, m) == pytest.approx(eq_s4)
    assert mod.effective_gamma(sites, m) != pytest.approx(0.25 * unweighted)


def test_endmembers_have_zero_disorder(mod, rows) -> None:
    """A single occupant on each site cannot fluctuate."""
    for comp in (REFERENCE, "Fe0.75Al0.25"):
        assert float(rows[comp]["Gamma_M"]) == pytest.approx(0.0, abs=1e-12)


def test_fe_ga_pt_series_is_not_evaluated(rows) -> None:
    """Supplementary Note S7: the Fe-Ga-Pt series was not evaluated."""
    for comp, r in rows.items():
        if r["series"] == "Fe-Ga-Pt":
            assert r["Gamma_M"] == "", comp


def test_pt_site_assignments_are_recorded(mod) -> None:
    """Gamma_M of Fe0.74Ga0.24Pt0.02 under the three placements of the 0.01
    Fe-site deficit, written out for reference."""
    gm = {k: mod.effective_gamma(s, mod.ATOMIC_MASS)
          for k, s in mod.PT_ASSIGNMENTS.items()}
    assert gm["Pt fills the Fe-site deficit"] == pytest.approx(0.0891, abs=5e-4)
    assert gm["Ga fills the Fe-site deficit"] == pytest.approx(0.0758, abs=5e-4)
    assert gm["Fe-site deficit left vacant"] == pytest.approx(0.0732, abs=5e-4)


def test_al_series_matches_fig6d(rows) -> None:
    """The plotted values, and Gamma_M peaking at the lowest-kappa_L composition."""
    al = {c: r for c, r in rows.items() if r["Gamma_M"] != ""}
    gm = {c: float(r["Gamma_M"]) for c, r in al.items()}
    kl = {c: float(r["kappa_L_W_per_mK"]) for c, r in al.items()}
    for comp, value in FIG6D_GAMMA_M.items():
        assert gm[comp] * 100 == pytest.approx(value, abs=0.005), comp
    assert max(gm, key=gm.get) == BEST_AL == min(kl, key=kl.get)
    two_lowest_kl = sorted(kl, key=kl.get)[:2]
    two_largest_gm = sorted(gm, key=gm.get, reverse=True)[:2]
    assert set(two_lowest_kl) == set(two_largest_gm)


def test_fwhm_maximum_coincides_with_lattice_conductivity_minimum(rows) -> None:
    """The correspondence the main text draws for the Fe-Ga-Pt series."""
    pt = [REFERENCE, BEST_PT, "Fe0.75Ga0.125Pt0.125", "Fe0.75Pt0.25"]
    fwhm = {c: float(rows[c]["FWHM_220_deg"]) for c in pt}
    kl = {c: float(rows[c]["kappa_L_W_per_mK"]) for c in pt}
    assert max(fwhm, key=fwhm.get) == BEST_PT
    assert min(kl, key=kl.get) == BEST_PT


def test_al_series_shows_no_comparable_broadening(rows) -> None:
    """Stated in the main text as the contrast with the Pt-containing alloy."""
    assert (float(rows[BEST_AL]["FWHM_220_deg"])
            < float(rows[REFERENCE]["FWHM_220_deg"]))


def test_source_data_covers_every_alloy_series_composition(rows) -> None:
    assert len(rows) == 8
    measured = [c for c, r in rows.items() if r["FWHM_220_deg"] != ""]
    assert len(measured) == 6, "FWHM was measured for six of the eight"
