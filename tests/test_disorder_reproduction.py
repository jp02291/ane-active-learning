"""The Fig. 6(d) disorder parameters must stay reproducible, and stay distinct
from the Callaway-Klemens reconstruction of Methods and Supplementary Note S4.

Two conventions decide the numbers plotted in Fig. 6(d): the site sums carry no
(M_i / M_s)^2 weight, and the elemental data come from Supplementary Table S5(a)
rather than the Callaway-Klemens set of Table S5(b). Either change is invisible in the output but moves Gamma_V by an
order of magnitude, so both are pinned here alongside the qualitative
correspondence the main text draws between the disorder parameters, the (220)
FWHM and the lattice thermal conductivity. The Fe-Ga-Pt series is not
evaluated (Methods), and the tests pin both that omission and the reason for it.
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


def test_uses_the_descriptor_radii_not_the_callaway_set(mod) -> None:
    """Table S5(a), not S5(b). Ga is the entry that separates the two."""
    assert mod.METALLIC_RADIUS_A["Ga"] == pytest.approx(1.408)
    assert mod.METALLIC_RADIUS_A["Al"] == pytest.approx(1.429)


def test_site_sum_carries_no_mass_weight(mod) -> None:
    """A two-element site with equal fractions and equal masses is a case the
    weighted and unweighted forms agree on; an unequal-mass site is not."""
    site = {"Ga": 0.5, "Al": 0.5}
    got = mod.site_gamma(site, mod.ATOMIC_MASS)
    f, m = 0.5, mod.ATOMIC_MASS
    m_bar = f * m["Ga"] + f * m["Al"]
    unweighted = sum(f * (1 - m[e] / m_bar) ** 2 for e in site)
    weighted = sum(f * (m[e] / m_bar) ** 2 * (1 - m[e] / m_bar) ** 2 for e in site)
    assert got == pytest.approx(unweighted)
    assert got != pytest.approx(weighted)


def test_endmembers_have_zero_disorder(mod, rows) -> None:
    """A single occupant on each site cannot fluctuate."""
    for comp in (REFERENCE, "Fe0.75Al0.25"):
        assert float(rows[comp]["Gamma_M"]) == pytest.approx(0.0, abs=1e-12)
        assert float(rows[comp]["Gamma_V"]) == pytest.approx(0.0, abs=1e-12)


def test_fe_ga_pt_series_is_not_evaluated(rows) -> None:
    """Methods: the Fe-Ga-Pt series was not evaluated, so its Gamma cells are empty."""
    for comp, r in rows.items():
        if r["series"] == "Fe-Ga-Pt":
            assert r["Gamma_M"] == "" and r["Gamma_V"] == "", comp


def test_pt_result_depends_on_the_site_assignment(mod) -> None:
    """The reason given in Methods: the 0.01 Fe-site deficit decides the answer."""
    volume = {el: r ** 3 for el, r in mod.METALLIC_RADIUS_A.items()}
    gm = [mod.effective_gamma(s, mod.ATOMIC_MASS) for s in mod.PT_ASSIGNMENTS.values()]
    gv = [mod.effective_gamma(s, volume) for s in mod.PT_ASSIGNMENTS.values()]
    assert max(gm) / min(gm) > 1.8
    assert max(gv) / min(gv) > 50


def test_al_series_extrema_match_fig6d(rows) -> None:
    """Gamma_V peaks at the lowest-kappa_L composition, Gamma_M at x = 0.1875."""
    al = {c: r for c, r in rows.items() if r["Gamma_M"] != ""}
    gm = {c: float(r["Gamma_M"]) for c, r in al.items()}
    gv = {c: float(r["Gamma_V"]) for c, r in al.items()}
    kl = {c: float(r["kappa_L_W_per_mK"]) for c, r in al.items()}
    assert max(gv, key=gv.get) == BEST_AL == min(kl, key=kl.get)
    assert max(gm, key=gm.get) == "Fe0.75Ga0.0625Al0.1875"


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
