"""The Callaway reconstruction must match what the manuscript reports.

Thirteen of the 45 initial compositions carry a thermal conductivity that was
reconstructed rather than measured, so this model produced training labels. It
is the one calculation in the repository whose output is an input to everything
downstream, and until now nothing tested it: the suite could pass while the
reconstruction drifted away from Supplementary Table S1.

Two things are pinned here.

Equivalence. `physics.py` applies the Umklapp exponential with a positive
exponent and gamma = 2.0; Supplementary Note S1 quotes the standard negative
exponent and Table S6 reports gamma = 3.12. These are the same calculation at a
single temperature, and the first tests assert it numerically. Without this, a
reader comparing the two sees a contradiction where there is none.

Labels. The deposited kappa_total must be the kappa that appears in `data.csv`
and in Supplementary Table S1. Thirteen training labels rest on it.

Reproduction. The deposited kappa_L must be what `kappa_lattice` returns for
that composition, so a change to the model or its parameters cannot pass
unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ane.physics import CallawayParams, kappa_lattice, kappa_electronic

ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "data" / "literature_reconstruction.csv"

#: Column order `kappa_lattice` expects, which is not `elements.ELEMENTS`.
CALLAWAY_ORDER = ("Fe", "Ga", "Al", "Si", "Ge")

L0 = 2.44e-8
T = 300.0


@pytest.fixture(scope="module")
def entries() -> pd.DataFrame:
    return pd.read_csv(RECON)


def _composition(row) -> np.ndarray:
    c = np.array([row.get(e, 0.0) or 0.0 for e in CALLAWAY_ORDER], dtype=float)
    return c / c.sum()


# ---------------------------------------------------------------------------
# The two parameterizations are one calculation
# ---------------------------------------------------------------------------

def test_the_default_is_the_form_the_supplement_reports() -> None:
    """Defaults must be the standard negative-exponent pair, not the legacy one."""
    assert CallawayParams.UMKLAPP_SIGN == -1
    assert CallawayParams.LEGACY_UMKLAPP_SIGN == +1
    assert CallawayParams.LEGACY_GAMMA == 2.0


def test_si_equivalent_gamma_is_the_documented_ratio() -> None:
    """gamma = gamma_legacy * sqrt(exp(2 theta_D / 3T)), because B goes as gamma^2."""
    ratio = np.exp(2 * CallawayParams.THETA_D / (3 * T))
    expected = CallawayParams.LEGACY_GAMMA * np.sqrt(ratio)
    assert CallawayParams.GAMMA == pytest.approx(expected, rel=1e-12)
    assert CallawayParams.GAMMA_SI_EQUIVALENT == pytest.approx(expected, rel=1e-12)
    assert round(CallawayParams.GAMMA, 2) == 3.12, (
        "Supplementary Table S6 reports 3.12"
    )


def test_both_parameterisations_give_the_same_lattice_conductivity(entries: pd.DataFrame) -> None:
    """The sign in Supplementary Note S1 and the campaign's sign are one calculation."""
    for _, row in entries.iterrows():
        comp = _composition(row)
        as_reported = kappa_lattice(comp)  # defaults: the reported form
        as_run = kappa_lattice(
            comp,
            umklapp_sign=CallawayParams.LEGACY_UMKLAPP_SIGN,
            gamma=CallawayParams.LEGACY_GAMMA,
        )
        assert as_run == pytest.approx(as_reported, rel=1e-9), row["label"]


def test_the_signs_are_not_interchangeable_on_their_own() -> None:
    """Flipping the sign without the compensating gamma changes kappa_L by tens of percent.

    This is what makes the equivalence worth asserting rather than assuming, and
    what makes it unsafe to change either default alone.
    """
    comp = np.array([0.750, 0.138, 0.113, 0.0, 0.0])
    comp = comp / comp.sum()
    a = kappa_lattice(comp, umklapp_sign=+1, gamma=CallawayParams.GAMMA)
    b = kappa_lattice(comp, umklapp_sign=-1, gamma=CallawayParams.GAMMA)
    assert abs(b - a) / a > 0.5


# ---------------------------------------------------------------------------
# The deposited reconstruction reproduces Supplementary Table S1
# ---------------------------------------------------------------------------

def test_file_covers_the_thirteen_literature_entries(entries: pd.DataFrame) -> None:
    assert len(entries) == 13
    assert set(entries["reference"]) == {"[1]", "[2]"}


def test_electronic_term_follows_wiedemann_franz() -> None:
    """kappa_electronic is L0 sigma T and nothing else."""
    assert kappa_electronic(1.0e6, T, L0) == pytest.approx(1.0e6 * L0 * T)
    assert kappa_electronic(0.0, T, L0) == 0.0


def test_lattice_term_matches_the_model(entries: pd.DataFrame) -> None:
    """The deposited kappa_L is what `kappa_lattice` returns for that composition.

    Without this the suite could pass while the reconstruction drifted away from
    the deposited column and from Supplementary Note S1.
    """
    for _, row in entries.iterrows():
        assert kappa_lattice(_composition(row)) == pytest.approx(
            row["kappa_L"], abs=1e-3
        ), row["label"]


def test_totals_match_the_training_labels_in_data_csv(entries: pd.DataFrame) -> None:
    """The reconstruction is not a side calculation: it is 13 of the 45 training labels."""
    data = pd.read_csv(ROOT / "data" / "data.csv")
    model_rows = data[(data.cycle_added == 0) & (data.kappa_source == "model")]
    assert len(model_rows) == 13

    deposited = dict(zip(entries["label"], entries["kappa_total"]))
    for _, row in model_rows.iterrows():
        assert row["label"] in deposited, row["label"]
        assert deposited[row["label"]] == pytest.approx(row["kxx"], abs=1e-6)
