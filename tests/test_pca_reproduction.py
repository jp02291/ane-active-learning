"""The Fig. 3(d) projection must stay reproducible from the released data.

`analysis/pca/run_pca.py` rebuilds the biplot from `data/candidates.csv`. Two
things about that analysis can break silently. An edit to the candidate table
changes the projection without changing anything a reader can see, and dropping
the standardization step changes it far more -- Fe and Ga carry the largest raw
spread, so mean-centering alone lets them dominate the first component. These
tests pin the published variance split and the fact that the two preprocessing
choices genuinely disagree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "candidates.csv"

ELEMENTS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]

#: percentages reported for Fig. 3(d), to one decimal
PUBLISHED_VARIANCE = (31.0, 27.3)

#: the campaign enumerated candidates on a 0.01 grid (Algorithm S5, line 2)
GRID_STEP = 0.01


@pytest.fixture(scope="module")
def candidates() -> pd.DataFrame:
    return pd.read_csv(CANDIDATES)


def test_candidate_table_has_the_thirty_nominations(candidates: pd.DataFrame) -> None:
    assert len(candidates) == 30
    assert candidates["cycle"].value_counts().to_dict() == {1: 10, 2: 10, 3: 10}
    assert candidates["selection_type"].value_counts().to_dict() == {
        "exploitation": 15,
        "exploration": 15,
    }


def test_compositions_close_to_unit_sum(candidates: pd.DataFrame) -> None:
    sums = candidates[ELEMENTS].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=5e-3), candidates.loc[
        ~np.isclose(sums, 1.0, atol=5e-3), "label"
    ].tolist()


def test_compositions_lie_on_the_enumeration_grid(candidates: pd.DataFrame) -> None:
    """A nominated composition off the 0.01 grid could not have been enumerated."""
    x = candidates[ELEMENTS].to_numpy(dtype=float)
    remainder = np.abs(x / GRID_STEP - np.round(x / GRID_STEP))
    bad = np.where(remainder.max(axis=1) > 1e-6)[0]
    assert not len(bad), (
        "off-grid compositions: "
        + ", ".join(candidates.loc[bad, "label"].tolist())
    )


def test_standardized_pca_reproduces_the_published_variance(
    candidates: pd.DataFrame,
) -> None:
    scaled = StandardScaler().fit_transform(candidates[ELEMENTS].to_numpy(dtype=float))
    ratio = PCA(n_components=2).fit(scaled).explained_variance_ratio_ * 100
    assert (round(float(ratio[0]), 1), round(float(ratio[1]), 1)) == PUBLISHED_VARIANCE


def test_mean_centering_alone_would_change_the_figure(
    candidates: pd.DataFrame,
) -> None:
    """Guards the standardization choice, which the caption depends on."""
    raw = candidates[ELEMENTS].to_numpy(dtype=float)
    centered = PCA(n_components=2).fit(raw).explained_variance_ratio_ * 100
    assert abs(centered[0] - PUBLISHED_VARIANCE[0]) > 5.0, (
        "standardization no longer changes the projection; the README and the "
        "Methods description of Fig. 3(d) would both need revisiting"
    )
