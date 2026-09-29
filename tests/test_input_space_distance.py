"""The Discussion's input-space distances must follow from the released data.

The Discussion explains the cycle-1 overpredictions of three Co-free Fe-Mn
alloys by their distance from the training data. These tests recompute that
distance as Methods define it, the Euclidean distance between compositions in
the 15-dimensional input space with each feature min-max scaled to the cycle-1
training data, and pin the deposited summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "input_space"))

from run_distance import scaled_features  # noqa: E402

ELEMENTS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]
SUMMARY = ROOT / "analysis" / "input_space" / "results" / "distance_summary.json"


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train = pd.read_csv(ROOT / "data" / "split" / "cycle1" / "train.csv")
    cand = pd.read_csv(ROOT / "data" / "candidates.csv")
    cand = cand[cand.cycle == 1].reset_index(drop=True)
    s_train, s_cand = scaled_features(train, cand)
    return train, cand, s_train, s_cand


def test_training_set_is_the_cycle1_partition(data) -> None:
    train, cand, s_train, _ = data
    assert len(train) == 36
    assert len(cand) == 10
    assert s_train.shape == (36, 15)
    assert np.allclose(s_train.min(axis=0), 0.0) and np.allclose(s_train.max(axis=0), 1.0)


def test_fe_mn_candidates_are_the_most_isolated(data) -> None:
    """The three Co-free Mn-containing candidates are farther than every other one."""
    _, cand, s_train, s_cand = data
    nn = np.linalg.norm(s_cand[:, None, :] - s_train[None, :, :], axis=2).min(axis=1)
    fe_mn = ((cand.Mn > 0) & (cand.Co == 0)).to_numpy()
    assert fe_mn.sum() == 3
    assert nn[fe_mn].min() > nn[~fe_mn].max()
    assert round(float(nn[fe_mn].min()), 2) == 0.93
    assert round(float(nn[fe_mn].max()), 2) == 1.15


def test_nearest_training_entries_have_small_s_ane(data) -> None:
    """Their two closest training compositions contain 12.5 at.% Mn and |S_ANE| <= 1.4."""
    train, cand, s_train, s_cand = data
    d = np.linalg.norm(s_cand[:, None, :] - s_train[None, :, :], axis=2)
    for i in np.flatnonzero(((cand.Mn > 0) & (cand.Co == 0)).to_numpy()):
        two = np.argsort(d[i])[:2]
        assert np.allclose(train.Mn.to_numpy()[two], 0.125)
        assert train.S_ANE.to_numpy()[two].max() <= 1.4


def test_training_nearest_neighbour_spread(data) -> None:
    _, _, s_train, _ = data
    d = np.linalg.norm(s_train[:, None, :] - s_train[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    assert round(float(np.median(nn)), 2) == 0.17
    assert round(float(nn.max()), 2) == 1.16


def test_deposited_summary_matches(data) -> None:
    summary = json.loads(SUMMARY.read_text())
    assert summary["fe_mn_cycle1_candidates"]["nn_distance_min"] == 0.928
    assert summary["fe_mn_cycle1_candidates"]["nn_distance_max"] == 1.149
    assert summary["training_nn_distance"] == {"median": 0.174, "max": 1.163}
