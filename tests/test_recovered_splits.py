"""The cycle 2 and cycle 3 partitions were recovered, so they have to stay checkable.

The original split files for those cycles were not archived. What survived is
the Fig. 2 prediction record, whose `y_true` values are the measured properties
of the held-out rows. Because every (kappa, |S_ANE|) pair in `data/data.csv` is
unique, each of those rows identifies exactly one composition, and the held-out
membership follows.

That argument only holds while three things remain true, and each is pinned
here: the pairs stay unique, every deposited row still matches its prediction
record far more closely than any alternative, and the recovered partition obeys
the protocol it is supposed to have come from -- the right pool for the cycle,
no composition from the top 15 percent by figure of merit, train and test
disjoint and covering the pool exactly.

A change to `data/data.csv` is what would break this. If two compositions ever
share a (kappa, |S_ANE|) pair, the recovery stops being an identification and
these tests should fail rather than let the partition drift quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data.csv"

RECOVERED = (2, 3)
#: the recovered membership is pinned here and copied out by stage 0
REPORTED = ROOT / "data" / "reported_splits"
#: pool size and held-out count for every cycle, including the retained cycle 1
EXPECTED = {1: (45, 9), 2: (55, 10), 3: (65, 10)}

#: the recovery is an identification only while the runner-up is far away; the
#: worst observed ratio is about 2e-2 against a match at about 8e-8
MIN_SEPARATION_RATIO = 1e4


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return pd.read_csv(DATA)


def _manifest(cycle: int) -> dict:
    """What stage 0 wrote out."""
    return json.loads(
        (ROOT / "data" / "split" / f"cycle{cycle}" / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _recovery(cycle: int) -> dict:
    """How the pinned membership was obtained, for the two recovered cycles."""
    return json.loads(
        (REPORTED / f"cycle{cycle}" / "recovery_manifest.json").read_text(encoding="utf-8")
    )


def _split(cycle: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / "data" / "split" / f"cycle{cycle}"
    return pd.read_csv(base / "train.csv"), pd.read_csv(base / "test.csv")


def test_property_pairs_identify_a_composition(dataset: pd.DataFrame) -> None:
    """The whole recovery rests on this. Without it the match is a guess."""
    pairs = list(zip(dataset.kxx.round(6), dataset.S_ANE.abs().round(6)))
    assert len(set(pairs)) == len(dataset), "a (kappa, |S_ANE|) pair is shared"


@pytest.mark.parametrize("cycle", sorted(EXPECTED))
def test_split_sizes_and_pool(cycle: int, dataset: pd.DataFrame) -> None:
    train, test = _split(cycle)
    pool_size, n_test = EXPECTED[cycle]
    assert (len(train), len(test)) == (pool_size - n_test, n_test)
    assert not set(train.label) & set(test.label), "train and test overlap"

    manifest = _manifest(cycle)
    pool = dataset[dataset.cycle_added < manifest["up_to_cycle"]]
    assert len(pool) == pool_size
    assert set(train.label) | set(test.label) == set(pool.label)


@pytest.mark.parametrize("cycle", sorted(EXPECTED))
def test_no_held_out_composition_comes_from_the_top_fraction(
    cycle: int, dataset: pd.DataFrame
) -> None:
    """The protocol keeps the top 15 percent by figure of merit in training."""
    manifest = _manifest(cycle)
    pool = dataset[dataset.cycle_added < manifest["up_to_cycle"]]
    cut = (pool.S_ANE.abs() / pool.kxx).quantile(1.0 - manifest["top_fraction"])
    _, test = _split(cycle)
    assert ((test.S_ANE.abs() / test.kxx) >= cut).sum() == 0


@pytest.mark.parametrize("cycle", RECOVERED)
def test_recovery_is_declared_and_sourced(cycle: int) -> None:
    """A recovered partition must say so, and name what it was recovered from."""
    import hashlib

    recovery = _recovery(cycle)
    assert recovery["split_source"] == "recovered_from_historical_test_predictions"
    assert recovery["original_split_csv_retained"] is False
    assert recovery["source_file_sha256"], "no source file recorded"
    for rel, digest in recovery["source_file_sha256"].items():
        source = ROOT / rel
        assert source.exists(), rel
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, rel


@pytest.mark.parametrize("cycle", RECOVERED)
def test_the_pinned_membership_is_what_stage_0_copies(cycle: int) -> None:
    """Stage 0 must reproduce the historical split, not draw a fresh one.

    A fresh draw on the same pool and the same seed does not reproduce it: the
    campaign's split came from the original notebooks, and only three of ten
    held-out compositions coincide in cycle 2. Pinning the membership is what
    keeps a re-run comparable with Fig. 2.
    """
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / f"cycle{cycle}.yaml").read_text(encoding="utf-8"))
    split = cfg["split"]
    assert split["fixed_train_csv"] == f"data/reported_splits/cycle{cycle}/train.csv"
    assert split["fixed_test_csv"] == f"data/reported_splits/cycle{cycle}/test.csv"
    assert _manifest(cycle)["split_source"] == "fixed_reported_membership"

    for name in ("train.csv", "test.csv"):
        pinned = pd.read_csv(REPORTED / f"cycle{cycle}" / name)
        written = pd.read_csv(ROOT / "data" / "split" / f"cycle{cycle}" / name)
        assert set(pinned.label) == set(written.label), name


@pytest.mark.parametrize("cycle", RECOVERED)
def test_every_held_out_row_still_matches_its_prediction_record(
    cycle: int, dataset: pd.DataFrame
) -> None:
    """Re-derive the match, and require the runner-up to stay far away."""
    record_path = next(iter(_recovery(cycle)["source_file_sha256"]))
    record = pd.read_csv(ROOT / record_path)
    _, test = _split(cycle)
    assert len(record) == len(test)

    kappa = dataset.kxx.to_numpy(float)
    s_ane = dataset.S_ANE.abs().to_numpy(float)
    for (_, row), label in zip(record.iterrows(), test.label):
        distance = np.hypot(
            (kappa - row.y_true_kxx) / np.maximum(np.abs(kappa), 1e-9),
            (s_ane - abs(row.y_true_S_ANE)) / np.maximum(s_ane, 1e-9),
        )
        order = np.argsort(distance)
        assert dataset.label[order[0]] == label, "the recovered label moved"
        best, runner_up = distance[order[0]], distance[order[1]]
        assert runner_up > MIN_SEPARATION_RATIO * max(best, 1e-18), (
            f"cycle {cycle}: {label} is no longer unambiguous"
        )


@pytest.mark.parametrize("cycle", RECOVERED)
def test_what_the_recovery_does_not_provide(cycle: int) -> None:
    """The partition is back; the run records behind the branch choice are not."""
    missing = _recovery(cycle)["not_recovered"]
    assert "best_params.json" in missing
    assert "scenario_summary.csv" in missing


def test_the_sampler_draws_at_random() -> None:
    """The released sampler must be the one the manuscript describes.

    The draw is random. Stratifying it over clusters of the compositions and
    targets would not reproduce any of the three recorded memberships, and it
    would not make the partition any more target informed than it already is --
    the retention of the top fraction is what does that.
    """
    import inspect

    from ane import data as ane_data

    src = inspect.getsource(ane_data.make_split)
    assert "stratify=" not in src, "the draw must not be stratified"
    assert "KMeans" not in src
    assert "train_test_split(" in src
