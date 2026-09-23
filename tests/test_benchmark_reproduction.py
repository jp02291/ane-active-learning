"""The Note S2 / Fig. S6 benchmark must stay reproducible and stay comparable.

`analysis/benchmark/run_benchmark.py` compares five regressors under one
protocol. The comparison only means anything while every model sees the same
inputs and the same splits, and three things can break that without changing
anything a reader can see.

The descriptors can drift. A surrogate scored on descriptors the campaign never
used is not the surrogate the manuscript describes, so these tests pin the
featurization to `ane.features`.

The partition can drift. The benchmark runs on the deposited cycle-1 split, so
the sizes and the absence of overlap are pinned.

The deposited numbers can drift from the manuscript. The published MAEs are
pinned here against the deposited results.

Re-running the search takes hours, so it is not exercised. What is exercised is
that the deposited hyperparameters, the deposited splits and the deposited
metrics are mutually consistent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "benchmark" / "run_benchmark.py"
RESULTS = ROOT / "analysis" / "benchmark" / "results"
TRAIN = ROOT / "data" / "split" / "cycle1" / "train.csv"
TEST = ROOT / "data" / "split" / "cycle1" / "test.csv"

MODELS = ("DNN", "KRR", "SVR", "XGB", "GPR")
ELEMENTS = ("Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt")

#: cross-validation MAE, mean over the nine splits, to three decimals
PUBLISHED_CV = {
    "kxx": {"DNN": 2.371, "KRR": 2.671, "SVR": 2.198, "XGB": 2.243, "GPR": 2.566},
    "S_ANE": {"DNN": 0.836, "KRR": 0.746, "SVR": 0.657, "XGB": 0.811, "GPR": 0.691},
}

#: held-out MAE, to three decimals
PUBLISHED_TEST = {
    "kxx": {"DNN": 3.069, "KRR": 2.260, "SVR": 2.540, "XGB": 2.953, "GPR": 2.183},
    "S_ANE": {"DNN": 0.478, "KRR": 0.436, "SVR": 0.419, "XGB": 0.628, "GPR": 0.552},
}


@pytest.fixture(scope="module")
def cv_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "cv_metrics_summary.csv")


@pytest.fixture(scope="module")
def cv_folds() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "cv_fold_metrics.csv")


@pytest.fixture(scope="module")
def test_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "held_out_test_metrics.csv")


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads((RESULTS / "comparison_protocol.json").read_text(encoding="utf-8"))


def test_script_uses_the_campaign_featurization() -> None:
    """Not a local copy. The import is the thing being pinned."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from ane.features import" in source
    assert "def make_15d_features" not in source
    assert "ELEMENT_PROPS = {" not in source


def test_partition_is_the_deposited_cycle1_split() -> None:
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    assert len(train) == 36
    assert len(test) == 9
    overlap = set(map(tuple, train[list(ELEMENTS)].round(9).to_numpy())) & set(
        map(tuple, test[list(ELEMENTS)].round(9).to_numpy())
    )
    assert not overlap, sorted(overlap)


def test_compositions_are_closed_to_unit_sum() -> None:
    """An unnormalized row would change the ILR coordinates for every model."""
    for path in (TRAIN, TEST):
        sums = pd.read_csv(path)[list(ELEMENTS)].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9), path.name


def test_protocol_records_the_shared_settings(protocol) -> None:
    assert protocol["seed"] == 42
    assert protocol["cross_validation"]["n_splits"] == 3
    assert protocol["cross_validation"]["n_repeats"] == 3
    assert protocol["cross_validation"]["same_splits_for_all_models"] is True
    assert protocol["hyperparameter_optimization"]["n_trials_per_model"] == 150
    assert protocol["features"]["total_input_dimensions"] == 15
    assert set(protocol["models"]) == set(MODELS)


def test_every_model_was_scored_on_the_same_nine_splits(cv_folds) -> None:
    counts = cv_folds.groupby(["model", "target"]).size().unique()
    assert list(counts) == [9]
    assert set(cv_folds.model) == set(MODELS)


@pytest.mark.parametrize("target", sorted(PUBLISHED_CV))
def test_published_cross_validation_mae(target, cv_summary) -> None:
    got = {
        r.model: round(float(r.MAE_mean), 3)
        for r in cv_summary[cv_summary.target == target].itertuples()
    }
    assert got == PUBLISHED_CV[target]


@pytest.mark.parametrize("target", sorted(PUBLISHED_TEST))
def test_published_held_out_mae(target, test_metrics) -> None:
    got = {
        r.model: round(float(r.MAE), 3)
        for r in test_metrics[test_metrics.target == target].itertuples()
    }
    assert got == PUBLISHED_TEST[target]


def test_summary_is_the_mean_of_the_folds(cv_summary, cv_folds) -> None:
    """The quoted table cannot drift away from the per-fold numbers."""
    for row in cv_summary.itertuples():
        folds = cv_folds[
            (cv_folds.model == row.model) & (cv_folds.target == row.target)
        ].MAE
        assert row.MAE_mean == pytest.approx(folds.mean())
        assert row.MAE_std == pytest.approx(folds.std(ddof=1))


def test_no_single_model_wins_both_evaluations(cv_summary, test_metrics) -> None:
    """Note S2 rests on this. If it ever stops holding, the note needs rewriting."""
    cv_best = {
        t: cv_summary[cv_summary.target == t].sort_values("MAE_mean").model.iloc[0]
        for t in PUBLISHED_CV
    }
    test_best = {
        t: test_metrics[test_metrics.target == t].sort_values("MAE").model.iloc[0]
        for t in PUBLISHED_TEST
    }
    winners = set(cv_best.values()) | set(test_best.values())
    assert len(winners) > 1, winners


def test_dnn_is_not_claimed_to_be_the_most_accurate(cv_summary, test_metrics) -> None:
    """The manuscript justifies the DNN structurally, not by accuracy."""
    for target in PUBLISHED_TEST:
        ranked = test_metrics[test_metrics.target == target].sort_values("MAE")
        assert ranked.model.iloc[0] != "DNN", target


def test_held_out_predictions_cover_every_model(test_metrics) -> None:
    preds = pd.read_csv(RESULTS / "held_out_test_predictions.csv")
    assert len(preds) == 9
    for model in MODELS:
        for target in ("kxx", "S_ANE"):
            assert f"pred_{target}_{model}" in preds.columns


def test_module_imports_without_running() -> None:
    """A syntax or import error here would only surface hours into a run.

    Skipped where xgboost is absent. It is declared in `requirements.txt`, but
    the rest of the suite does not need it and should stay runnable without it.
    """
    pytest.importorskip("xgboost")
    spec = importlib.util.spec_from_file_location("benchmark_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MODELS == list(MODELS)
    assert module.N_TRIALS == 150
    assert module.K_FOLDS == 3 and module.N_REPEATS == 3
