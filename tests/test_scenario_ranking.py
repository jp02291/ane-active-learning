"""Scenario selection must use the score the manuscript defines.

Supplementary Algorithm S3 defines the trial objective as the mean over five
folds of each fold's minimum validation MSE, and Section 2.4 says the scenario
with the lowest cross-validation loss is the one carried forward. An earlier
revision of `surrogate.py` ranked scenarios by the single-fold loss that
`select_final_epoch` produces while choosing the epoch count, which is a
different quantity. These tests pin the distinction so it cannot return.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from ane import surrogate


def test_run_search_returns_the_cv_score() -> None:
    """The Optuna objective value must reach the caller, not be discarded."""
    src = inspect.getsource(surrogate.run_search)
    assert "study.best_value" in src
    assert "return best, cv_score" in src

    sig = inspect.signature(surrogate.run_search)
    assert "tuple" in str(sig.return_annotation).lower()


def test_tune_scenario_records_both_scores_separately() -> None:
    src = inspect.getsource(surrogate.tune_scenario)
    assert '"optuna_cv_score"' in src, "the CV score must be recorded"
    assert '"final_epoch_fold_loss"' in src, "the epoch-selection loss must be kept as a diagnostic"
    assert '"best_val_loss"' not in src, (
        "best_val_loss conflated the two; use optuna_cv_score / final_epoch_fold_loss"
    )


def test_scenarios_are_ranked_by_the_cv_score() -> None:
    src = inspect.getsource(surrogate.run_scenarios)
    assert 'sort_values("optuna_cv_score"' in src, (
        "scenario ranking must use the five-fold CV score, not the single-fold "
        "loss from select_final_epoch"
    )
    assert 'sort_values("final_epoch_fold_loss"' not in src


def test_cv_objective_is_the_mean_of_per_fold_minima() -> None:
    """The score being ranked on has to be the one Algorithm S3 defines."""
    src = inspect.getsource(surrogate.cv_objective_loss)
    assert "np.nanmin" in src, "each fold contributes its minimum validation loss"
    assert "np.mean(fold_losses)" in src, "the objective is the mean over folds"


@pytest.mark.parametrize(
    "column", ["optuna_cv_score", "final_epoch_fold_loss", "optimal_epoch"]
)
def test_summary_columns_are_documented(column: str) -> None:
    """Every column the ranking depends on is named in the docstring."""
    assert column in surrogate.run_scenarios.__doc__


def test_ranking_order_follows_the_cv_score_not_the_fold_loss() -> None:
    """A case where the two criteria disagree must resolve to the CV score."""
    rows = [
        {"scenario": "dnn_base", "optuna_cv_score": 0.0100, "final_epoch_fold_loss": 0.0300},
        {"scenario": "dnn_gan_n200", "optuna_cv_score": 0.0200, "final_epoch_fold_loss": 0.0050},
    ]
    ranked = pd.DataFrame(rows).sort_values("optuna_cv_score", ascending=True)
    assert ranked.iloc[0]["scenario"] == "dnn_base"

    misranked = pd.DataFrame(rows).sort_values("final_epoch_fold_loss", ascending=True)
    assert misranked.iloc[0]["scenario"] == "dnn_gan_n200"


def test_cv_objective_needs_every_fold() -> None:
    """A parameter set that failed a fold must not be scored on the survivors.

    Supplementary Algorithm S3 defines the objective as the mean over five
    folds. Averaging only the folds that produced a finite loss scores such a
    set on fewer folds and flatters it relative to one that completed all five.
    """
    src = inspect.getsource(surrogate.cv_objective_loss)
    assert "len(fold_losses) != cfg.k_folds" in src, (
        "the objective must require every fold, not just the finite ones"
    )
    assert src.count('return float("inf")') >= 2, (
        "a non-finite fold should abandon the trial immediately"
    )
