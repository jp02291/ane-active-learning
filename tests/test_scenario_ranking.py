"""Scenario summaries must not replace the campaign's manual choice.

CV still tunes hyperparameters within each scenario. The researchers reviewed
both held-out property MAEs manually, and stage 3 uses the configured scenario.
Summary row order must not imply a winner under any metric.
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


def test_cv_objective_is_the_mean_of_per_fold_minima() -> None:
    """Within-scenario HPO still averages per-fold minimum losses."""
    src = inspect.getsource(surrogate.cv_objective_loss)
    assert "np.nanmin" in src, "each fold contributes its minimum validation loss"
    assert "np.mean(fold_losses)" in src, "the objective is the mean over folds"


@pytest.mark.parametrize(
    "column", ["optuna_cv_score", "final_epoch_fold_loss", "optimal_epoch"]
)
def test_summary_columns_are_documented(column: str) -> None:
    """The tuning diagnostics remain documented after removing ranking."""
    assert column in surrogate.run_scenarios.__doc__


@pytest.mark.parametrize("reverse", [False, True])
def test_summary_preserves_execution_order_without_selecting_a_winner(
    monkeypatch, tmp_path, reverse
) -> None:
    """Exercise the real summary writer with competing diagnostic/MAE rankings."""
    from types import SimpleNamespace
    from tensorflow.keras import backend as K

    rows = [
        {"scenario": "dnn_base", "optuna_cv_score": 0.01,
         "final_epoch_fold_loss": 0.03, "test_mae_kxx": 4.0,
         "test_mae_S_ANE": 0.9},
        {"scenario": "dnn_gan_n200", "optuna_cv_score": 0.02,
         "final_epoch_fold_loss": 0.005, "test_mae_kxx": 2.0,
         "test_mae_S_ANE": 0.5},
    ]
    if reverse:
        rows.reverse()
    configs = [SimpleNamespace(artifact_dir=r["scenario"]) for r in rows]
    by_name = {r["scenario"]: r for r in rows}
    monkeypatch.setattr(surrogate, "tune_scenario",
                        lambda cfg, verbose: by_name[cfg.artifact_dir].copy())
    monkeypatch.setattr(K, "clear_session", lambda: None)
    output = tmp_path / "scenario_summary.csv"
    summary = surrogate.run_scenarios(configs, output, verbose=False)
    expected = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(summary, expected)
    pd.testing.assert_frame_equal(pd.read_csv(output), expected)
    assert [cfg.artifact_dir for cfg in configs] == expected.scenario.tolist()


def test_cv_objective_needs_every_fold() -> None:
    """A parameter set that failed a fold must not be scored on the survivors.

    Methods defines the objective as the mean over five
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
def test_search_training_is_seeded() -> None:
    """Seeding the sampler and the fold split does not make the search reproducible.

    Weight initialization, dropout and batch order also decide a trial's score.
    Without a seed reset before the model is built, the same parameter set
    scores differently on every run and the search result is not reproducible.
    """
    src = inspect.getsource(surrogate.cv_objective_loss)
    seed_at = src.index("set_random_seed(cfg.seed)")
    build_at = src.index("build_model(")
    assert seed_at < build_at, "the seed has to be set before the model is built"


def test_final_refit_is_seeded() -> None:
    """The refit whose test metrics are reported is seeded like the folds."""
    src = inspect.getsource(surrogate.train_final_model)
    assert "set_random_seed(cfg.seed)" in src
    assert src.index("set_random_seed(cfg.seed)") < src.index("build_model(")


def test_a_configured_generated_set_must_exist() -> None:
    """A missing generated file must fail, not quietly become a real-only run.

    Falling back would run the control and write it under the augmented
    scenario's name, which no downstream artifact distinguishes.
    """
    src = inspect.getsource(surrogate.tune_scenario)
    assert "raise FileNotFoundError" in src, "a missing generated_csv must raise"
    assert "if cfg.generated_csv is not None and Path(cfg.generated_csv).exists()" not in src, (
        "existence must not be part of the condition that enables augmentation"
    )
