"""Numerical parity between `ane.surrogate` and the original notebook code.

The reference block is transcribed from ensemble_v3.ipynb. The notebook's
`Config` class becomes `_RefConfig`, functions that read it directly now take
their values as arguments so both sides can be driven identically, and nothing
else is changed. Do not tidy it.

What is checked. Everything that decides *which* members become the ensemble:
the robust outlier threshold and both of its fallbacks, the data-derived kappa
floor, the pass/fail flags, and all three selection modes including the
fallback. Also the scaling path -- that the scalers are fit on measured data
only and generated samples are merely transformed -- and the per-member
train/validation split, which is seeded explicitly and so is reproducible
without TensorFlow.

What is not. Training. It needs TensorFlow and forty minutes, and Keras does
not promise bitwise reproducibility across machines, so an exact comparison is
not available at any price. The consequence is worth stating plainly: this file
shows that a given set of member metrics produces the same ensemble as the
notebook would, not that the same metrics arise. The metrics themselves come
from `model.fit`, and the guard there is that the port calls the same Keras
objects with the same arguments, which is a reading check, not a test.

    python -m pytest tests/test_surrogate_parity.py -q
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ane.config import SurrogateConfig
from ane.surrogate import (
    add_selection_flags,
    load_table,
    prepare_arrays,
    resolve_kappa_floor,
    robust_upper_threshold,
    select_members,
)

COMP_COLS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]
TARGET_COLS = ["kxx", "S_ANE"]


# ---------------------------------------------------------------------------
# reference implementation, from ensemble_v3.ipynb
# ---------------------------------------------------------------------------


class _RefConfig:
    COMP_COLS = COMP_COLS
    TARGET_COLS = TARGET_COLS

    RAW_ENSEMBLE_SIZE = 60
    SELECT_MAX_MODELS = 30
    SELECT_MIN_MODELS = 20

    SEED_BASE = 2025
    EPOCHS = 300
    VAL_SIZE = 0.20

    EARLYSTOP_PATIENCE = 50
    LR_PATIENCE = 15
    LR_FACTOR = 0.8

    MAD_FACTOR = 2.5
    IQR_FACTOR = 1.5

    USE_TARGETWISE_MAE_FILTER = True
    USE_VAL_KXX_FLOOR_FILTER = True

    KXX_MEMBER_FLOOR: Optional[float] = None
    KXX_FLOOR_QUANTILE = 0.05
    KXX_FLOOR_MIN = 0.5

    ALLOW_FALLBACK_TO_MIN_MODELS = True


def _ref_robust_upper_threshold(
    values, mad_factor=_RefConfig.MAD_FACTOR, iqr_factor=_RefConfig.IQR_FACTOR
):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.inf

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return med + mad_factor * mad

    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr > 1e-12:
        return float(q3 + iqr_factor * iqr)

    return np.inf


def _ref_resolve_kxx_member_floor(df_train):
    if _RefConfig.KXX_MEMBER_FLOOR is not None:
        return float(_RefConfig.KXX_MEMBER_FLOOR)

    kxx = df_train[_RefConfig.TARGET_COLS[0]].to_numpy(dtype=float)
    kxx = kxx[np.isfinite(kxx)]
    if len(kxx) == 0:
        return float(_RefConfig.KXX_FLOOR_MIN)

    q_floor = float(np.quantile(kxx, _RefConfig.KXX_FLOOR_QUANTILE))
    floor = max(q_floor, float(_RefConfig.KXX_FLOOR_MIN))
    return floor


def _ref_add_selection_flags(df, kxx_floor):
    df = df.copy()

    thr_mean = _ref_robust_upper_threshold(df["val_mae_mean"].to_numpy())
    thr_kxx = _ref_robust_upper_threshold(df["val_mae_kxx"].to_numpy())
    thr_s = _ref_robust_upper_threshold(df["val_mae_S_ANE"].to_numpy())

    df["threshold_val_mae_mean"] = thr_mean
    df["threshold_val_mae_kxx"] = thr_kxx
    df["threshold_val_mae_S_ANE"] = thr_s

    df["pass_val_mae_mean"] = df["val_mae_mean"] <= thr_mean

    if _RefConfig.USE_TARGETWISE_MAE_FILTER:
        df["pass_val_mae_kxx"] = df["val_mae_kxx"] <= thr_kxx
        df["pass_val_mae_S_ANE"] = df["val_mae_S_ANE"] <= thr_s
    else:
        df["pass_val_mae_kxx"] = True
        df["pass_val_mae_S_ANE"] = True

    if _RefConfig.USE_VAL_KXX_FLOOR_FILTER:
        df["pass_val_kxx_floor"] = df["val_pred_kxx_min"] > kxx_floor
    else:
        df["pass_val_kxx_floor"] = True

    df["pass_all_filters"] = (
        df["pass_val_mae_mean"]
        & df["pass_val_mae_kxx"]
        & df["pass_val_mae_S_ANE"]
        & df["pass_val_kxx_floor"]
    )

    return df


def _ref_select_members(df_metrics, min_models=None, max_models=None):
    min_models = _RefConfig.SELECT_MIN_MODELS if min_models is None else min_models
    max_models = _RefConfig.SELECT_MAX_MODELS if max_models is None else max_models

    df = df_metrics.copy()
    df = df.sort_values("val_mae_mean", ascending=True).reset_index(drop=True)

    selected = df[df["pass_all_filters"]].copy()
    selection_mode = "robust_filters"

    if len(selected) > max_models:
        selected = selected.sort_values("val_mae_mean", ascending=True).head(max_models).copy()
        selection_mode = "robust_filters_top_by_val_mae"

    if len(selected) < min_models:
        if not _RefConfig.ALLOW_FALLBACK_TO_MIN_MODELS:
            raise RuntimeError("too few members")
        fallback_pool = df[np.isfinite(df["val_mae_mean"])].copy()
        selected = fallback_pool.sort_values("val_mae_mean", ascending=True).head(min_models).copy()
        selection_mode = "fallback_best_val_mae_to_min_models"

    selected = selected.sort_values("val_mae_mean", ascending=True).reset_index(drop=True)
    selected["selected_rank"] = np.arange(len(selected), dtype=int)

    info = {
        "selection_mode": selection_mode,
        "n_raw_members": int(len(df_metrics)),
        "n_pass_all_filters": int(df_metrics["pass_all_filters"].sum()),
        "n_selected": int(len(selected)),
        "select_min_models": int(min_models),
        "select_max_models": int(max_models),
        "mad_factor": float(_RefConfig.MAD_FACTOR),
        "iqr_factor": float(_RefConfig.IQR_FACTOR),
    }
    return selected, info


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> SurrogateConfig:
    cfg = SurrogateConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _member_metrics(n=60, seed=0, contaminate=True) -> pd.DataFrame:
    """A plausible sixty-member metrics table.

    Contaminated on purpose: a run of sixty members on a few dozen rows
    reliably produces a few that land in bad optima, and those are the rows the
    pruning exists for. A clean table would exercise none of it.
    """
    rng = np.random.default_rng(seed)
    mae_kxx = np.abs(rng.normal(0.8, 0.15, size=n))
    mae_s = np.abs(rng.normal(0.25, 0.05, size=n))
    kxx_min = rng.uniform(1.0, 4.0, size=n)

    if contaminate and n >= 10:
        mae_kxx[:3] *= 8.0            # gross outliers on one target
        mae_s[3:5] *= 10.0            # gross outliers on the other
        kxx_min[5:8] = rng.uniform(-0.5, 0.4, size=3)   # below any plausible floor
        mae_kxx[9] = np.nan           # a member that failed to produce a metric

    return pd.DataFrame(
        {
            "member_id": np.arange(n),
            "val_mae_mean": (mae_kxx + mae_s) / 2.0,
            "val_mae_kxx": mae_kxx,
            "val_mae_S_ANE": mae_s,
            "val_pred_kxx_min": kxx_min,
        }
    )


def _synthetic_frame(n, seed, with_targets=True) -> pd.DataFrame:
    """Compositions shaped like the measured data: three to five elements."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        k = int(rng.integers(3, 6))
        idx = rng.choice(len(COMP_COLS), size=k, replace=False)
        x = np.zeros(len(COMP_COLS))
        x[idx] = rng.dirichlet(np.ones(k))
        rows.append(x)
    df = pd.DataFrame(rows, columns=COMP_COLS)
    if with_targets:
        df["kxx"] = rng.uniform(2.0, 20.0, size=n)
        df["S_ANE"] = rng.normal(1.0, 0.6, size=n)
    return df


@pytest.fixture
def scenario(tmp_path):
    """A scenario directory: train, test, generated, and best_params.json."""
    _synthetic_frame(57, seed=1).to_csv(tmp_path / "train.csv", index=False)
    _synthetic_frame(10, seed=2).to_csv(tmp_path / "test.csv", index=False)
    _synthetic_frame(200, seed=3).to_csv(tmp_path / "generated.csv", index=False)

    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    params = {
        "num_layers": 3,
        "num_neurons": 64,
        "dropout": 0.1,
        "l2": 1e-5,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "w_gen": 0.42,
    }
    (artifact / "best_params.json").write_text(json.dumps(params), encoding="utf-8")

    cfg = _cfg(
        train_csv=str(tmp_path / "train.csv"),
        test_csv=str(tmp_path / "test.csv"),
        generated_csv=str(tmp_path / "generated.csv"),
        artifact_dir=str(artifact),
    )
    return cfg, params, tmp_path


# ---------------------------------------------------------------------------
# robust threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_robust_threshold_matches_reference(seed):
    rng = np.random.default_rng(seed)
    values = np.abs(rng.normal(1.0, 0.3, size=60))
    values[:4] *= 12.0
    values[7] = np.nan

    ours = robust_upper_threshold(values, 2.5, 1.5)
    assert ours == _ref_robust_upper_threshold(values)


def test_robust_threshold_iqr_fallback_matches_reference():
    """Exactly half identical: the MAD is zero, the IQR is not.

    Half the values sitting on the median drives the MAD to zero, and the other
    half being spread keeps the quartiles apart, so this is the narrow case
    where the fallback produces a real cutoff rather than infinity.
    """
    values = np.concatenate([np.full(26, 1.0), np.linspace(1.1, 4.0, 24)])
    ours = robust_upper_threshold(values, 2.5, 1.5)
    ref = _ref_robust_upper_threshold(values)
    assert ours == ref
    assert np.isfinite(ours), "the IQR fallback should have produced a finite cutoff"


def test_robust_threshold_degenerate_matches_reference():
    identical = np.full(30, 2.0)
    assert robust_upper_threshold(identical, 2.5, 1.5) == _ref_robust_upper_threshold(identical)
    assert robust_upper_threshold(identical, 2.5, 1.5) == np.inf

    empty = np.array([np.nan, np.inf, -np.inf])
    assert robust_upper_threshold(empty, 2.5, 1.5) == _ref_robust_upper_threshold(empty)


# ---------------------------------------------------------------------------
# kappa floor
# ---------------------------------------------------------------------------


def test_kappa_floor_matches_reference():
    df = _synthetic_frame(57, seed=1)
    cfg = _cfg()
    assert resolve_kappa_floor(df, cfg, verbose=False) == _ref_resolve_kxx_member_floor(df)


def test_kappa_floor_respects_hard_minimum():
    """A cycle whose measured kappa runs low must not lower the floor freely."""
    df = pd.DataFrame({"kxx": np.full(20, 0.2), "S_ANE": np.ones(20)})
    cfg = _cfg()
    assert resolve_kappa_floor(df, cfg, verbose=False) == cfg.kappa_floor_min
    assert resolve_kappa_floor(df, cfg, verbose=False) == _ref_resolve_kxx_member_floor(df)


def test_kappa_floor_override():
    df = _synthetic_frame(30, seed=9)
    assert resolve_kappa_floor(df, _cfg(kappa_member_floor=3.5), verbose=False) == 3.5


# ---------------------------------------------------------------------------
# selection flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_flags_match_reference(seed):
    df = _member_metrics(seed=seed)
    floor = 0.5
    ours = add_selection_flags(df, _cfg(), floor)
    ref = _ref_add_selection_flags(df, floor)

    for col in ref.columns:
        assert np.array_equal(
            ours[col].to_numpy(), ref[col].to_numpy(), equal_nan=ref[col].dtype.kind == "f"
        ), col


def test_selection_flags_actually_reject():
    """A test that never rejects anything would pass against a broken filter."""
    df = add_selection_flags(_member_metrics(seed=0), _cfg(), 0.5)
    assert not df["pass_all_filters"].all()
    assert not df["pass_val_mae_kxx"].all()
    assert not df["pass_val_kxx_floor"].all()


def test_targetwise_filter_can_be_disabled_like_the_reference():
    df = _member_metrics(seed=0)
    ours = add_selection_flags(df, _cfg(prune_targetwise_mae=False), 0.5)
    assert ours["pass_val_mae_kxx"].all() and ours["pass_val_mae_S_ANE"].all()


# ---------------------------------------------------------------------------
# member selection, all three modes
# ---------------------------------------------------------------------------


def test_select_members_matches_reference_robust_mode():
    df = _ref_add_selection_flags(_member_metrics(seed=0), 0.5)
    ours, info = select_members(df, _cfg(), verbose=False)
    ref, ref_info = _ref_select_members(df)

    assert info["selection_mode"] == ref_info["selection_mode"]
    assert info["n_selected"] == ref_info["n_selected"]
    assert ours["member_id"].tolist() == ref["member_id"].tolist()
    assert ours["selected_rank"].tolist() == ref["selected_rank"].tolist()


def test_select_members_matches_reference_cap_mode():
    """More survivors than the cap: both must keep the same best 30."""
    df = _ref_add_selection_flags(_member_metrics(seed=4, contaminate=False), 0.5)
    assert df["pass_all_filters"].sum() > 30

    ours, info = select_members(df, _cfg(), verbose=False)
    ref, ref_info = _ref_select_members(df)

    assert info["selection_mode"] == "robust_filters_top_by_val_mae" == ref_info["selection_mode"]
    assert ours["member_id"].tolist() == ref["member_id"].tolist()


def test_select_members_matches_reference_fallback_mode():
    """Fewer survivors than the minimum: both must fall back the same way."""
    df = _ref_add_selection_flags(_member_metrics(seed=0), 0.5)
    df.loc[df.index[10:], "pass_all_filters"] = False
    assert df["pass_all_filters"].sum() < 20

    ours, info = select_members(df, _cfg(), verbose=False)
    ref, ref_info = _ref_select_members(df)

    assert info["selection_mode"] == "fallback_best_val_mae_to_min_models"
    assert info["selection_mode"] == ref_info["selection_mode"]
    assert ours["member_id"].tolist() == ref["member_id"].tolist()
    assert len(ours) == 20


def test_select_members_can_refuse_to_fall_back():
    df = _ref_add_selection_flags(_member_metrics(seed=0), 0.5)
    df.loc[df.index[10:], "pass_all_filters"] = False
    with pytest.raises(RuntimeError):
        select_members(df, _cfg(allow_fallback_to_min_models=False), verbose=False)


def test_selected_members_are_ranked_by_validation_mae():
    df = _ref_add_selection_flags(_member_metrics(seed=1), 0.5)
    selected, _ = select_members(df, _cfg(), verbose=False)
    mae = selected["val_mae_mean"].to_numpy()
    assert np.all(np.diff(mae) >= 0)


# ---------------------------------------------------------------------------
# the scaling path
# ---------------------------------------------------------------------------


def test_prepare_arrays_matches_reference(scenario):
    """Reproduces load_and_preprocess_data, including the fit/transform split."""
    from ane.features import featurize

    cfg, params, tmp = scenario
    (X_real, y_real), (X_gen, y_gen), info = prepare_arrays(cfg, params, verbose=False)

    df_tr = pd.read_csv(tmp / "train.csv")
    df_ge = pd.read_csv(tmp / "generated.csv")

    Xr_raw = featurize(df_tr[COMP_COLS].to_numpy(dtype=np.float64))
    yr_raw = df_tr[TARGET_COLS].to_numpy(dtype=np.float64)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    ref_Xr = sx.fit_transform(Xr_raw)
    ref_yr = sy.fit_transform(yr_raw)
    ref_Xg = sx.transform(featurize(df_ge[COMP_COLS].to_numpy(dtype=np.float64)))
    ref_yg = sy.transform(df_ge[TARGET_COLS].to_numpy(dtype=np.float64))

    assert np.array_equal(X_real, ref_Xr)
    assert np.array_equal(y_real, ref_yr)
    assert np.array_equal(X_gen, ref_Xg)
    assert np.array_equal(y_gen, ref_yg)
    assert info == {
        "n_train_real": len(df_tr),
        "n_generated": len(df_ge),
        "input_dim": 15,
        "target_dim": 2,
    }


def test_scalers_are_fit_on_measured_data_only(scenario):
    """The claim made in section 2.4, checked rather than asserted.

    The generated table is constructed to lie outside the measured range. If
    the scaler had been fit on the union, the measured rows could not span
    [0, 1] exactly and the generated rows could not fall outside it.
    """
    cfg, params, tmp = scenario
    gen = pd.read_csv(tmp / "generated.csv")
    gen["kxx"] = gen["kxx"] * 10.0 + 500.0
    gen.to_csv(tmp / "generated.csv", index=False)

    (X_real, y_real), (_, y_gen), _ = prepare_arrays(cfg, params, verbose=False)

    assert np.isclose(y_real.min(), 0.0) and np.isclose(y_real.max(), 1.0)
    assert y_gen[:, 0].min() > 1.0, "generated kappa should sit outside the fitted range"


def test_generated_data_without_w_gen_is_refused(scenario):
    cfg, params, _ = scenario
    with pytest.raises(KeyError, match="w_gen"):
        prepare_arrays(cfg, {k: v for k, v in params.items() if k != "w_gen"}, verbose=False)


def test_real_only_scenario_skips_generated(scenario):
    cfg, params, _ = scenario
    cfg.generated_csv = None
    (_, _), (X_gen, y_gen), info = prepare_arrays(cfg, params, verbose=False)
    assert X_gen is None and y_gen is None
    assert info["n_generated"] == 0


# ---------------------------------------------------------------------------
# the per-member validation split
# ---------------------------------------------------------------------------


def test_member_validation_split_matches_reference(scenario):
    """Seeded explicitly, so it is reproducible without TensorFlow.

    Worth pinning: the split decides which rows a member is scored on, and the
    pruning threshold is computed from those scores. A different seed base
    would give a different ensemble from the same training data.
    """
    cfg, params, _ = scenario
    (X_real, y_real), _, _ = prepare_arrays(cfg, params, verbose=False)

    for member_id in (0, 1, 17, 59):
        seed = cfg.ensemble_seed_base + member_id
        ref_seed = _RefConfig.SEED_BASE + member_id
        assert seed == ref_seed

        ours = train_test_split(
            X_real, y_real, test_size=cfg.ensemble_val_fraction,
            random_state=seed, shuffle=True,
        )
        ref = train_test_split(
            X_real, y_real, test_size=_RefConfig.VAL_SIZE,
            random_state=ref_seed, shuffle=True,
        )
        for a, b in zip(ours, ref):
            assert np.array_equal(a, b)


def test_seed_base_is_not_the_pipeline_seed():
    """2025, not 42. A single shared seed would repeat the tuning run."""
    cfg = _cfg()
    assert cfg.ensemble_seed_base == 2025
    assert cfg.ensemble_seed_base != cfg.seed


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_ensemble_config_matches_the_notebook():
    """Every value the notebook hard-coded, as it was actually run."""
    cfg = _cfg()
    assert cfg.ensemble_size_raw == _RefConfig.RAW_ENSEMBLE_SIZE
    assert cfg.ensemble_size_max_kept == _RefConfig.SELECT_MAX_MODELS
    assert cfg.ensemble_size_min_kept == _RefConfig.SELECT_MIN_MODELS
    assert cfg.ensemble_seed_base == _RefConfig.SEED_BASE
    assert cfg.ensemble_epochs == _RefConfig.EPOCHS
    assert cfg.ensemble_val_fraction == _RefConfig.VAL_SIZE
    assert cfg.ensemble_early_stopping_patience == _RefConfig.EARLYSTOP_PATIENCE
    assert cfg.ensemble_lr_patience == _RefConfig.LR_PATIENCE
    assert cfg.ensemble_lr_factor == _RefConfig.LR_FACTOR
    assert cfg.prune_mad_factor == _RefConfig.MAD_FACTOR
    assert cfg.prune_iqr_factor == _RefConfig.IQR_FACTOR
    assert cfg.prune_targetwise_mae == _RefConfig.USE_TARGETWISE_MAE_FILTER
    assert cfg.prune_kappa_floor == _RefConfig.USE_VAL_KXX_FLOOR_FILTER
    assert cfg.kappa_member_floor == _RefConfig.KXX_MEMBER_FLOOR
    assert cfg.kappa_floor_quantile == _RefConfig.KXX_FLOOR_QUANTILE
    assert cfg.kappa_floor_min == _RefConfig.KXX_FLOOR_MIN
    assert cfg.allow_fallback_to_min_models == _RefConfig.ALLOW_FALLBACK_TO_MIN_MODELS


def test_ensemble_dir_is_what_selection_reads():
    """The two stages must agree on the path without being told twice."""
    from ane.config import PipelineConfig
    from ane.surrogate import ensemble_dir

    cfg = PipelineConfig()
    assert str(ensemble_dir(cfg.surrogate)).replace("\\", "/") == cfg.selection.ensemble_dir


def test_load_table_drops_incomplete_rows(tmp_path):
    df = _synthetic_frame(20, seed=5)
    df.loc[3, "kxx"] = np.nan
    df.loc[7, "Fe"] = np.nan
    df.to_csv(tmp_path / "t.csv", index=False)

    out = load_table(tmp_path / "t.csv", "t")
    assert len(out) == 18


def test_load_table_rejects_missing_columns(tmp_path):
    df = _synthetic_frame(10, seed=6).drop(columns=["Pt"])
    df.to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(KeyError):
        load_table(tmp_path / "t.csv", "t")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ===========================================================================
# hyperparameter search (Algorithm S3), against DNN_GAN.ipynb
# ===========================================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from ane.features import featurize  # noqa: E402
from ane.surrogate import (  # noqa: E402
    _fold_arrays,
    check_best_params,
    metrics_full,
    scenario_configs,
    suggest_params,
)


class _RefTuneConfig:
    """Globals from DNN_GAN.ipynb, as the released notebook has them."""

    SEED = 42
    K_FOLDS = 5
    N_TRIALS = 100
    EPOCHS_TUNE = 100
    EPOCHS_FINAL = 200
    TUNE_L2 = True
    L2_FIXED = 1e-4

    # objective_dnn search space
    NUM_LAYERS = (1, 8)
    NUM_NEURONS = (16, 128)          # log
    LEARNING_RATE = (1e-4, 3e-3)     # log
    L2 = (1e-6, 1e-3)                # log
    DROPOUT = (0.0, 0.2)
    BATCH_SIZE = [4, 8, 16, 32]
    W_GEN = (0.01, 0.3)              # log

    TUNE_EARLYSTOP_PATIENCE = 10
    TUNE_LR_PATIENCE = 5
    FINAL_EARLYSTOP_PATIENCE = 80
    FINAL_LR_PATIENCE = 15
    LR_FACTOR = 0.8
    MIN_LR = 1e-6


def _ref_get_metrics_full(y_true, y_pred, target_cols=TARGET_COLS):
    mse_vec = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mae_vec = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2_vec = r2_score(y_true, y_pred, multioutput="raw_values")

    out = {
        "rmse_mean": float(np.sqrt(mse_vec).mean()),
        "mae_mean": float(mae_vec.mean()),
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }
    for i, t in enumerate(target_cols):
        out[f"rmse_{t}"] = float(np.sqrt(mse_vec[i]))
        out[f"mae_{t}"] = float(mae_vec[i])
        out[f"r2_{t}"] = float(r2_vec[i])
    return out


def _ref_fold_scaling(Xr_comp, yr, tr_idx, val_idx, Xg_comp, yg, w_gen):
    """The scaling block inside objective_dnn, transcribed."""
    X_tr_real_raw = _ref_ilr_concat(Xr_comp[tr_idx])
    X_val_real_raw = _ref_ilr_concat(Xr_comp[val_idx])

    sx = MinMaxScaler().fit(X_tr_real_raw)
    sy = MinMaxScaler().fit(yr[tr_idx])

    X_tr_real = sx.transform(X_tr_real_raw)
    y_tr_real = sy.transform(yr[tr_idx])
    X_val_real = sx.transform(X_val_real_raw)
    y_val_real = sy.transform(yr[val_idx])

    if Xg_comp is not None and len(Xg_comp) > 0:
        X_gen_s = sx.transform(_ref_ilr_concat(Xg_comp))
        y_gen_s = sy.transform(yg)
        X_tr_all = np.vstack([X_tr_real, X_gen_s])
        y_tr_all = np.vstack([y_tr_real, y_gen_s])
        sw = np.concatenate([
            np.ones(len(X_tr_real), dtype=np.float32),
            np.full(len(X_gen_s), w_gen, dtype=np.float32),
        ])
    else:
        X_tr_all, y_tr_all, sw = X_tr_real, y_tr_real, None

    return X_tr_all, y_tr_all, X_val_real, y_val_real, sw


def _ref_ilr_concat(X_comp8):
    """DNN_GAN's featurization: float32 at the intermediate steps, unlike the
    other notebooks. Transcribed to check the cast does not change anything."""
    import math

    R_GAS = 8.31446261815324
    props = {
        "Fe": {"radius": 1.26, "vec": 8.0, "weight": 55.845, "en": 1.83},
        "Co": {"radius": 1.25, "vec": 9.0, "weight": 58.933195, "en": 1.88},
        "Mn": {"radius": 1.37, "vec": 7.0, "weight": 54.938044, "en": 1.55},
        "Ga": {"radius": 1.408, "vec": 3.0, "weight": 69.723, "en": 1.81},
        "Al": {"radius": 1.429, "vec": 3.0, "weight": 26.9815385, "en": 1.61},
        "Si": {"radius": 1.316, "vec": 4.0, "weight": 28.085, "en": 1.90},
        "Ge": {"radius": 1.366, "vec": 4.0, "weight": 72.63, "en": 2.01},
        "Pt": {"radius": 1.39, "vec": 10.0, "weight": 195.084, "en": 2.28},
    }

    def closure(A, axis=-1):
        A = np.asarray(A, dtype=np.float64)
        s = A.sum(axis=axis, keepdims=True)
        s[s == 0] = 1.0
        return A / s

    def mult_repl(A, delta=1e-3):
        A = closure(A)
        Z = A == 0
        if not Z.any():
            return A
        B = A.copy()
        for i in range(A.shape[0]):
            m = int(Z[i].sum())
            if m == 0:
                continue
            nz_sum = A[i, ~Z[i]].sum()
            if nz_sum == 0:
                B[i] = 1.0 / A.shape[1]
                continue
            B[i, Z[i]] = delta
            B[i, ~Z[i]] = A[i, ~Z[i]] * ((1.0 - m * delta) / nz_sum)
        return closure(B)

    def helmert(D):
        H = np.zeros((D, D - 1), dtype=np.float64)
        for i in range(1, D):
            a = 1.0 / math.sqrt(i * (i + 1))
            H[:i, i - 1] = a
            H[i, i - 1] = -i * a
        assert np.allclose(H.T @ H, np.eye(D - 1), atol=1e-10)
        return H

    X = mult_repl(X_comp8)
    logx = np.log(closure(X))
    clr = logx - logx.mean(axis=1, keepdims=True)
    ilr = (clr @ helmert(8)).astype(np.float32)          # float32 here

    Xc = closure(np.asarray(X_comp8, dtype=np.float64))
    radius = np.array([props[e]["radius"] for e in COMP_COLS])
    vec = np.array([props[e]["vec"] for e in COMP_COLS])
    weight = np.array([props[e]["weight"] for e in COMP_COLS])
    en = np.array([props[e]["en"] for e in COMP_COLS])

    r_avg = Xc @ radius
    asd = np.sqrt(np.sum(Xc * (1.0 - radius[None, :] / r_avg[:, None]) ** 2, axis=1))
    vec_avg = Xc @ vec
    vec_std = np.sqrt(np.sum(Xc * (vec[None, :] - vec_avg[:, None]) ** 2, axis=1))
    weight_avg = Xc @ weight
    Xs = np.clip(Xc, 1e-12, None)
    smix = -R_GAS * np.sum(Xs * np.log(Xs), axis=1)
    en_avg = Xc @ en
    en_std = np.sqrt(np.sum(Xc * (en[None, :] - en_avg[:, None]) ** 2, axis=1))
    calc = np.column_stack(
        [r_avg, asd, vec_avg, vec_std, weight_avg, smix, en_avg, en_std]
    ).astype(np.float32)                                  # float32 here too

    return np.hstack([ilr, calc]).astype(np.float32)


class _RecordingTrial:
    """Stands in for an Optuna trial and records the space it is asked for."""

    def __init__(self):
        self.calls = []

    def suggest_int(self, name, low, high, log=False):
        self.calls.append(("int", name, low, high, log))
        return low

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(("float", name, low, high, log))
        return low

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, tuple(choices), None, None))
        return choices[0]


# ---------------------------------------------------------------------------
# featurization
# ---------------------------------------------------------------------------


def test_featurize_matches_dnn_gan_ilr_concat():
    """DNN_GAN casts to float32 mid-way; `featurize` casts once at the end.

    Both round the same float64 value to float32 exactly once, so the results
    agree bit for bit -- worth pinning, because the cast looks like it should
    matter and does not.
    """
    X = _synthetic_frame(300, seed=11, with_targets=False)[COMP_COLS].to_numpy(np.float64)
    assert np.array_equal(featurize(X), _ref_ilr_concat(X))


def test_float32_compositions_are_the_only_real_difference():
    """The notebook feeds float32 compositions; the port feeds float64.

    This is the one genuine numerical difference between DNN_GAN and the port,
    and it comes from `to_numpy(dtype=np.float32)` at load time rather than
    from the featurization. It is far below the precision of any measured
    composition, but it is recorded here so the difference is a known quantity
    rather than a surprise.
    """
    X64 = _synthetic_frame(300, seed=12, with_targets=False)[COMP_COLS].to_numpy(np.float64)
    X32 = X64.astype(np.float32)

    assert np.array_equal(featurize(X64), _ref_ilr_concat(X64))
    assert not np.array_equal(featurize(X64), _ref_ilr_concat(X32))

    diff = np.abs(featurize(X64).astype(np.float64) - _ref_ilr_concat(X32).astype(np.float64))
    assert diff.max() < 1e-4


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_full_matches_reference(seed):
    rng = np.random.default_rng(seed)
    y_true = np.column_stack([rng.uniform(2, 20, 40), rng.normal(1, 0.5, 40)])
    y_pred = y_true + rng.normal(0, 0.3, y_true.shape)

    ours = metrics_full(y_true, y_pred)
    ref = _ref_get_metrics_full(y_true, y_pred)
    assert ours.keys() == ref.keys()
    for key in ref:
        assert ours[key] == ref[key], key


def test_rmse_mean_is_the_mean_of_rmse_not_the_rmse_of_pooled_error():
    """The two differ, and the reported figure is the first.

    `sqrt(mse).mean()` averages the per-target root errors; `sqrt(mse.mean())`
    pools first and roots once. With kappa and S_ANE an order of magnitude
    apart, pooling is dominated by kappa, so swapping these would change the
    reported number without changing anything else.
    """
    rng = np.random.default_rng(3)
    y_true = np.column_stack([rng.uniform(2, 20, 60), rng.normal(1, 0.5, 60)])
    y_pred = y_true + np.column_stack([rng.normal(0, 2.0, 60), rng.normal(0, 0.1, 60)])

    m = metrics_full(y_true, y_pred)
    mse_vec = mean_squared_error(y_true, y_pred, multioutput="raw_values")

    assert m["rmse_mean"] == float(np.sqrt(mse_vec).mean())
    assert m["rmse_mean"] != float(np.sqrt(mse_vec.mean()))


def test_r2_mean_is_the_unweighted_mean_of_per_target_r2():
    """`uniform_average`, not `variance_weighted`.

    For two targets the unweighted average coincides with the mean of the
    per-target values, so the distinction is invisible here -- but
    `variance_weighted` would not coincide, and it is the other plausible
    default. Pinned so a change of multioutput mode shows up as a failure.
    """
    rng = np.random.default_rng(3)
    y_true = np.column_stack([rng.uniform(2, 20, 60), rng.normal(1, 0.5, 60)])
    y_pred = y_true + rng.normal(0, 0.4, y_true.shape)

    m = metrics_full(y_true, y_pred)
    assert m["r2_mean"] == pytest.approx((m["r2_kxx"] + m["r2_S_ANE"]) / 2.0)

    weighted = r2_score(y_true, y_pred, multioutput="variance_weighted")
    assert m["r2_mean"] != weighted


# ---------------------------------------------------------------------------
# search space
# ---------------------------------------------------------------------------


def test_search_space_shape_matches_reference():
    """Which parameters are searched, and on which scale."""
    trial = _RecordingTrial()
    suggest_params(trial, _cfg(), has_generated=True)

    by_name = {c[1]: c for c in trial.calls}
    assert set(by_name) == {
        "num_layers", "num_neurons", "learning_rate", "l2", "dropout",
        "batch_size", "w_gen",
    }

    assert by_name["num_layers"][4] is False
    assert by_name["num_neurons"][4] is True
    assert by_name["learning_rate"][4] is True
    assert by_name["l2"][4] is True
    assert by_name["dropout"][4] is False
    assert by_name["w_gen"][4] is True
    assert by_name["batch_size"][2] == tuple(_RefTuneConfig.BATCH_SIZE)


def test_search_bounds_agreeing_with_the_notebook():
    """The four ranges where the configuration and the notebook agree."""
    cfg = _cfg()
    assert (cfg.layers_min, cfg.layers_max) == _RefTuneConfig.NUM_LAYERS
    assert (cfg.learning_rate_min, cfg.learning_rate_max) == _RefTuneConfig.LEARNING_RATE
    assert (cfg.l2_min, cfg.l2_max) == _RefTuneConfig.L2
    assert (cfg.dropout_min, cfg.dropout_max) == _RefTuneConfig.DROPOUT
    assert tuple(cfg.batch_sizes) == tuple(_RefTuneConfig.BATCH_SIZE)
    assert cfg.optuna_trials == _RefTuneConfig.N_TRIALS
    assert cfg.k_folds == _RefTuneConfig.K_FOLDS
    assert cfg.seed == _RefTuneConfig.SEED


def test_search_bounds_superseding_the_surviving_notebook():
    """Two ranges where the package deliberately does not follow the notebook.

    The surviving DNN_GAN.ipynb was edited after the campaign, during
    exploratory testing, and its narrower `num_neurons` and `w_gen` ranges are
    from that editing rather than from the run behind the reported results.
    The configuration carries the campaign values.

    Both sides are pinned. The test fails if the configuration drifts to the
    notebook's numbers -- which would silently release bounds that cannot
    reproduce the results -- and equally if the transcribed notebook values are
    "corrected" to match, which would erase the evidence that they differ.
    """
    cfg = _cfg()
    assert (cfg.neurons_min, cfg.neurons_max) == (16, 256)
    assert (cfg.w_gen_min, cfg.w_gen_max) == (0.01, 0.70)

    assert _RefTuneConfig.NUM_NEURONS == (16, 128)
    assert _RefTuneConfig.W_GEN == (0.01, 0.3)

    # a result from the campaign must be admissible under the campaign bounds
    campaign_best = {"num_neurons": 200, "w_gen": 0.55, "batch_size": 16}
    assert check_best_params(campaign_best, cfg) == []


def test_real_only_scenario_fixes_w_gen():
    trial = _RecordingTrial()
    params = suggest_params(trial, _cfg(), has_generated=False)
    assert params["w_gen"] == 1.0
    assert "w_gen" not in {c[1] for c in trial.calls}


def test_l2_can_be_fixed_like_the_reference():
    trial = _RecordingTrial()
    params = suggest_params(trial, _cfg(tune_l2=False), has_generated=True)
    assert params["l2"] == _RefTuneConfig.L2_FIXED
    assert "l2" not in {c[1] for c in trial.calls}


def test_training_schedule_matches_the_notebook():
    cfg = _cfg()
    assert cfg.epochs_tune == _RefTuneConfig.EPOCHS_TUNE
    assert cfg.epochs_final == _RefTuneConfig.EPOCHS_FINAL
    assert cfg.early_stopping_patience == _RefTuneConfig.TUNE_EARLYSTOP_PATIENCE
    assert cfg.tune_lr_patience == _RefTuneConfig.TUNE_LR_PATIENCE
    assert cfg.final_early_stopping_patience == _RefTuneConfig.FINAL_EARLYSTOP_PATIENCE
    assert cfg.final_lr_patience == _RefTuneConfig.FINAL_LR_PATIENCE
    assert cfg.lr_factor == _RefTuneConfig.LR_FACTOR
    assert cfg.lr_min == _RefTuneConfig.MIN_LR


def test_tuning_and_ensemble_patience_are_different_values():
    """They were a single field once; 10 belongs to the search, 50 to the
    ensemble, and DNN_GAN is where the 10 is confirmed."""
    cfg = _cfg()
    assert cfg.early_stopping_patience == 10
    assert cfg.ensemble_early_stopping_patience == 50


# ---------------------------------------------------------------------------
# best_params validation
# ---------------------------------------------------------------------------


def test_check_best_params_accepts_a_consistent_result():
    params = {
        "num_layers": 3, "num_neurons": 96, "learning_rate": 1e-3,
        "l2": 1e-5, "dropout": 0.1, "batch_size": 16, "w_gen": 0.2,
    }
    assert check_best_params(params, _cfg()) == []


def test_check_best_params_catches_the_neurons_question():
    """A best of 200 neurons could only come from a search whose ceiling was
    above 128, which is what makes a recorded result evidence."""
    params = {"num_neurons": 200, "batch_size": 16}
    assert check_best_params(params, _cfg(neurons_max=128)) != []
    assert check_best_params(params, _cfg(neurons_max=256)) == []


def test_check_best_params_catches_w_gen_and_batch_size():
    assert check_best_params({"w_gen": 0.5}, _cfg(w_gen_max=0.3)) != []
    assert check_best_params({"batch_size": 64}, _cfg()) != []


def test_check_best_params_ignores_absent_keys():
    assert check_best_params({}, _cfg()) == []


# ---------------------------------------------------------------------------
# fold scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_generated", [True, False])
def test_fold_arrays_match_reference(with_generated):
    df_train = _synthetic_frame(57, seed=21)
    df_gen = _synthetic_frame(200, seed=22)

    X_comp = df_train[COMP_COLS].to_numpy(np.float64)
    y = df_train[TARGET_COLS].to_numpy(np.float64)
    Xg = df_gen[COMP_COLS].to_numpy(np.float64) if with_generated else None
    yg = df_gen[TARGET_COLS].to_numpy(np.float64) if with_generated else None

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kfold.split(X_comp):
        ours = _fold_arrays(X_comp, y, tr_idx, val_idx, Xg, yg, 0.2)
        ref = _ref_fold_scaling(X_comp, y, tr_idx, val_idx, Xg, yg, 0.2)
        for a, b in zip(ours, ref):
            if a is None or b is None:
                assert a is None and b is None
            else:
                assert np.array_equal(a, b)


def test_fold_scalers_see_only_the_fold_training_rows():
    """Section 2.4, at the fold level rather than the run level."""
    df_train = _synthetic_frame(57, seed=23)
    X_comp = df_train[COMP_COLS].to_numpy(np.float64)
    y = df_train[TARGET_COLS].to_numpy(np.float64)

    df_gen = _synthetic_frame(100, seed=24)
    df_gen["kxx"] = df_gen["kxx"] * 20.0 + 900.0
    Xg = df_gen[COMP_COLS].to_numpy(np.float64)
    yg = df_gen[TARGET_COLS].to_numpy(np.float64)

    tr_idx, val_idx = next(iter(KFold(5, shuffle=True, random_state=42).split(X_comp)))
    X_tr, y_tr, X_val, y_val, sw = _fold_arrays(X_comp, y, tr_idx, val_idx, Xg, yg, 0.2)

    n_real = len(tr_idx)
    assert np.isclose(y_tr[:n_real, 0].min(), 0.0)
    assert np.isclose(y_tr[:n_real, 0].max(), 1.0)
    assert y_tr[n_real:, 0].min() > 1.0
    assert np.array_equal(sw[:n_real], np.ones(n_real, dtype=np.float32))
    assert np.array_equal(sw[n_real:], np.full(len(Xg), 0.2, dtype=np.float32))


def test_folds_are_reproducible_from_the_seed():
    X = _synthetic_frame(57, seed=25)[COMP_COLS].to_numpy(np.float64)
    a = [tuple(v) for _, v in KFold(5, shuffle=True, random_state=42).split(X)]
    b = [tuple(v) for _, v in KFold(5, shuffle=True, random_state=42).split(X)]
    assert a == b
    assert sorted(i for fold in a for i in fold) == list(range(len(X)))


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def test_scenario_configs_cover_every_size_plus_the_control():
    configs = scenario_configs(_cfg(), (100, 200, 300, 400, 500))
    assert len(configs) == 6

    generated = [c.generated_csv for c in configs]
    assert generated[-1] is None, "the real-only control must be present"
    assert all(f"n{n}" in str(g) for n, g in zip((100, 200, 300, 400, 500), generated[:-1]))

    dirs = [c.artifact_dir for c in configs]
    assert len(set(dirs)) == len(dirs), "scenarios must not share an artifact directory"


def test_scenario_configs_do_not_mutate_the_original():
    cfg = _cfg()
    before = cfg.artifact_dir
    scenario_configs(cfg, (100, 200))
    assert cfg.artifact_dir == before
