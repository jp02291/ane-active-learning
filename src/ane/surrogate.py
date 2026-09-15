"""Surrogate model: hyperparameter search and deep ensemble.

This module covers two stages that share a model definition.

Algorithm S3, the Optuna search, picks the architecture and the weight given to
generatively augmented samples, one search per augmentation scenario. It then
runs a K-fold pass to decide how many epochs to train for, and refits on the
whole training set for exactly that many. That refit -- `final_best_model.h5`
-- is the model whose accuracy the manuscript reports.

Algorithm S4 reads the same parameters and builds the ensemble the *selection*
stage uses. The two are not the same model and are not used for the same
thing: reported accuracy comes from the single refit model, while the ensemble
exists to supply the disagreement that drives exploration. Sixty members are
trained from different seeds and
different real-only validation splits, then pruned to the members that are not
outliers. What survives is written as `ensemble_00.h5`, `ensemble_01.h5`, ...
alongside the two scalers, which is exactly what `ane.select` expects to load.

Why sixty and then prune. The spread across members is one of the three
objectives in the selection stage, so a member that trained badly does not
merely add noise to the mean -- it inflates the disagreement and pulls
candidates onto the Pareto front for the wrong reason. With sixty members and
a dataset of a few dozen rows, a handful land in bad optima every run. They are
removed by a robust criterion (median + 2.5 MAD of validation MAE) rather than
a fixed cutoff, because the scale of a reasonable MAE changes from cycle to
cycle as the dataset grows.

Two details worth knowing before reading the numbers.

The scalers are fit on the real training split only, and generated samples are
transformed with them, never fit on. That is the claim made in section 2.4 of
the manuscript and it holds here.

Each member's own validation split is carved out of the training data *after*
scaling, so the fitted min and max saw those rows. This affects member pruning
slightly -- validation MAE is marginally optimistic -- but not any reported
metric: the test-set evaluation at the end uses the same train-fitted scalers
and touches `test.csv` only through `transform`.

`tests/test_surrogate_parity.py` checks the deterministic parts against the
original notebook. Training itself is not compared; TensorFlow does not promise
bitwise reproducibility across machines.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import SurrogateConfig
from .data import PROPERTY_COLUMNS
from .elements import ELEMENTS
from .features import featurize

__all__ = [
    "set_global_seed",
    "metrics_full",
    "suggest_params",
    "check_best_params",
    "cv_objective_loss",
    "run_search",
    "select_final_epoch",
    "train_final_model",
    "tune_scenario",
    "run_scenarios",
    "load_best_params",
    "load_table",
    "prepare_arrays",
    "build_model",
    "robust_upper_threshold",
    "resolve_kappa_floor",
    "add_selection_flags",
    "select_members",
    "train_member",
    "train_ensemble",
    "evaluate_on_test",
]

COMP_COLS: list[str] = list(ELEMENTS)
TARGET_COLS: list[str] = list(PROPERTY_COLUMNS)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def ensemble_dir(cfg: SurrogateConfig) -> Path:
    """Where the pruned ensemble is written; what `ane.select` reads."""
    return Path(cfg.artifact_dir) / "ensemble_trained"


def raw_member_dir(cfg: SurrogateConfig) -> Path:
    """Where all sixty members are kept before pruning.

    Retained rather than deleted: the pruning criterion is computed from the
    whole population, so re-deciding it later without retraining requires the
    members that were dropped.
    """
    return Path(cfg.artifact_dir) / "ensemble_raw60"


def scaler_paths(cfg: SurrogateConfig) -> tuple[Path, Path]:
    d = ensemble_dir(cfg)
    return d / "sx_scaler.joblib", d / "sy_scaler.joblib"


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow together.

    Called once per ensemble member rather than once per run: each member is
    meant to differ from its neighbours only through its seed, so the seed has
    to be re-set before the member's validation split is drawn, not before the
    loop.
    """
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_best_params(cfg: SurrogateConfig) -> dict[str, Any]:
    """Read `best_params.json` produced by the Optuna stage."""
    path = Path(cfg.artifact_dir) / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"best_params.json not found in {cfg.artifact_dir}. Run the "
            f"hyperparameter search for this scenario first."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_table(path: str | Path, label: str) -> pd.DataFrame:
    """Read a composition-property table, dropping incomplete rows.

    Deliberately more permissive than `ane.data.load_dataset`, which raises on
    a missing property. Generated tables can carry rows the filters left
    incomplete, and here they are simply dropped; the measured data has already
    been validated strictly by stage 0.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(path)
    required = COMP_COLS + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing columns {missing}: {path}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{label} has no complete rows: {path}")
    return df


def prepare_arrays(
    cfg: SurrogateConfig, params: dict[str, Any], verbose: bool = True
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray | None, np.ndarray | None],
    dict[str, int],
]:
    """Featurize, fit the scalers on real data, and write them to disk.

    Returns ((X_real, y_real), (X_gen, y_gen), info). The scalers are saved
    here rather than at the end because `ane.select` loads them from this
    directory and they must describe the space the members were trained in.
    """
    import joblib

    df_train = load_table(cfg.train_csv, "train csv")

    if cfg.generated_csv is not None:
        df_gen = load_table(cfg.generated_csv, "generated csv")
        if "w_gen" not in params:
            raise KeyError(
                "generated data is configured but best_params.json has no "
                "'w_gen'. Refusing to default the augmented samples to weight "
                "1.0: the weight is a tuned quantity and silently assuming "
                "parity with measured data would change what is being reported."
            )
    else:
        df_gen = pd.DataFrame(columns=COMP_COLS + TARGET_COLS)

    X_real_raw = featurize(df_train[COMP_COLS].to_numpy(dtype=np.float64))
    y_real_raw = df_train[TARGET_COLS].to_numpy(dtype=np.float64)

    # fit on measured data only; generated samples get transform, never fit
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    X_real = sx.fit_transform(X_real_raw)
    y_real = sy.fit_transform(y_real_raw)

    out = ensemble_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    sx_path, sy_path = scaler_paths(cfg)
    joblib.dump(sx, sx_path)
    joblib.dump(sy, sy_path)

    X_gen = y_gen = None
    if len(df_gen) > 0:
        X_gen = sx.transform(featurize(df_gen[COMP_COLS].to_numpy(dtype=np.float64)))
        y_gen = sy.transform(df_gen[TARGET_COLS].to_numpy(dtype=np.float64))

    info = {
        "n_train_real": int(len(df_train)),
        "n_generated": int(len(df_gen)),
        "input_dim": int(X_real.shape[1]),
        "target_dim": int(y_real.shape[1]),
    }
    if verbose:
        print(f"  {info['n_train_real']} measured rows, {info['n_generated']} generated")
        print(f"  {info['input_dim']} features -> {info['target_dim']} targets")

    return (X_real, y_real), (X_gen, y_gen), info


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_model(input_dim: int, params: dict[str, Any], n_targets: int = 2):
    """Build and compile one surrogate from a parameter dictionary.

    Shared between the Optuna search and the ensemble: the architecture the
    search scores has to be the architecture the ensemble then trains, and
    keeping one builder is what guarantees it. Returns (model, batch_size),
    since the batch size is tuned alongside the architecture but is not part
    of the model object.
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    num_layers = int(params.get("num_layers", 3))
    num_neurons = int(params.get("num_neurons", 64))
    dropout = float(params.get("dropout", 0.0))
    l2 = float(params.get("l2", 0.0))
    learning_rate = float(params.get("learning_rate", 1e-3))

    model = Sequential([tf.keras.Input(shape=(input_dim,))])
    for _ in range(num_layers):
        model.add(
            Dense(
                num_neurons,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2),
            )
        )
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(n_targets, activation="linear"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return model, int(params.get("batch_size", 32))


# ---------------------------------------------------------------------------
# member pruning
# ---------------------------------------------------------------------------


def robust_upper_threshold(
    values: np.ndarray, mad_factor: float, iqr_factor: float
) -> float:
    """Upper cutoff for "this member is an outlier", resistant to the outliers.

    median + mad_factor * MAD, since the members being screened out are
    precisely the ones that would inflate a mean and a standard deviation. When
    the MAD is degenerate -- members agreeing almost exactly, which happens on
    an easy cycle -- falls back to Q3 + iqr_factor * IQR, and if that is
    degenerate too, returns infinity so that this metric filters nothing rather
    than filtering arbitrarily.
    """
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


def resolve_kappa_floor(
    df_train: pd.DataFrame, cfg: SurrogateConfig, verbose: bool = True
) -> float:
    """Lowest kappa a member may predict on its validation split.

    Data-derived rather than fixed: the 5th percentile of the measured training
    kappa, but never below `kappa_floor_min`. A member predicting below the
    range of anything ever measured has found a degenerate solution, and since
    the selection stage maximizes 1/kappa such a member would otherwise steer
    the search straight at its own artefact.
    """
    if cfg.kappa_member_floor is not None:
        return float(cfg.kappa_member_floor)

    kappa = df_train[TARGET_COLS[0]].to_numpy(dtype=float)
    kappa = kappa[np.isfinite(kappa)]
    if len(kappa) == 0:
        return float(cfg.kappa_floor_min)

    q_floor = float(np.quantile(kappa, cfg.kappa_floor_quantile))
    floor = max(q_floor, float(cfg.kappa_floor_min))

    if verbose:
        print(
            f"  kappa floor {floor:.6g} "
            f"(train q{cfg.kappa_floor_quantile:.2f} = {q_floor:.6g}, "
            f"hard minimum {cfg.kappa_floor_min:.6g})"
        )
    return floor


def add_selection_flags(
    df: pd.DataFrame, cfg: SurrogateConfig, kappa_floor: float
) -> pd.DataFrame:
    """Annotate the member table with the thresholds and pass/fail flags.

    The thresholds are stored on every row, not just applied, so that the
    manifest records why a given member was dropped.
    """
    df = df.copy()

    thr_mean = robust_upper_threshold(
        df["val_mae_mean"].to_numpy(), cfg.prune_mad_factor, cfg.prune_iqr_factor
    )
    thr_kappa = robust_upper_threshold(
        df["val_mae_kxx"].to_numpy(), cfg.prune_mad_factor, cfg.prune_iqr_factor
    )
    thr_s = robust_upper_threshold(
        df["val_mae_S_ANE"].to_numpy(), cfg.prune_mad_factor, cfg.prune_iqr_factor
    )

    df["threshold_val_mae_mean"] = thr_mean
    df["threshold_val_mae_kxx"] = thr_kappa
    df["threshold_val_mae_S_ANE"] = thr_s

    df["pass_val_mae_mean"] = df["val_mae_mean"] <= thr_mean

    if cfg.prune_targetwise_mae:
        # a member can look fine on the mean while being badly wrong on one
        # target, and the two targets are used separately downstream
        df["pass_val_mae_kxx"] = df["val_mae_kxx"] <= thr_kappa
        df["pass_val_mae_S_ANE"] = df["val_mae_S_ANE"] <= thr_s
    else:
        df["pass_val_mae_kxx"] = True
        df["pass_val_mae_S_ANE"] = True

    if cfg.prune_kappa_floor:
        df["pass_val_kxx_floor"] = df["val_pred_kxx_min"] > kappa_floor
    else:
        df["pass_val_kxx_floor"] = True

    df["pass_all_filters"] = (
        df["pass_val_mae_mean"]
        & df["pass_val_mae_kxx"]
        & df["pass_val_mae_S_ANE"]
        & df["pass_val_kxx_floor"]
    )
    return df


def select_members(
    df_metrics: pd.DataFrame, cfg: SurrogateConfig, verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose which members become the ensemble, and record how."""
    df = df_metrics.sort_values("val_mae_mean", ascending=True).reset_index(drop=True)

    selected = df[df["pass_all_filters"]].copy()
    mode = "robust_filters"

    if len(selected) > cfg.ensemble_size_max_kept:
        selected = (
            selected.sort_values("val_mae_mean", ascending=True)
            .head(cfg.ensemble_size_max_kept)
            .copy()
        )
        mode = "robust_filters_top_by_val_mae"

    if len(selected) < cfg.ensemble_size_min_kept:
        message = (
            f"only {len(selected)} of {len(df)} members passed the robust "
            f"filters; {cfg.ensemble_size_min_kept} are required"
        )
        if not cfg.allow_fallback_to_min_models:
            raise RuntimeError(message)
        if verbose:
            print(f"  warning: {message}")
            print("  falling back to the best members by validation MAE")
        pool = df[np.isfinite(df["val_mae_mean"])].copy()
        selected = (
            pool.sort_values("val_mae_mean", ascending=True)
            .head(cfg.ensemble_size_min_kept)
            .copy()
        )
        mode = "fallback_best_val_mae_to_min_models"

    selected = selected.sort_values("val_mae_mean", ascending=True).reset_index(drop=True)
    selected["selected_rank"] = np.arange(len(selected), dtype=int)

    info: dict[str, Any] = {
        "selection_mode": mode,
        "n_raw_members": int(len(df_metrics)),
        "n_pass_all_filters": int(df_metrics["pass_all_filters"].sum()),
        "n_selected": int(len(selected)),
        "select_min_models": int(cfg.ensemble_size_min_kept),
        "select_max_models": int(cfg.ensemble_size_max_kept),
        "mad_factor": float(cfg.prune_mad_factor),
        "iqr_factor": float(cfg.prune_iqr_factor),
    }
    return selected, info


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target R2, NaN where it is undefined rather than raising.

    A member's validation split can be small enough that one target has no
    variance, which is a fact about that split and not a reason to abort a
    sixty-member run.
    """
    try:
        return r2_score(y_true, y_pred, multioutput="raw_values")
    except Exception:
        return np.full(y_true.shape[1], np.nan, dtype=float)


def train_member(
    member_id: int,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_gen: np.ndarray | None,
    y_gen: np.ndarray | None,
    params: dict[str, Any],
    cfg: SurrogateConfig,
    kappa_floor: float,
) -> dict[str, Any]:
    """Train one member and return its validation metrics.

    The validation split is real data only. Generated samples go into training
    with weight `w_gen`, never into validation: a member selected on its
    ability to fit synthetic data would be selected on the generator's biases.
    """
    import joblib
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    seed = cfg.ensemble_seed_base + member_id
    set_global_seed(seed)

    X_tr_real, X_val, y_tr_real, y_val = train_test_split(
        X_real,
        y_real,
        test_size=cfg.ensemble_val_fraction,
        random_state=seed,
        shuffle=True,
    )

    if X_gen is not None and y_gen is not None:
        w_gen = float(params["w_gen"])
        X_tr = np.vstack([X_tr_real, X_gen])
        y_tr = np.vstack([y_tr_real, y_gen])
        sample_weight = np.concatenate(
            [
                np.ones(len(X_tr_real), dtype=np.float32),
                np.full(len(X_gen), w_gen, dtype=np.float32),
            ]
        )
    else:
        w_gen = None
        X_tr, y_tr, sample_weight = X_tr_real, y_tr_real, None

    model, batch_size = build_model(X_real.shape[1], params, n_targets=len(TARGET_COLS))

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg.ensemble_epochs,
        batch_size=batch_size,
        sample_weight=sample_weight,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.ensemble_early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.ensemble_lr_factor,
                patience=cfg.ensemble_lr_patience,
                verbose=0,
            ),
        ],
        verbose=0,
    )

    # metrics in physical units, not scaled ones: the pruning thresholds are
    # meant to be readable against the measured data
    _, sy_path = scaler_paths(cfg)
    sy = joblib.load(sy_path)
    y_val_true = sy.inverse_transform(y_val)
    y_val_pred = sy.inverse_transform(model.predict(X_val, verbose=0))

    mae_each = mean_absolute_error(y_val_true, y_val_pred, multioutput="raw_values")
    rmse_each = np.sqrt(
        mean_squared_error(y_val_true, y_val_pred, multioutput="raw_values")
    )
    r2_each = _safe_r2(y_val_true, y_val_pred)

    val_loss = np.asarray(history.history.get("val_loss", []), dtype=float)
    train_loss = np.asarray(history.history.get("loss", []), dtype=float)

    raw_dir = raw_member_dir(cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"member_raw_{member_id:02d}.h5"
    model.save(raw_path)

    record: dict[str, Any] = {
        "member_id": int(member_id),
        "seed": int(seed),
        "raw_model_path": str(raw_path),
        "epochs_trained": int(len(train_loss)),
        "best_val_loss_scaled": float(np.nanmin(val_loss)) if len(val_loss) else np.nan,
        "final_val_loss_scaled": float(val_loss[-1]) if len(val_loss) else np.nan,
        "final_train_loss_scaled": float(train_loss[-1]) if len(train_loss) else np.nan,
        "val_mae_mean": float(np.mean(mae_each)),
        "val_mae_kxx": float(mae_each[0]),
        "val_mae_S_ANE": float(mae_each[1]),
        "val_rmse_kxx": float(rmse_each[0]),
        "val_rmse_S_ANE": float(rmse_each[1]),
        "val_r2_kxx": float(r2_each[0]) if np.isfinite(r2_each[0]) else np.nan,
        "val_r2_S_ANE": float(r2_each[1]) if np.isfinite(r2_each[1]) else np.nan,
        "val_pred_kxx_min": float(np.min(y_val_pred[:, 0])),
        "val_pred_kxx_mean": float(np.mean(y_val_pred[:, 0])),
        "val_pred_kxx_nonpositive_count": int(np.sum(y_val_pred[:, 0] <= 0)),
        "val_pred_kxx_below_floor_count": int(np.sum(y_val_pred[:, 0] <= kappa_floor)),
        "kxx_member_floor_used": float(kappa_floor),
        "w_gen": None if w_gen is None else float(w_gen),
    }

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return record


def save_selected_models(selected: pd.DataFrame, cfg: SurrogateConfig) -> list[str]:
    """Copy the kept members into the ensemble directory, renumbered.

    Renumbered contiguously from zero because `ane.select` discovers members by
    matching `ensemble_(\\d+).h5`; a gap would not break it, but the index would
    stop meaning rank. The originals stay in the raw directory.
    """
    out = ensemble_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("ensemble_*.h5"):
        stale.unlink()

    paths = []
    for new_id, (_, row) in enumerate(selected.iterrows()):
        dst = out / f"ensemble_{new_id:02d}.h5"
        shutil.copy2(row["raw_model_path"], dst)
        paths.append(str(dst))
    return paths


def evaluate_on_test(
    model_paths: list[str], cfg: SurrogateConfig, verbose: bool = True
) -> dict[str, Any]:
    """Evaluate the pruned ensemble on the held-out test split.

    These are the numbers reported in the manuscript. No held-out target value
    enters generator training, hyperparameter search or member pruning. The
    compositions are read once more, in the duplicate-exclusion step of
    `ane.augment`, so that a generated sample cannot coincide with a held-out
    one; no property of those rows is used there.
    """
    import joblib
    import tensorflow as tf

    if not Path(cfg.test_csv).exists():
        if verbose:
            print(f"  test csv not found ({cfg.test_csv}); skipping evaluation")
        return {}

    df_test = load_table(cfg.test_csv, "test csv")
    y_true = df_test[TARGET_COLS].to_numpy(dtype=np.float64)

    sx_path, sy_path = scaler_paths(cfg)
    sx = joblib.load(sx_path)
    sy = joblib.load(sy_path)

    X = sx.transform(featurize(df_test[COMP_COLS].to_numpy(dtype=np.float64)))

    models = [tf.keras.models.load_model(p, compile=False) for p in model_paths]
    scaled = np.asarray(
        [m(X, training=False).numpy() for m in models], dtype=np.float64
    )
    preds = np.asarray([sy.inverse_transform(p) for p in scaled], dtype=np.float64)

    y_pred = preds.mean(axis=0)
    y_std = preds.std(axis=0)

    r2_each = _safe_r2(y_true, y_pred)
    mae_each = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_each = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))

    metrics: dict[str, Any] = {
        "n_selected_models": int(len(model_paths)),
        "r2_kxx": float(r2_each[0]) if np.isfinite(r2_each[0]) else np.nan,
        "r2_S_ANE": float(r2_each[1]) if np.isfinite(r2_each[1]) else np.nan,
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "mae_kxx": float(mae_each[0]),
        "mae_S_ANE": float(mae_each[1]),
        "rmse_kxx": float(rmse_each[0]),
        "rmse_S_ANE": float(rmse_each[1]),
        "mean_pred_std_kxx": float(y_std[:, 0].mean()),
        "mean_pred_std_S_ANE": float(y_std[:, 1].mean()),
        "member_level_nonpositive_kxx_count": int(np.sum(preds[:, :, 0] <= 0)),
    }

    pred_df = df_test[COMP_COLS + TARGET_COLS].copy()
    pred_df["pred_kxx_mean"] = y_pred[:, 0]
    pred_df["pred_S_ANE_mean"] = y_pred[:, 1]
    pred_df["pred_kxx_std"] = y_std[:, 0]
    pred_df["pred_S_ANE_std"] = y_std[:, 1]
    pred_df["abs_err_kxx"] = np.abs(y_true[:, 0] - y_pred[:, 0])
    pred_df["abs_err_S_ANE"] = np.abs(y_true[:, 1] - y_pred[:, 1])

    out = ensemble_dir(cfg)
    with open(out / "ensemble_test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    pred_df.to_csv(out / "ensemble_test_predictions.csv", index=False)

    if verbose:
        print(json.dumps(metrics, indent=2))

    tf.keras.backend.clear_session()
    gc.collect()

    return metrics


def train_ensemble(cfg: SurrogateConfig, verbose: bool = True) -> dict[str, Any]:
    """Run stage 3: train sixty members, prune, evaluate, write the manifest."""
    params = load_best_params(cfg)
    if verbose:
        print("[1/5] best parameters")
        print(json.dumps(params, indent=2))

    if verbose:
        print("[2/5] preparing data")
    (X_real, y_real), (X_gen, y_gen), data_info = prepare_arrays(cfg, params, verbose)

    raw_dir = raw_member_dir(cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("member_raw_*.h5"):
        stale.unlink()

    df_train = load_table(cfg.train_csv, "train csv")
    kappa_floor = resolve_kappa_floor(df_train, cfg, verbose)

    if verbose:
        print(f"[3/5] training {cfg.ensemble_size_raw} members")
    records = []
    for i in range(cfg.ensemble_size_raw):
        rec = train_member(i, X_real, y_real, X_gen, y_gen, params, cfg, kappa_floor)
        records.append(rec)
        if verbose:
            print(
                f"  member {i + 1:2d}/{cfg.ensemble_size_raw}  "
                f"val MAE {rec['val_mae_mean']:.6g}  "
                f"(kappa {rec['val_mae_kxx']:.6g}, S_ANE {rec['val_mae_S_ANE']:.6g})  "
                f"min kappa {rec['val_pred_kxx_min']:.6g}"
            )

    if verbose:
        print("[4/5] pruning")
    df_metrics = add_selection_flags(pd.DataFrame(records), cfg, kappa_floor)
    selected, selection_info = select_members(df_metrics, cfg, verbose)

    out = ensemble_dir(cfg)
    df_metrics.to_csv(out / "ensemble_member_metrics_all.csv", index=False)
    selected.to_csv(out / "ensemble_member_metrics_selected.csv", index=False)
    model_paths = save_selected_models(selected, cfg)

    selection_info["kappa_member_floor"] = float(kappa_floor)
    selection_info["selected_model_paths"] = model_paths
    if verbose:
        print(f"  kept {len(model_paths)} of {cfg.ensemble_size_raw} "
              f"({selection_info['selection_mode']})")

    if verbose:
        print("[5/5] evaluating on the held-out test split")
    test_metrics = evaluate_on_test(model_paths, cfg, verbose)

    manifest = {
        "artifact_dir": cfg.artifact_dir,
        "output_dir": str(out),
        "raw_model_dir": str(raw_dir),
        "train_csv": cfg.train_csv,
        "test_csv": cfg.test_csv,
        "generated_csv": cfg.generated_csv,
        "raw_ensemble_size": cfg.ensemble_size_raw,
        "selected_ensemble_size": int(selection_info.get("n_selected", 0)),
        "select_min_models": cfg.ensemble_size_min_kept,
        "select_max_models": cfg.ensemble_size_max_kept,
        "ensemble_seed_base": cfg.ensemble_seed_base,
        "epochs": cfg.ensemble_epochs,
        "val_fraction": cfg.ensemble_val_fraction,
        "params": params,
        "data_info": data_info,
        "selection_info": selection_info,
        "test_metrics": test_metrics,
        "feature_definition": "7 ILR coordinates (delta = 1e-3) + 8 atomic descriptors",
        "selection_policy": (
            "train ensemble_size_raw members; drop robust validation-MAE "
            "outliers and members violating the kappa floor; keep at most "
            "ensemble_size_max_kept by validation MAE"
        ),
    }
    with open(out / "ensemble_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nwritten to {out}/")
        print(f"set selection.ensemble_dir = '{out}' for the next stage")

    return manifest


# ---------------------------------------------------------------------------
# hyperparameter search (Algorithm S3)
# ---------------------------------------------------------------------------


def metrics_full(
    y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str] | None = None
) -> dict[str, float]:
    """RMSE, MAE and R2, averaged and per target.

    Reported per target as well as averaged because kappa and S_ANE are an
    order of magnitude apart and an average over them says little about either.

    Two choices here are the ones that would silently change a reported number.
    `rmse_mean` is the mean of the per-target RMSEs, not the RMSE of the pooled
    residuals -- the latter would be dominated by kappa. `r2_mean` is
    sklearn's `uniform_average`, which for two targets is their plain mean;
    `variance_weighted` is the other plausible default and gives a different
    figure.
    """
    target_cols = target_cols or TARGET_COLS
    mse_vec = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mae_vec = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2_vec = r2_score(y_true, y_pred, multioutput="raw_values")

    out = {
        "rmse_mean": float(np.sqrt(mse_vec).mean()),
        "mae_mean": float(mae_vec.mean()),
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }
    for i, name in enumerate(target_cols):
        out[f"rmse_{name}"] = float(np.sqrt(mse_vec[i]))
        out[f"mae_{name}"] = float(mae_vec[i])
        out[f"r2_{name}"] = float(r2_vec[i])
    return out


def suggest_params(trial, cfg: SurrogateConfig, has_generated: bool) -> dict[str, Any]:
    """Draw one point from the search space.

    `num_neurons`, `learning_rate`, `l2` and `w_gen` are log-uniform: each
    spans two or more orders of magnitude, and a uniform draw would put almost
    all the budget in the top decade.

    `w_gen` is only a parameter when there is generated data to weight. In the
    real-only scenario it is fixed rather than sampled, so that scenario's
    search is over a strictly smaller space -- worth remembering when comparing
    scenarios, since the real-only branch is searching fewer dimensions with
    the same trial budget.
    """
    params: dict[str, Any] = {
        "num_layers": trial.suggest_int("num_layers", cfg.layers_min, cfg.layers_max),
        "num_neurons": trial.suggest_int(
            "num_neurons", cfg.neurons_min, cfg.neurons_max, log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", cfg.learning_rate_min, cfg.learning_rate_max, log=True
        ),
        "l2": (
            trial.suggest_float("l2", cfg.l2_min, cfg.l2_max, log=True)
            if cfg.tune_l2
            else cfg.l2_fixed
        ),
        "dropout": trial.suggest_float("dropout", cfg.dropout_min, cfg.dropout_max),
        "batch_size": trial.suggest_categorical("batch_size", list(cfg.batch_sizes)),
    }
    if has_generated:
        params["w_gen"] = trial.suggest_float(
            "w_gen", cfg.w_gen_min, cfg.w_gen_max, log=True
        )
    else:
        params["w_gen"] = cfg.w_gen_without_generated
    return params


#: Recorded parameters and the configuration bounds they must lie within.
_BOUNDED_PARAMS: tuple[tuple[str, str, str], ...] = (
    ("num_layers", "layers_min", "layers_max"),
    ("num_neurons", "neurons_min", "neurons_max"),
    ("learning_rate", "learning_rate_min", "learning_rate_max"),
    ("l2", "l2_min", "l2_max"),
    ("dropout", "dropout_min", "dropout_max"),
    ("w_gen", "w_gen_min", "w_gen_max"),
)


def check_best_params(params: dict[str, Any], cfg: SurrogateConfig) -> list[str]:
    """Report where a recorded result is inconsistent with the search space.

    Exists because the configured bounds and the released notebook disagree on
    `num_neurons` (16-256 here, 16-128 there) and `w_gen` (0.01-0.70 against
    0.01-0.30). A `best_params.json` cannot have come from a search whose range
    excluded it, so the recorded optimum is evidence about which bounds
    actually ran -- though only one-directional: a best of 64 neurons is
    consistent with either range and settles nothing.

    Returns a list of human-readable problems, empty when consistent.
    """
    problems: list[str] = []
    for key, lo_attr, hi_attr in _BOUNDED_PARAMS:
        if key not in params:
            continue
        value = params[key]
        lo, hi = getattr(cfg, lo_attr), getattr(cfg, hi_attr)
        if not (lo <= value <= hi):
            problems.append(
                f"{key}={value!r} lies outside the configured range "
                f"[{lo}, {hi}]; the search that produced it used different bounds"
            )

    if "batch_size" in params and params["batch_size"] not in cfg.batch_sizes:
        problems.append(
            f"batch_size={params['batch_size']!r} is not among the configured "
            f"choices {list(cfg.batch_sizes)}"
        )
    return problems


def _fold_arrays(
    X_comp: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_gen_comp: np.ndarray | None,
    y_gen: np.ndarray | None,
    w_gen: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Featurize and scale one fold, fitting the scalers on its training part.

    This is the step section 2.4 describes. The scalers see the fold's
    training rows only; the fold's validation rows and every generated sample
    are transformed with them. Fitting on the union would leak the validation
    range into the model's input normalization, and fitting on the generated
    samples would let the generator move the scale.

    The generated samples do carry an indirect dependence on the validation
    rows, because the generator was trained once on the whole training split
    before the folds were drawn. That is a property of the pipeline, not of
    this function, and it is stated in the README.
    """
    X_tr_raw = featurize(X_comp[train_idx])
    X_val_raw = featurize(X_comp[val_idx])

    sx = MinMaxScaler().fit(X_tr_raw)
    sy = MinMaxScaler().fit(y[train_idx])

    X_tr = sx.transform(X_tr_raw)
    y_tr = sy.transform(y[train_idx])
    X_val = sx.transform(X_val_raw)
    y_val = sy.transform(y[val_idx])

    if X_gen_comp is not None and len(X_gen_comp) > 0:
        X_gen = sx.transform(featurize(X_gen_comp))
        y_gen_scaled = sy.transform(y_gen)
        sample_weight = np.concatenate(
            [
                np.ones(len(X_tr), dtype=np.float32),
                np.full(len(X_gen), w_gen, dtype=np.float32),
            ]
        )
        X_tr = np.vstack([X_tr, X_gen])
        y_tr = np.vstack([y_tr, y_gen_scaled])
    else:
        sample_weight = None

    return X_tr, y_tr, X_val, y_val, sample_weight


def cv_objective_loss(
    params: dict[str, Any],
    X_comp: np.ndarray,
    y: np.ndarray,
    X_gen_comp: np.ndarray | None,
    y_gen: np.ndarray | None,
    cfg: SurrogateConfig,
) -> float:
    """Mean best validation loss over the folds, for one parameter set.

    The per-fold score is the minimum validation loss reached, not the last:
    early stopping restores the best weights, so the minimum is the loss of the
    model that would actually be kept. A parameter set scores infinity as soon
    as any fold fails to produce a finite loss: averaging the folds that
    survived would score it on fewer than `k_folds`, which is not the objective
    Algorithm S3 defines. Optuna then abandons that region.
    """
    import tensorflow as tf
    from sklearn.model_selection import KFold
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    kfold = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    fold_losses: list[float] = []

    for train_idx, val_idx in kfold.split(X_comp):
        X_tr, y_tr, X_val, y_val, sample_weight = _fold_arrays(
            X_comp, y, train_idx, val_idx, X_gen_comp, y_gen, params.get("w_gen", 1.0)
        )

        # Seeding the sampler and the fold split is not enough: weight
        # initialization, dropout and batch order also decide the score, and
        # without this the same parameter set scores differently on every run,
        # so the search itself is not reproducible. The seed is the same for
        # every trial on purpose -- Optuna is comparing parameter sets, and a
        # trial-dependent seed would put initialization noise into that
        # comparison.
        tf.keras.utils.set_random_seed(cfg.seed)
        model, batch_size = build_model(X_tr.shape[1], params, n_targets=y.shape[1])
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs_tune,
            batch_size=batch_size,
            sample_weight=sample_weight,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=cfg.lr_factor,
                    patience=cfg.tune_lr_patience,
                    min_lr=cfg.lr_min,
                    verbose=0,
                ),
            ],
        )

        losses = history.history.get("val_loss", [])
        if len(losses):
            best = float(np.nanmin(losses))
            if not np.isfinite(best):
                # Averaging the folds that happened to survive would score this
                # parameter set on fewer than `k_folds`, which is not the
                # objective Algorithm S3 defines and would flatter a set that
                # failed somewhere. Abandon the region instead.
                return float("inf")
            fold_losses.append(best)

        del model, history
        K.clear_session()
        gc.collect()

    if len(fold_losses) != cfg.k_folds:
        return float("inf")
    return float(np.mean(fold_losses))


def run_search(
    cfg: SurrogateConfig,
    X_comp: np.ndarray,
    y: np.ndarray,
    X_gen_comp: np.ndarray | None,
    y_gen: np.ndarray | None,
    verbose: bool = True,
) -> tuple[dict[str, Any], float]:
    """Run the Optuna study and write `best_params.json`.

    TPE with a fixed seed. The sampler matters enough to name: TPE builds a
    model of the objective from the trials so far, so the search is sequential
    and the trial order is part of the result. Restarting a study with a
    different trial budget does not give a prefix of the longer run.
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    has_generated = X_gen_comp is not None and len(X_gen_comp) > 0

    def objective(trial):
        params = suggest_params(trial, cfg, has_generated)
        try:
            return cv_objective_loss(params, X_comp, y, X_gen_comp, y_gen, cfg)
        except Exception as exc:  # a failed trial must not abort the study
            if verbose:
                print(f"  trial {trial.number} failed: {exc}")
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=cfg.seed)
    )

    artifact = Path(cfg.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)

    def on_trial_end(study, trial):
        if verbose:
            value = trial.value if trial.value is not None else float("nan")
            print(f"  trial {trial.number:03d}  {value:.6f}  best {study.best_value:.6f}")
        # written every improvement so that a long search killed partway
        # through still leaves the best point found
        with open(artifact / "best_params_live.json", "w", encoding="utf-8") as fh:
            json.dump(study.best_params, fh, indent=2, ensure_ascii=False)

    study.optimize(
        objective,
        n_trials=cfg.optuna_trials,
        callbacks=[on_trial_end],
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    if not has_generated:
        best.setdefault("w_gen", cfg.w_gen_without_generated)

    # The objective value of the winning trial IS the cross-validation score
    # the manuscript defines -- the mean over folds of each fold's minimum
    # validation MSE (`cv_objective_loss`). It is returned rather than
    # discarded because it, and not the single-fold loss that
    # `select_final_epoch` produces, is what ranks the scenarios.
    cv_score = float(study.best_value)

    with open(artifact / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2, ensure_ascii=False, default=_json_default)

    if verbose:
        print(f"  best trial {study.best_trial.number}, value {study.best_value:.6f}")
        print(json.dumps(best, indent=2))

    return best, cv_score


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def select_final_epoch(
    best_params: dict[str, Any],
    X_comp: np.ndarray,
    y: np.ndarray,
    X_gen_comp: np.ndarray | None,
    y_gen: np.ndarray | None,
    cfg: SurrogateConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    """Decide how many epochs the final refit should run for.

    Trains the chosen architecture once per fold with a long patience, and
    takes the epoch count from whichever fold reached the lowest validation
    loss. The architecture does not vary across folds -- it is fixed by the
    search -- so the epoch count is the only thing this pass produces.

    Two caveats, both inherited from the original implementation. The fold
    scores are compared directly although each fold has its own target scaler,
    so they are not strictly on the same scale; and taking the minimum over
    folds rather than the mean makes the chosen epoch count the one that suited
    a single fold best.

    The effect is confined to how long the final model trains. The loss
    returned here is reported as a diagnostic under `final_epoch_fold_loss` and
    is deliberately not used to rank scenarios: that is done with the
    cross-validation score defined in the manuscript, which `run_search`
    returns.
    """
    from sklearn.model_selection import KFold
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    import tensorflow as tf

    kfold = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    w_gen = best_params.get("w_gen", cfg.w_gen_without_generated)

    best = {"val_loss": np.inf, "fold": None, "best_epoch": cfg.epochs_final}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_comp), start=1):
        X_tr, y_tr, X_val, y_val, sample_weight = _fold_arrays(
            X_comp, y, train_idx, val_idx, X_gen_comp, y_gen, w_gen
        )
        if not np.all(np.isfinite(X_tr)) or not np.all(np.isfinite(y_tr)):
            raise ValueError(f"fold {fold} produced non-finite scaled arrays")

        tf.keras.utils.set_random_seed(cfg.seed)
        model, batch_size = build_model(X_tr.shape[1], best_params, n_targets=y.shape[1])

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            sample_weight=sample_weight,
            epochs=cfg.epochs_final,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.final_early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=cfg.lr_factor,
                    patience=cfg.final_lr_patience,
                    min_lr=cfg.lr_min,
                    verbose=0,
                ),
            ],
        )

        val_loss = float(mean_squared_error(y_val, model(X_val, training=False).numpy()))
        losses = history.history.get("val_loss", [])
        epoch = int(np.argmin(losses) + 1) if len(losses) else cfg.epochs_final

        if verbose:
            print(f"  fold {fold}  val_loss {val_loss:.6f}  best epoch {epoch}")

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "fold": fold, "best_epoch": epoch}

        del model, history
        K.clear_session()
        gc.collect()

    if verbose:
        print(f"  taking {best['best_epoch']} epochs from fold {best['fold']}")
    return best


def train_final_model(
    best_params: dict[str, Any],
    epochs: int,
    X_comp: np.ndarray,
    y: np.ndarray,
    X_gen_comp: np.ndarray | None,
    y_gen: np.ndarray | None,
    X_test_comp: np.ndarray,
    y_test: np.ndarray,
    cfg: SurrogateConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    """Refit on all training data for a fixed number of epochs and evaluate.

    No validation split and no callbacks: the epoch count is already decided,
    and holding data back here would mean the reported model was trained on
    less data than the campaign had. The evaluation that follows is the only
    read of `test.csv` in the whole pipeline.
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K

    K.clear_session()

    X_raw = featurize(X_comp)
    sx = MinMaxScaler().fit(X_raw)
    sy = MinMaxScaler().fit(y)

    X_train = sx.transform(X_raw)
    y_train = sy.transform(y)
    sample_weight = None

    if X_gen_comp is not None and len(X_gen_comp) > 0:
        w_gen = best_params.get("w_gen", cfg.w_gen_without_generated)
        X_gen = sx.transform(featurize(X_gen_comp))
        sample_weight = np.concatenate(
            [
                np.ones(len(X_train), dtype=np.float32),
                np.full(len(X_gen), w_gen, dtype=np.float32),
            ]
        )
        X_train = np.vstack([X_train, X_gen])
        y_train = np.vstack([y_train, sy.transform(y_gen)])

    # The model this produces is the one whose test metrics are reported, so it
    # is seeded like the folds above rather than left to whatever state the
    # epoch-selection loop happened to end in.
    tf.keras.utils.set_random_seed(cfg.seed)
    model, batch_size = build_model(X_train.shape[1], best_params, n_targets=y.shape[1])
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    artifact = Path(cfg.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    try:
        model_path = artifact / "final_best_model.h5"
        model.save(model_path)
    except Exception:
        model_path = artifact / "final_best_model.keras"
        model.save(model_path)

    # Saved alongside the model because the generator's consistency filter
    # loads this model and has to feed it inputs scaled the way it was trained.
    # A model without its scalers is not usable by anything but this function.
    import joblib

    joblib.dump(sx, artifact / "sx_final.joblib")
    joblib.dump(sy, artifact / "sy_final.joblib")

    X_test_scaled = sx.transform(featurize(X_test_comp))
    y_pred_scaled = model(X_test_scaled, training=False).numpy()
    y_pred = sy.inverse_transform(y_pred_scaled)

    metrics = {
        "scaled": metrics_full(sy.transform(y_test), y_pred_scaled),
        "physical": metrics_full(y_test, y_pred),
    }
    with open(artifact / "metrics_test.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=_json_default)

    out = pd.DataFrame(y_test, columns=[f"y_true_{t}" for t in TARGET_COLS])
    for i, name in enumerate(TARGET_COLS):
        out[f"y_pred_{name}"] = y_pred[:, i]
    out.to_csv(artifact / "test_predictions.csv", index=False, encoding="utf-8-sig")

    if verbose:
        print(f"  saved {model_path}")
        print(json.dumps(metrics["physical"], indent=2))

    return metrics


def tune_scenario(cfg: SurrogateConfig, verbose: bool = True) -> dict[str, Any]:
    """Run stage 2 for one augmentation scenario, end to end."""
    df_train = load_table(cfg.train_csv, "train csv")
    df_test = load_table(cfg.test_csv, "test csv")

    if len(df_train) < cfg.k_folds:
        raise ValueError(
            f"{len(df_train)} training rows cannot be split into "
            f"{cfg.k_folds} folds"
        )

    X_comp = df_train[COMP_COLS].to_numpy(dtype=np.float64)
    y = df_train[TARGET_COLS].to_numpy(dtype=np.float64)
    X_test_comp = df_test[COMP_COLS].to_numpy(dtype=np.float64)
    y_test = df_test[TARGET_COLS].to_numpy(dtype=np.float64)

    X_gen_comp = y_gen = None
    n_generated = 0
    if cfg.generated_csv is not None:
        # Falling back to real-only here would run the control and file it under
        # the augmented scenario's name, which is indistinguishable from the
        # augmented run in every artifact it writes. Fail instead.
        generated_path = Path(cfg.generated_csv)
        if not generated_path.exists():
            raise FileNotFoundError(
                f"generated_csv is set to {cfg.generated_csv} but that file does "
                "not exist; stage 1 has to run before this scenario"
            )
        df_gen = load_table(cfg.generated_csv, "generated csv")
        if len(df_gen) == 0:
            raise ValueError(f"generated_csv {cfg.generated_csv} has no rows")
        X_gen_comp = df_gen[COMP_COLS].to_numpy(dtype=np.float64)
        y_gen = df_gen[TARGET_COLS].to_numpy(dtype=np.float64)
        n_generated = len(df_gen)

    if verbose:
        print(f"[1/3] searching ({cfg.optuna_trials} trials, {cfg.k_folds} folds)")
        print(f"  {len(df_train)} measured rows, {n_generated} generated, "
              f"{len(df_test)} held out")

    best_params, optuna_cv_score = run_search(cfg, X_comp, y, X_gen_comp, y_gen, verbose)

    problems = check_best_params(best_params, cfg)
    if problems and verbose:
        for problem in problems:
            print(f"  warning: {problem}")

    if verbose:
        print("[2/3] choosing the epoch count")
    epoch_info = select_final_epoch(
        best_params, X_comp, y, X_gen_comp, y_gen, cfg, verbose
    )

    if verbose:
        print(f"[3/3] refitting on all data for {epoch_info['best_epoch']} epochs")
    metrics = train_final_model(
        best_params,
        epoch_info["best_epoch"],
        X_comp,
        y,
        X_gen_comp,
        y_gen,
        X_test_comp,
        y_test,
        cfg,
        verbose,
    )

    return {
        "scenario": Path(cfg.artifact_dir).name,
        "generated_rows": n_generated,
        "optuna_cv_score": float(optuna_cv_score),
        "final_epoch_fold_loss": float(epoch_info["val_loss"]),
        "optimal_epoch": int(epoch_info["best_epoch"]),
        "best_w_gen": best_params.get("w_gen", cfg.w_gen_without_generated),
        "test_r2_mean": metrics["physical"]["r2_mean"],
        "test_r2_kxx": metrics["physical"].get("r2_kxx"),
        "test_r2_S_ANE": metrics["physical"].get("r2_S_ANE"),
        "test_mae_kxx": metrics["physical"].get("mae_kxx"),
        "test_mae_S_ANE": metrics["physical"].get("mae_S_ANE"),
        "out_of_range_params": problems,
    }


def scenario_configs(
    cfg: SurrogateConfig, generated_sizes: tuple[int, ...], artifact_root: str = "artifacts"
) -> list[SurrogateConfig]:
    """One configuration per augmentation scenario, plus the real-only branch.

    The real-only scenario is the control the central claim rests on: it is
    what "augmentation did not always win" is measured against, so it is not
    optional and is always included.
    """
    from dataclasses import replace

    scenarios = [
        replace(
            cfg,
            generated_csv=f"{artifact_root}/augmented_data_n{n}.csv",
            artifact_dir=f"{artifact_root}/dnn_gan_n{n}",
        )
        for n in generated_sizes
    ]
    scenarios.append(
        replace(cfg, generated_csv=None, artifact_dir=f"{artifact_root}/dnn_base")
    )
    return scenarios


def run_scenarios(
    configs: list[SurrogateConfig],
    summary_path: str | Path = "artifacts/scenario_summary.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every scenario and rank them by the cross-validation score.

    The ranking column is `optuna_cv_score`: the objective of the winning
    Optuna trial, which is the mean over five folds of each fold's minimum
    validation MSE. This is the criterion the manuscript defines, and it
    decides which branch and which augmentation size are carried into the
    ensemble stage.

    `final_epoch_fold_loss` and `optimal_epoch` are also written out. They come
    from `select_final_epoch`, which takes a minimum over folds to decide how
    long the final refit should run, and they are diagnostics only. Ranking by
    `final_epoch_fold_loss` would select scenarios on the fold that happened to
    fit best.

    The score remains noisy. Scenarios separated by less than the between-fold
    spread should not be treated as ordered -- Supplementary Note S4 quantifies
    the comparable variation produced by the training seed alone.
    """
    from tensorflow.keras import backend as K

    rows = []
    for cfg in configs:
        if verbose:
            print(f"\n=== {cfg.artifact_dir} ===")
        rows.append(tune_scenario(cfg, verbose))
        K.clear_session()
        gc.collect()

    summary = pd.DataFrame(rows).sort_values("optuna_cv_score", ascending=True)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"\nwritten to {summary_path}")
        print(summary.to_string(index=False))
    return summary
