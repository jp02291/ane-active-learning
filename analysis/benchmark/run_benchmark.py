"""Surrogate-model benchmark behind Supplementary Note S2 and Fig. S6.

The manuscript uses a deep neural network as the surrogate. Note S2 asks the
obvious question: would a simpler model have done better on a dataset this
small? Five regressors are compared under one protocol -- the DNN, kernel ridge
regression, support vector regression, extreme gradient boosting, and
Gaussian-process regression.

Everything except the regressor is held fixed. All five see the same
15-dimensional inputs from `ane.features`, the same repeated 3-fold
cross-validation splits, the same Optuna budget of 150 trials, and the same
target-scaled MSE objective. Scalers are fitted inside each training fold only.
The cycle-1 partition of `data/split/cycle1/` provides the training data and
the held-out set, so the benchmark runs on the same split the campaign reported.

What the numbers say, and do not say
------------------------------------
Cross-validation and the held-out set disagree, and the held-out set has nine
samples. Neither is a basis for declaring one regressor better. The comparison
exists to show that the DNN is not an arbitrary choice, not to rank the five.
The DNN was adopted for what it provides structurally: joint prediction of both
targets and a deep ensemble whose spread becomes the disagreement score U.

Two caveats belong with the table. The DNN entries come from a single seeded
model rather than the pruned ensemble, so they carry more run-to-run variation
than the other four. And the held-out compositions were used to select the
partition itself (Algorithm S1 stratifies on the targets), so these errors
describe the bulk of the composition space rather than the high-performance
region.

    python run_benchmark.py
    python run_benchmark.py --only SVR
    python run_benchmark.py --reuse-params results   # skip the search, refit

The DNN search dominates the runtime: 150 trials take hours, while the other
four take minutes each. `--reuse-params` reads the deposited hyperparameters
and re-runs only the evaluation, which is what a reader checking the table
needs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import json
import os
import random
import time
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from ane.elements import ELEMENTS  # noqa: E402
from ane.features import FEATURE_NAMES, featurize  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRAIN_FILE = REPO / "data" / "split" / "cycle1" / "train.csv"
TEST_FILE = REPO / "data" / "split" / "cycle1" / "test.csv"
OUT = HERE / "results"

MODELS = ["DNN", "KRR", "SVR", "XGB", "GPR"]

#: repeated cross-validation, matching the surrogate protocol of Algorithm S3
K_FOLDS = 3
N_REPEATS = 3

#: the same Optuna budget for every model, so none is favored by search effort
N_TRIALS = 150

EPOCHS_TUNE = 100
EPOCHS_EVAL = 200
EARLY_STOPPING_PATIENCE_TUNE = 30
EARLY_STOPPING_PATIENCE_EVAL = 50
REDUCE_LR_PATIENCE = 10

#: One seeded DNN, not the pruned ensemble. The ensemble is what the campaign
#: uses for candidate ranking; here the point is to compare model classes under
#: one protocol, and averaging only the DNN over seeds would break that.
DNN_EVAL_SEEDS = [SEED]

COMP_COLS = list(ELEMENTS)
TARGET_COLS = ["kxx", "S_ANE"]

#: the eight composition-derived descriptors, named as `ane.features` emits them
CALC_COLS = list(FEATURE_NAMES[7:])


# ============================================================
# shared utilities
# ============================================================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"unsupported type for JSON serialization: {type(obj)}")


def load_table(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path.resolve()}")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_required_columns(df: pd.DataFrame, required: List[str], tag: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"[{tag}] missing required columns: {missing}")


def assert_finite(*arrays: np.ndarray, context: str = "") -> None:
    for arr in arrays:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"[{context}] contains NaN or inf")


def warn_composition_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    decimals: int = 8,
) -> None:
    train_keys = {
        tuple(row) for row in train_df[COMP_COLS].round(decimals).to_numpy()
    }
    test_keys = {
        tuple(row) for row in test_df[COMP_COLS].round(decimals).to_numpy()
    }
    overlap = train_keys.intersection(test_keys)
    if overlap:
        warnings.warn(
            f"{len(overlap)} composition(s) appear in both the training and the "
            "held-out set. Check the partition before reading the errors."
        )


# ============================================================
# model construction
# ============================================================
def build_dnn(
    input_dim: int,
    num_layers: int,
    num_neurons: int,
    learning_rate: float,
    l2: float,
    dropout: float,
) -> tf.keras.Model:
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

    model.add(Dense(len(TARGET_COLS), activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def suggest_params(
    trial: optuna.Trial,
    model_name: str,
) -> Dict[str, Any]:
    if model_name == "DNN":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 8),
            "num_neurons": trial.suggest_int(
                "num_neurons", 16, 128, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "batch_size": trial.suggest_categorical(
                "batch_size", [8, 16, 32]
            ),
        }

    if model_name == "KRR":
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        params = {
            "kernel": kernel,
            "alpha": trial.suggest_float("alpha", 1e-6, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1e2, log=True),
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
            params["coef0"] = trial.suggest_float(
                "coef0", 1e-3, 10.0, log=True
            )
        return params

    if model_name == "SVR":
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "epsilon": trial.suggest_float(
                "epsilon", 1e-4, 0.5, log=True
            ),
            "gamma": trial.suggest_float(
                "gamma", 1e-5, 1e2, log=True
            ),
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
            params["coef0"] = trial.suggest_float(
                "coef0", 1e-3, 10.0, log=True
            )
        return params

    if model_name == "XGB":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 800
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-2, 20.0, log=True
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-6, 100.0, log=True
            ),
        }

    if model_name == "GPR":
        return {
            "kernel_type": trial.suggest_categorical(
                "kernel_type", ["RBF", "Matern32", "Matern52"]
            ),
            "constant_value": trial.suggest_float(
                "constant_value", 1e-2, 1e2, log=True
            ),
            "length_scale": trial.suggest_float(
                "length_scale", 1e-2, 1e2, log=True
            ),
            "noise_level": trial.suggest_float(
                "noise_level", 1e-8, 1e0, log=True
            ),
            "alpha": trial.suggest_float(
                "alpha", 1e-10, 1e-4, log=True
            ),
        }

    raise ValueError(f"unsupported model: {model_name}")


def build_sklearn_model(
    model_name: str,
    params: Dict[str, Any],
    seed: int,
):
    if model_name == "KRR":
        base_model = KernelRidge(
            kernel=params["kernel"],
            alpha=params["alpha"],
            gamma=params["gamma"],
            degree=params.get("degree", 3),
            coef0=params.get("coef0", 1.0),
        )
        return MultiOutputRegressor(base_model, n_jobs=1)

    if model_name == "SVR":
        base_model = SVR(
            kernel=params["kernel"],
            C=params["C"],
            epsilon=params["epsilon"],
            gamma=params["gamma"],
            degree=params.get("degree", 3),
            coef0=params.get("coef0", 0.0),
        )
        return MultiOutputRegressor(base_model, n_jobs=1)

    if model_name == "XGB":
        base_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        return MultiOutputRegressor(base_model, n_jobs=1)

    if model_name == "GPR":
        constant_kernel = ConstantKernel(
            constant_value=params["constant_value"],
            constant_value_bounds="fixed",
        )

        if params["kernel_type"] == "RBF":
            core_kernel = RBF(
                length_scale=params["length_scale"],
                length_scale_bounds="fixed",
            )
        elif params["kernel_type"] == "Matern32":
            core_kernel = Matern(
                length_scale=params["length_scale"],
                length_scale_bounds="fixed",
                nu=1.5,
            )
        else:
            core_kernel = Matern(
                length_scale=params["length_scale"],
                length_scale_bounds="fixed",
                nu=2.5,
            )

        noise_kernel = WhiteKernel(
            noise_level=params["noise_level"],
            noise_level_bounds="fixed",
        )
        kernel = constant_kernel * core_kernel + noise_kernel

        base_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=params["alpha"],
            normalize_y=False,
            optimizer=None,
            random_state=seed,
        )
        return MultiOutputRegressor(base_model, n_jobs=1)

    raise ValueError(f"cannot build sklearn model: {model_name}")


# ============================================================
# fold preprocessing, shared by every model
# ============================================================
def prepare_fold(
    train_comp: np.ndarray,
    train_y: np.ndarray,
    valid_comp: np.ndarray,
    valid_y: np.ndarray,
):
    train_x_raw = featurize(train_comp)
    valid_x_raw = featurize(valid_comp)

    x_scaler = MinMaxScaler().fit(train_x_raw)
    y_scaler = MinMaxScaler().fit(train_y)

    train_x = x_scaler.transform(train_x_raw).astype(np.float32)
    valid_x = x_scaler.transform(valid_x_raw).astype(np.float32)
    train_y_scaled = y_scaler.transform(train_y).astype(np.float32)
    valid_y_scaled = y_scaler.transform(valid_y).astype(np.float32)

    assert_finite(
        train_x,
        valid_x,
        train_y_scaled,
        valid_y_scaled,
        context="prepare_fold",
    )

    return (
        train_x,
        train_y_scaled,
        valid_x,
        valid_y_scaled,
        x_scaler,
        y_scaler,
    )


# ============================================================
# DNN fitting
# ============================================================
def fit_predict_dnn_fold(
    params: Dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
) -> Tuple[np.ndarray, int]:
    set_all_seeds(seed)
    K.clear_session()

    model_params = {
        key: value
        for key, value in params.items()
        if key != "batch_size"
    }

    model = build_dnn(
        input_dim=train_x.shape[1],
        **model_params,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-6,
        verbose=0,
    )

    history = model.fit(
        train_x,
        train_y,
        validation_data=(valid_x, valid_y),
        epochs=epochs,
        batch_size=int(params["batch_size"]),
        verbose=0,
        callbacks=[early_stopping, reduce_lr],
    )

    predictions = model.predict(valid_x, verbose=0)
    val_losses = np.asarray(history.history["val_loss"], dtype=float)
    best_epoch = int(np.nanargmin(val_losses) + 1)

    del model, history
    K.clear_session()
    gc.collect()

    return predictions, best_epoch


def fit_full_dnn(
    params: Dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    seeds: List[int],
) -> np.ndarray:
    predictions = []

    model_params = {
        key: value
        for key, value in params.items()
        if key != "batch_size"
    }

    for seed in seeds:
        set_all_seeds(seed)
        K.clear_session()

        model = build_dnn(
            input_dim=train_x.shape[1],
            **model_params,
        )

        model.fit(
            train_x,
            train_y,
            epochs=max(1, int(epochs)),
            batch_size=int(params["batch_size"]),
            verbose=0,
        )

        predictions.append(model.predict(test_x, verbose=0))

        del model
        K.clear_session()
        gc.collect()

    return np.mean(np.stack(predictions, axis=0), axis=0)


# ============================================================
# metrics
# ============================================================
def calculate_metric_rows(
    model_name: str,
    split_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []

    for target_idx, target_name in enumerate(TARGET_COLS):
        target_true = y_true[:, target_idx]
        target_pred = y_pred[:, target_idx]

        mse = mean_squared_error(target_true, target_pred)

        rows.append(
            {
                "model": model_name,
                "split": split_id,
                "target": target_name,
                "MAE": mean_absolute_error(target_true, target_pred),
                "RMSE": math.sqrt(mse),
                "R2": r2_score(target_true, target_pred),
            }
        )

    return rows


# ============================================================
# HPO objective
# ============================================================
def make_objective(
    model_name: str,
    train_comp: np.ndarray,
    train_y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        fold_losses = []

        try:
            for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
                (
                    fold_train_x,
                    fold_train_y,
                    fold_valid_x,
                    fold_valid_y,
                    _,
                    _,
                ) = prepare_fold(
                    train_comp[train_idx],
                    train_y[train_idx],
                    train_comp[valid_idx],
                    train_y[valid_idx],
                )

                fold_seed = SEED + 1000 * trial.number + fold_idx

                if model_name == "DNN":
                    pred_scaled, _ = fit_predict_dnn_fold(
                        params=params,
                        train_x=fold_train_x,
                        train_y=fold_train_y,
                        valid_x=fold_valid_x,
                        valid_y=fold_valid_y,
                        seed=fold_seed,
                        epochs=EPOCHS_TUNE,
                        patience=EARLY_STOPPING_PATIENCE_TUNE,
                    )
                else:
                    model = build_sklearn_model(
                        model_name,
                        params,
                        seed=fold_seed,
                    )
                    model.fit(fold_train_x, fold_train_y)
                    pred_scaled = model.predict(fold_valid_x)

                    del model
                    gc.collect()

                loss = mean_squared_error(
                    fold_valid_y,
                    pred_scaled,
                    multioutput="uniform_average",
                )

                if not np.isfinite(loss):
                    return float("inf")

                fold_losses.append(float(loss))

            if not fold_losses:
                return float("inf")

            return float(np.mean(fold_losses))

        except Exception as exc:
            trial.set_user_attr("exception", repr(exc))
            return float("inf")

    return objective

def print_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    completed_trials = trial.number + 1

    if completed_trials % 10 == 0 or completed_trials == N_TRIALS:
        print(
            f"  Trial {completed_trials:>3}/{N_TRIALS} | "
            f"Current value: {trial.value:.6f} | "
            f"Best value: {study.best_value:.6f}",
            flush=True,
        )

def optimize_model(
    model_name: str,
    train_comp: np.ndarray,
    train_y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> optuna.Study:
    print(f"\n{'=' * 72}")
    print(f"[HPO] {model_name}")
    print(f"{'=' * 72}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name=f"{model_name}_comparison",
    )

    study.optimize(
        make_objective(
            model_name=model_name,
            train_comp=train_comp,
            train_y=train_y,
            cv_splits=cv_splits,
        ),
        n_trials=N_TRIALS,
        gc_after_trial=True,
        show_progress_bar=False,
        callbacks=[print_callback],
    )

    print(f"[HPO] {model_name} best scaled MSE = {study.best_value:.6f}")
    print(json.dumps(study.best_params, indent=2, ensure_ascii=False))

    return study


# ============================================================
# repeated-CV evaluation of the selected hyperparameters
# ============================================================
def evaluate_best_params_cv(
    model_name: str,
    best_params: Dict[str, Any],
    train_comp: np.ndarray,
    train_y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    metric_rows = []
    best_epochs = []

    for split_idx, (train_idx, valid_idx) in enumerate(cv_splits):
        (
            fold_train_x,
            fold_train_y,
            fold_valid_x,
            fold_valid_y,
            _,
            y_scaler,
        ) = prepare_fold(
            train_comp[train_idx],
            train_y[train_idx],
            train_comp[valid_idx],
            train_y[valid_idx],
        )

        fold_seed = SEED + 10000 + split_idx

        if model_name == "DNN":
            seed_predictions = []
            seed_epochs = []

            for dnn_seed in DNN_EVAL_SEEDS:
                pred_scaled, best_epoch = fit_predict_dnn_fold(
                    params=best_params,
                    train_x=fold_train_x,
                    train_y=fold_train_y,
                    valid_x=fold_valid_x,
                    valid_y=fold_valid_y,
                    seed=fold_seed + dnn_seed,
                    epochs=EPOCHS_EVAL,
                    patience=EARLY_STOPPING_PATIENCE_EVAL,
                )
                seed_predictions.append(pred_scaled)
                seed_epochs.append(best_epoch)

            pred_scaled = np.mean(
                np.stack(seed_predictions, axis=0),
                axis=0,
            )
            best_epochs.extend(seed_epochs)

        else:
            model = build_sklearn_model(
                model_name,
                best_params,
                seed=fold_seed,
            )
            model.fit(fold_train_x, fold_train_y)
            pred_scaled = model.predict(fold_valid_x)

            del model
            gc.collect()

        pred_physical = y_scaler.inverse_transform(pred_scaled)
        true_physical = train_y[valid_idx]

        metric_rows.extend(
            calculate_metric_rows(
                model_name=model_name,
                split_id=f"cv_{split_idx + 1}",
                y_true=true_physical,
                y_pred=pred_physical,
            )
        )

    return metric_rows, best_epochs


# ============================================================
# fit on the full training split, evaluate on the held-out set
# ============================================================
def fit_and_evaluate_test(
    model_name: str,
    best_params: Dict[str, Any],
    train_comp: np.ndarray,
    train_y: np.ndarray,
    test_comp: np.ndarray,
    test_y: np.ndarray,
    dnn_best_epochs: List[int],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    train_x_raw = featurize(train_comp)
    test_x_raw = featurize(test_comp)

    x_scaler = MinMaxScaler().fit(train_x_raw)
    y_scaler = MinMaxScaler().fit(train_y)

    train_x = x_scaler.transform(train_x_raw).astype(np.float32)
    test_x = x_scaler.transform(test_x_raw).astype(np.float32)
    train_y_scaled = y_scaler.transform(train_y).astype(np.float32)

    if model_name == "DNN":
        if dnn_best_epochs:
            final_epochs = int(np.median(dnn_best_epochs))
        else:
            final_epochs = EPOCHS_EVAL

        print(f"[final DNN] median best epoch over the CV folds = {final_epochs}")

        pred_scaled = fit_full_dnn(
            params=best_params,
            train_x=train_x,
            train_y=train_y_scaled,
            test_x=test_x,
            epochs=final_epochs,
            seeds=DNN_EVAL_SEEDS,
        )
    else:
        model = build_sklearn_model(
            model_name,
            best_params,
            seed=SEED,
        )
        model.fit(train_x, train_y_scaled)
        pred_scaled = model.predict(test_x)

        del model
        gc.collect()

    pred_physical = y_scaler.inverse_transform(pred_scaled)

    metric_rows = calculate_metric_rows(
        model_name=model_name,
        split_id="held_out_test",
        y_true=test_y,
        y_pred=pred_physical,
    )

    return metric_rows, pred_physical


# ============================================================
# summary and figure
# ============================================================
def summarize_cv_metrics(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cv_metrics
        .groupby(["model", "target"], as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            n_splits=("split", "count"),
        )
    )

    return summary


def plot_cv_comparison(
    cv_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    model_order = [
        model for model in MODELS
        if model in cv_summary["model"].unique()
    ]
    x_positions = np.arange(len(model_order))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panel_specs = [
        ("kxx", "MAE", "Thermal conductivity MAE"),
        ("S_ANE", "MAE", r"$|S_{\mathrm{ANE}}|$ MAE"),
        ("kxx", "R2", r"Thermal conductivity $R^2$"),
        ("S_ANE", "R2", r"$|S_{\mathrm{ANE}}|$ $R^2$"),
    ]

    for axis, (target, metric, title) in zip(
        axes.flat,
        panel_specs,
    ):
        subset = (
            cv_summary[cv_summary["target"] == target]
            .set_index("model")
            .reindex(model_order)
        )

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        axis.errorbar(
            x_positions,
            subset[mean_col].to_numpy(),
            yerr=subset[std_col].fillna(0.0).to_numpy(),
            fmt="D",
            capsize=4,
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(model_order)
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.4)

        if metric == "R2":
            axis.axhline(0.0, linewidth=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_protocol() -> None:
    protocol = {
        "seed": SEED,
        "models": MODELS,
        "features": {
            "composition_columns": COMP_COLS,
            "ILR_dimensions": 7,
            "calculated_descriptor_columns": CALC_COLS,
            "total_input_dimensions": 15,
            "zero_replacement_delta": 1e-3,
        },
        "targets": TARGET_COLS,
        "cross_validation": {
            "type": "RepeatedKFold",
            "n_splits": K_FOLDS,
            "n_repeats": N_REPEATS,
            "same_splits_for_all_models": True,
        },
        "hyperparameter_optimization": {
            "library": "Optuna",
            "sampler": "TPE",
            "n_trials_per_model": N_TRIALS,
            "objective": "mean scaled validation MSE across both targets",
        },
        "scaling": {
            "input": "MinMaxScaler fitted on fold train only",
            "target": "MinMaxScaler fitted on fold train only",
        },
        "dnn": {
            "evaluation_seeds": DNN_EVAL_SEEDS,
            "epochs_tune": EPOCHS_TUNE,
            "epochs_eval": EPOCHS_EVAL,
        },
    }

    with open(
        OUT / "comparison_protocol.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            protocol,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


# ============================================================
# Main
# ============================================================
def main(models: List[str], reuse_from: "Path | None") -> None:
    t_start = time.time()
    train_df_raw = load_table(TRAIN_FILE)
    test_df_raw = load_table(TEST_FILE)

    required_columns = COMP_COLS + TARGET_COLS
    ensure_required_columns(train_df_raw, required_columns, "train")
    ensure_required_columns(test_df_raw, required_columns, "test")

    train_df = (
        train_df_raw[required_columns]
        .copy()
        .dropna()
        .reset_index(drop=True)
    )
    test_df = (
        test_df_raw[required_columns]
        .copy()
        .dropna()
        .reset_index(drop=True)
    )

    if len(train_df) < K_FOLDS:
        raise ValueError(
            f"training samples ({len(train_df)}) fewer than K_FOLDS ({K_FOLDS})"
        )
    if len(test_df) < 2:
        raise ValueError("fewer than two held-out samples; R2 is undefined")

    warn_composition_overlap(train_df, test_df)

    train_comp = train_df[COMP_COLS].to_numpy(dtype=np.float32)
    train_y = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
    test_comp = test_df[COMP_COLS].to_numpy(dtype=np.float32)
    test_y = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

    assert_finite(
        train_comp,
        train_y,
        test_comp,
        test_y,
        context="input data",
    )

    print(f"[data] training samples     = {len(train_df)}")
    print(f"[data] held-out samples      = {len(test_df)}")
    print("[Features] ILR 7 + calculated descriptors 8 = 15")

    repeated_kfold = RepeatedKFold(
        n_splits=K_FOLDS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )
    cv_splits = list(repeated_kfold.split(train_comp))

    all_best_params = {}
    all_cv_metric_rows = []
    all_test_metric_rows = []
    test_predictions = pd.DataFrame(
        {
            "sample_index": np.arange(len(test_df)),
            **{
                f"true_{target}": test_y[:, idx]
                for idx, target in enumerate(TARGET_COLS)
            },
        }
    )

    for model_name in models:
        started = time.time()
        deposited = reuse_from / f"best_params_{model_name}.json" if reuse_from else None
        if deposited is not None and deposited.exists():
            record = json.loads(deposited.read_text(encoding="utf-8"))
            best_params = record["best_params"]
            all_best_params[model_name] = {
                "best_cv_scaled_mse": record.get("best_cv_scaled_mse"),
                "best_params": best_params,
            }
            print(f"\n[reuse] {model_name}: hyperparameters read from {deposited.name}")
        else:
            study = optimize_model(
                model_name=model_name,
                train_comp=train_comp,
                train_y=train_y,
                cv_splits=cv_splits,
            )
            best_params = study.best_params
            all_best_params[model_name] = {
                "best_cv_scaled_mse": study.best_value,
                "best_params": best_params,
            }
        print(f"[time] {model_name} search: {time.time() - started:.1f} s", flush=True)

        with open(
            OUT / f"best_params_{model_name}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                all_best_params[model_name],
                file,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )

        cv_metric_rows, dnn_best_epochs = evaluate_best_params_cv(
            model_name=model_name,
            best_params=best_params,
            train_comp=train_comp,
            train_y=train_y,
            cv_splits=cv_splits,
        )
        all_cv_metric_rows.extend(cv_metric_rows)

        test_metric_rows, test_pred = fit_and_evaluate_test(
            model_name=model_name,
            best_params=best_params,
            train_comp=train_comp,
            train_y=train_y,
            test_comp=test_comp,
            test_y=test_y,
            dnn_best_epochs=dnn_best_epochs,
        )
        all_test_metric_rows.extend(test_metric_rows)

        for target_idx, target_name in enumerate(TARGET_COLS):
            test_predictions[
                f"pred_{target_name}_{model_name}"
            ] = test_pred[:, target_idx]

    cv_metrics_df = pd.DataFrame(all_cv_metric_rows)
    cv_summary_df = summarize_cv_metrics(cv_metrics_df)
    test_metrics_df = pd.DataFrame(all_test_metric_rows)

    cv_metrics_df.to_csv(
        OUT / "cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary_df.to_csv(
        OUT / "cv_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_metrics_df.to_csv(
        OUT / "held_out_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_predictions.to_csv(
        OUT / "held_out_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        OUT / "all_best_params.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_best_params,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )

    plot_cv_comparison(
        cv_summary=cv_summary_df,
        output_path=OUT / "model_comparison_cv.png",
    )
    save_protocol()

    print(f"\n[time] total: {time.time() - t_start:.1f} s")
    print(f"wrote {OUT.relative_to(REPO)}")
    print("\n[held-out test metrics]")
    print(
        test_metrics_df
        .sort_values(["target", "MAE"])
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=MODELS, help="benchmark a single model")
    ap.add_argument(
        "--reuse-params",
        type=Path,
        metavar="DIR",
        help="read best_params_<MODEL>.json from DIR instead of searching; "
        "models without a file there are searched as usual",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main([args.only] if args.only else MODELS, args.reuse_params)
