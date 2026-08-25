"""Generative augmentation: WGAN-GP with filtering (Algorithm S2).

The generator produces a composition and its two properties jointly, in a
nine-dimensional space of seven isometric log-ratio coordinates plus kappa and
S_ANE. Generating them together is the point: a composition sampler would
still need labels from somewhere, and labelling with the surrogate would only
reproduce what the surrogate already believes.

Nothing generated is trusted. A sample survives only if it passes, in order:

1. the physical constraints -- stoichiometry, the Mn bound, dH_mix <= 0;
2. a range check against the measured properties, allowing modest
   extrapolation but not invention;
3. a consistency check against a surrogate trained on measured data alone,
   which rejects samples whose stated properties the surrogate disagrees with
   by more than a tolerance derived from its own training residuals.

Filter three is what makes the augmented data more than noise, and it is also
where the argument is most delicate: the threshold is derived from held-out
residuals when a validation file is available, and otherwise on a fifth of the
training data with a safety factor, which is an in-sample estimate. The safety
factor is there because of that. Section 2.3 and Algorithm S2 should say so.

Training is not run to a fixed epoch count. Every `eval_interval` epochs a
cheap proxy model is cross-validated with and without the current generator's
output, and the generator is checkpointed only when adding synthetic data
lowers the error *and* a label-shuffled control does worse than the real
labels. The second condition matters: without it a generator that merely adds
plausible noise in the right region scores as an improvement.

Two things about this module that look like duplication and are not.

`ilr_transform` here is not `ane.features.ILR`. This one clips zeros at 1e-12
and is exactly invertible, because the generator has to map back from ILR
coordinates to a composition. The surrogate's transform replaces zeros with
1e-3, which is a better-conditioned input for a network and is not invertible.
They are different transforms for different jobs, and the generator's samples
are fed to the surrogate through `ane.features.featurize`, never through this.

The physical constraints are re-read from `GANConfig` rather than shared with
`SelectionConfig`. What may be generated and what may be proposed for
synthesis are separate decisions that happen to coincide.

`tests/test_augment_parity.py` checks the deterministic parts against the
original notebook.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import GANConfig
from .data import PROPERTY_COLUMNS
from .elements import ELEMENTS, GROUP_A, GROUP_B
from .features import featurize
from .physics import delta_h_mix

__all__ = [
    "ilr_transform",
    "ilr_inverse",
    "quantize_on_grid",
    "composition_keys",
    "physical_mask",
    "property_mask",
    "kcenter_select",
    "rare_elements",
    "sample_probabilities",
    "surrogate_threshold",
    "surrogate_consistency_mask",
    "build_generator",
    "build_critic",
    "proxy_cv_delta",
    "train_gan",
]

COMP_COLS: list[str] = list(ELEMENTS)
TARGET_COLS: list[str] = list(PROPERTY_COLUMNS)

#: ILR coordinates plus the two properties: what the generator emits.
N_ILR = len(ELEMENTS) - 1
INPUT_DIM = N_ILR + len(PROPERTY_COLUMNS)


# ---------------------------------------------------------------------------
# invertible ILR, for generation only
# ---------------------------------------------------------------------------


def _helmert(D: int) -> np.ndarray:
    """(D-1, D) Helmert basis, the transpose of the one in `ane.features`."""
    H = np.zeros((D - 1, D), dtype=np.float64)
    for i in range(1, D):
        scale = np.sqrt(i * (i + 1))
        H[i - 1, :i] = 1.0 / scale
        H[i - 1, i] = -i / scale
    return H


def ilr_transform(comp: np.ndarray) -> np.ndarray:
    """Compositions to ILR coordinates, invertibly.

    Zeros are clipped at 1e-12 rather than replaced, which keeps the map exact
    but sends absent components to a log-ratio around -27. That is deliberate:
    the generator learns a distribution in this space and must be able to come
    back from it, and a component the generator drives to the floor should come
    back as absent. `ane.features.ILR` makes the opposite trade for the same
    reason in reverse -- it never has to invert.
    """
    comp = np.asarray(comp, dtype=np.float64)
    comp = np.clip(comp, 1e-12, None)
    comp = comp / comp.sum(axis=1, keepdims=True)
    return np.log(comp) @ _helmert(comp.shape[1]).T


def ilr_inverse(ilr_comp: np.ndarray) -> np.ndarray:
    """ILR coordinates back to compositions summing to one."""
    ilr_comp = np.asarray(ilr_comp, dtype=np.float64)
    D = ilr_comp.shape[1] + 1
    X = np.exp(ilr_comp @ _helmert(D))
    X = np.maximum(X, 1e-12)
    return X / X.sum(axis=1, keepdims=True)


def quantize_on_grid(comp: np.ndarray, step: float) -> np.ndarray:
    """Snap compositions to a grid while keeping them summing to one.

    Rounding each component independently would not sum to one. This takes the
    floor of each, then distributes the remaining units to the components with
    the largest residuals -- the largest-remainder method -- so the result is
    on the grid exactly and is the closest such point in the ordering that
    matters.
    """
    X = np.asarray(comp, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    n, d = X.shape
    resolution = int(round(1.0 / float(step)))

    out = np.zeros((n, d), dtype=int)
    for i in range(n):
        row = np.maximum(0.0, X[i].copy())
        row = row / (row.sum() + 1e-12)
        raw = row * resolution
        base = np.floor(raw).astype(int)
        residual = raw - base
        need = resolution - int(base.sum())
        if need > 0:
            order = np.argsort(-residual)
            base[order[:need]] += 1
        elif need < 0:
            order = np.argsort(residual)
            for idx in order[: (-need)]:
                if base[idx] > 0:
                    base[idx] -= 1
        out[i] = base

    return out.astype(np.float64) / float(resolution)


def composition_keys(comp: np.ndarray, step: float) -> set[tuple[int, ...]]:
    """Integer keys identifying compositions on the grid.

    Used to keep the generator from re-proposing something already measured.
    Integers rather than floats because two compositions that should be the
    same point can differ in the last bit after arithmetic.
    """
    quantized = quantize_on_grid(comp, step=step)
    resolution = int(round(1.0 / step))
    keys = (quantized * resolution + 1e-9).astype(int)
    return {tuple(row) for row in keys}


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def physical_mask(ilr7: np.ndarray, cfg: GANConfig) -> tuple[np.ndarray, np.ndarray]:
    """Constraints on generated compositions. Returns (mask, compositions)."""
    comp = ilr_inverse(ilr7)
    df = pd.DataFrame(comp, columns=COMP_COLS)

    sum_a = df[list(GROUP_A)].sum(axis=1).to_numpy()
    sum_b = df[list(GROUP_B)].sum(axis=1).to_numpy()
    ratio = sum_a / np.maximum(sum_b, 1e-12)

    mask = (
        (sum_b > 1e-12)
        & (ratio >= cfg.stoichiometry_min)
        & (ratio <= cfg.stoichiometry_max)
        & (df["Mn"].to_numpy() <= cfg.mn_max)
    )
    mask &= delta_h_mix(comp) <= cfg.h_mix_max
    return mask, comp


def property_mask(
    props: np.ndarray, measured_props: np.ndarray, cfg: GANConfig
) -> np.ndarray:
    """Generated properties must be finite, positive in kappa, and in range.

    The range is the 1st to 99th percentile of the measured values widened by
    `property_margin` of that span. Some extrapolation is wanted -- the search
    is looking for something better than anything measured -- but a generator
    is free to emit a kappa of 0.01 if nothing stops it, and the selection
    stage maximizes 1/kappa.
    """
    kappa, s_ane = props[:, 0], props[:, 1]

    lo = np.quantile(measured_props, cfg.property_quantile_low, axis=0)
    hi = np.quantile(measured_props, cfg.property_quantile_high, axis=0)
    width = hi - lo
    margin = cfg.property_margin

    within = (
        (kappa >= lo[0] - margin * width[0])
        & (kappa <= hi[0] + margin * width[0])
        & (s_ane >= lo[1] - margin * width[1])
        & (s_ane <= hi[1] + margin * width[1])
    )
    # The target is a magnitude, so a negative value is not a property that any
    # composition can have. The percentile window above does not enforce this on
    # its own: widening the measured 1-99% span by `property_margin` puts its
    # lower edge below zero, so without this term the filter admits negative
    # |S_ANE|. Added after the campaign; whether any such sample survived into a
    # historical training set cannot be checked, because the generated CSVs of
    # the campaign were not retained.
    physical = s_ane >= 0.0
    return (
        np.isfinite(kappa) & np.isfinite(s_ane) & (kappa > 0.0) & physical & within
    )


def surrogate_threshold(
    comp_train: np.ndarray,
    prop_train: np.ndarray,
    model,
    x_scaler,
    y_scaler,
    cfg: GANConfig,
    verbose: bool = True,
) -> list[float]:
    """Per-target error a generated sample is allowed to differ by.

    With a validation file, this is a quantile of genuine held-out residuals
    and needs no adjustment. Without one it is a quantile of residuals on a
    random fifth of the *training* data, which the surrogate has already seen,
    so the estimate is optimistic and is multiplied by `safety_factor` to
    compensate. That factor is a judgement call, not a derivation, and the
    manuscript should say the threshold was calibrated this way.
    """
    validation_csv = cfg.surrogate_validation_csv
    if validation_csv and Path(validation_csv).exists():
        df = pd.read_csv(validation_csv)
        X_raw = df[COMP_COLS].to_numpy(dtype=np.float64)
        y_raw = df[TARGET_COLS].to_numpy(dtype=np.float64)
        multiplier = 1.0
        source = f"held-out residuals from {validation_csv}"
    else:
        rng = np.random.default_rng(cfg.seed)
        n = int(len(comp_train) * cfg.surrogate_calibration_fraction)
        idx = rng.choice(len(comp_train), size=n, replace=False)
        X_raw, y_raw = comp_train[idx], prop_train[idx]
        multiplier = cfg.surrogate_safety_factor
        source = (
            f"in-sample residuals on {n} training rows "
            f"x safety factor {multiplier}"
        )

    X = x_scaler.transform(featurize(X_raw))
    if X.shape[1] != getattr(x_scaler, "n_features_in_", X.shape[1]):
        raise ValueError("surrogate feature width does not match its scaler")

    y_pred = y_scaler.inverse_transform(model(X, training=False).numpy())
    diff = np.abs(y_raw - y_pred)

    thresholds = [
        float(np.quantile(diff[:, i], cfg.surrogate_quantile)) * multiplier
        for i in range(diff.shape[1])
    ]
    if verbose:
        print(f"  surrogate tolerance from {source}")
        print(f"    kappa {thresholds[0]:.4f}, S_ANE {thresholds[1]:.4f}")
    return thresholds


def surrogate_consistency_mask(
    comp: np.ndarray,
    props: np.ndarray,
    model,
    x_scaler,
    y_scaler,
    thresholds: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep samples the surrogate agrees with. Returns (mask, predicted, diff).

    Note which way round this runs. The surrogate is not relabelling anything;
    the generated properties stand. It only vetoes samples where the generator
    has claimed a composition-property pair that a model fit to measured data
    finds implausible.
    """
    if model is None or thresholds is None:
        nan = np.full_like(props, np.nan)
        return np.ones(len(comp), dtype=bool), nan, nan

    X = x_scaler.transform(featurize(comp))
    y_pred = y_scaler.inverse_transform(model(X, training=False).numpy())
    diff = np.abs(props - y_pred)
    mask = (diff[:, 0] <= thresholds[0]) & (diff[:, 1] <= thresholds[1])
    return mask, y_pred, diff


# ---------------------------------------------------------------------------
# diversity and sampling
# ---------------------------------------------------------------------------


def kcenter_select(X: np.ndarray, n_select: int) -> np.ndarray:
    """Greedy k-center: repeatedly take the point furthest from those chosen.

    Chosen over random subsampling because the generator's output is denser
    where the real data is dense, and a random subset of it would be too. The
    value of synthetic data here is coverage, so the selection maximizes it
    directly. Starts from the point furthest from the centroid, so the result
    is deterministic.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n_select >= n:
        return np.arange(n, dtype=int)

    centre = X.mean(axis=0)
    first = int(np.argmax(np.sum((X - centre) ** 2, axis=1)))
    selected = [first]

    min_d2 = np.sum((X - X[first]) ** 2, axis=1)
    for _ in range(1, n_select):
        candidate = int(np.argmax(min_d2))
        selected.append(candidate)
        min_d2 = np.minimum(min_d2, np.sum((X - X[candidate]) ** 2, axis=1))

    return np.array(selected, dtype=int)


def rare_elements(
    comp_train: np.ndarray, cfg: GANConfig
) -> tuple[list[str], dict[str, int]]:
    """Elements present in few enough training rows to need oversampling.

    Bounded below as well as above: an element in a single row cannot be
    learned from and boosting it only amplifies that row.
    """
    n = comp_train.shape[0]
    counts = {
        el: int(np.sum(comp_train[:, j] > 1e-12)) for j, el in enumerate(COMP_COLS)
    }
    ceiling = int(np.floor(cfg.rare_max_ratio * n))
    rare = [el for el, c in counts.items() if cfg.rare_min_count <= c <= ceiling]
    return rare, counts


def sample_probabilities(
    comp_train: np.ndarray, rare: list[str], counts: dict[str, int], cfg: GANConfig
) -> np.ndarray:
    """Draw probabilities over the real rows, boosted for rare elements.

    The boost is sqrt(N / count), capped: full inverse-frequency weighting
    would let two rows dominate a batch of sixteen.
    """
    n = comp_train.shape[0]
    weights = np.ones(n, dtype=np.float64)
    for el in rare:
        j = COMP_COLS.index(el)
        boost = min(cfg.weight_boost_max, np.sqrt(n / max(counts[el], 1)))
        weights[comp_train[:, j] > 1e-12] *= boost
    weights = np.minimum(weights, cfg.weight_cap)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


#: Hidden width of both networks. Not a configuration field: the saved
#: checkpoint has this architecture, so changing it would make the released
#: weights unloadable rather than merely producing a different model.
HIDDEN_UNITS = 64


def build_generator(cfg: GANConfig):
    """latent -> 64 -> 64 -> 9, matching the released checkpoint."""
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.Input(shape=(cfg.latent_dim,)),
            keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
            keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
            keras.layers.Dense(INPUT_DIM, activation="linear"),
        ],
        name="generator",
    )


def build_critic(cfg: GANConfig):
    """No batch normalization, by requirement.

    The gradient penalty is imposed per sample, and batch normalization makes
    the critic's output depend on the rest of the batch, so the penalty would
    no longer constrain what it is meant to.
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.Input(shape=(INPUT_DIM,)),
            keras.layers.Dense(HIDDEN_UNITS),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Dense(HIDDEN_UNITS),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Dense(1),
        ],
        name="critic",
    )


# ---------------------------------------------------------------------------
# proxy evaluation
# ---------------------------------------------------------------------------


def proxy_cv_delta(
    X: np.ndarray, y: np.ndarray, fake_scaled: np.ndarray, cfg: GANConfig
) -> dict[str, float]:
    """Does adding this generator's output help a cheap model?

    Returns the mean baseline error, the mean augmented error, and a control
    signal: the same comparison with the synthetic *labels* shuffled. A useful
    generator beats its own shuffled labels, so the control being positive is
    what distinguishes learned composition-property structure from the mere
    regularizing effect of extra points.

    The synthetic set is shared across folds rather than regenerated per fold,
    and the generator that produced it was trained on the whole training split.
    Fold-validation rows therefore influenced the synthetic data. The bias runs
    in favor of augmentation, which is the direction that makes the
    manuscript's finding -- that augmentation did not always win -- the
    conservative reading. It is stated in the README.
    """
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPRegressor

    kfold = KFold(n_splits=cfg.proxy_kfold_splits, shuffle=True, random_state=cfg.seed)
    X_fake, y_fake = fake_scaled[:, :N_ILR], fake_scaled[:, N_ILR:]

    base_maes, aug_maes, controls = [], [], []

    def _model():
        return MLPRegressor(
            hidden_layer_sizes=tuple(cfg.proxy_hidden),
            max_iter=cfg.proxy_max_iter,
            random_state=cfg.seed,
        )

    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        base = _model().fit(X_tr, y_tr)
        base_mae = mean_absolute_error(y_val, base.predict(X_val))

        aug = _model().fit(np.vstack([X_tr, X_fake]), np.vstack([y_tr, y_fake]))
        aug_mae = mean_absolute_error(y_val, aug.predict(X_val))

        shuffled = np.random.permutation(y_fake)
        control = _model().fit(np.vstack([X_tr, X_fake]), np.vstack([y_tr, shuffled]))
        control_mae = mean_absolute_error(y_val, control.predict(X_val))

        base_maes.append(base_mae)
        aug_maes.append(aug_mae)
        controls.append(control_mae - aug_mae)

    return {
        "base_mae": float(np.mean(base_maes)),
        "aug_mae": float(np.mean(aug_maes)),
        "delta_mae": float(np.mean(aug_maes) - np.mean(base_maes)),
        "control_signal": float(np.mean(controls)),
    }


def _draw_filtered(
    generator,
    scaler,
    measured_props: np.ndarray,
    cfg: GANConfig,
    seed: int,
    model,
    x_scaler,
    y_scaler,
    thresholds,
    use_surrogate: bool,
) -> tuple[np.ndarray, int]:
    """Draw until `fake_pool_min` samples pass, then take a diverse subset."""
    rng = np.random.default_rng(int(seed))
    kept: list[np.ndarray] = []

    for _ in range(cfg.fake_max_rounds):
        z = rng.normal(size=(cfg.fake_draw_batch, cfg.latent_dim)).astype(np.float32)
        scaled = generator(z, training=False).numpy()
        raw = scaler.inverse_transform(scaled)
        ilr7, props = raw[:, :N_ILR], raw[:, N_ILR:]

        mask, comp = physical_mask(ilr7, cfg)
        mask &= property_mask(props, measured_props, cfg)

        if use_surrogate:
            surrogate_ok, _, _ = surrogate_consistency_mask(
                comp, props, model, x_scaler, y_scaler, thresholds
            )
            mask &= surrogate_ok

        if mask.any():
            kept.append(scaled[mask])
            if sum(len(k) for k in kept) >= cfg.fake_pool_min:
                break

    if not kept:
        return np.empty((0, INPUT_DIM), dtype=np.float32), 0

    pool = np.vstack(kept)
    if pool.shape[0] < cfg.fake_select_k:
        return np.empty((0, INPUT_DIM), dtype=np.float32), int(pool.shape[0])

    features = pool.astype(np.float64).copy()
    features[:, N_ILR:] *= float(cfg.kcenter_property_weight)
    chosen = kcenter_select(features, cfg.fake_select_k)
    return pool[chosen].astype(np.float32), int(pool.shape[0])


def _build_final_pool(
    generator,
    scaler,
    measured_props: np.ndarray,
    forbidden: set[tuple[int, ...]],
    cfg: GANConfig,
    model,
    x_scaler,
    y_scaler,
    thresholds,
    verbose: bool = True,
) -> pd.DataFrame:
    """Accumulate unique, filtered, grid-snapped candidates."""
    rng = np.random.default_rng(int(cfg.pool_seed))
    resolution = int(round(1.0 / cfg.snap_step))
    pools: list[pd.DataFrame] = []
    combined = pd.DataFrame()

    for attempt in range(cfg.final_max_rounds):
        z = rng.normal(size=(cfg.final_draw_per_round, cfg.latent_dim)).astype(np.float32)
        raw = scaler.inverse_transform(generator(z, training=False).numpy())
        ilr7, props = raw[:, :N_ILR], raw[:, N_ILR:]

        mask, comp = physical_mask(ilr7, cfg)
        mask &= property_mask(props, measured_props, cfg)
        if not mask.any():
            continue
        comp, props = comp[mask], props[mask]

        snapped = quantize_on_grid(comp, step=cfg.snap_step)
        close = np.linalg.norm(comp - snapped, axis=1) < cfg.snap_distance_max
        snapped, props = snapped[close], props[close]
        if len(snapped) == 0:
            continue

        surrogate_ok, predicted, diff = surrogate_consistency_mask(
            snapped, props, model, x_scaler, y_scaler, thresholds
        )
        if not surrogate_ok.any():
            continue

        df = pd.DataFrame(snapped[surrogate_ok], columns=COMP_COLS)
        df[TARGET_COLS[0]] = props[surrogate_ok][:, 0]
        df[TARGET_COLS[1]] = props[surrogate_ok][:, 1]
        df["kxx_surrogate"] = predicted[surrogate_ok][:, 0]
        df["S_ANE_surrogate"] = predicted[surrogate_ok][:, 1]
        df["kxx_diff_surrogate"] = diff[surrogate_ok][:, 0]
        df["S_ANE_diff_surrogate"] = diff[surrogate_ok][:, 1]
        df["source"] = "gan_surrogate_filtered" if cfg.use_surrogate_filter else "gan_raw"
        df["sample_weight"] = cfg.generated_sample_weight

        # the constraints again, now that the compositions have been snapped
        # onto the grid and are no longer exactly what was filtered above
        keep, _ = physical_mask(ilr_transform(df[COMP_COLS].to_numpy()), cfg)
        keep &= df["Co"].to_numpy() < cfg.co_max
        df = df.loc[keep]
        if len(df) == 0:
            continue

        keys = (df[COMP_COLS].to_numpy() * resolution + 1e-9).astype(int)
        unseen = np.array([tuple(row) not in forbidden for row in keys], dtype=bool)
        df = df.loc[unseen]
        if len(df) == 0:
            continue

        pools.append(df)
        combined = pd.concat(pools, ignore_index=True)
        keys = (combined[COMP_COLS].to_numpy() * resolution + 1e-9).astype(int)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        combined = combined.iloc[np.sort(unique_idx)].reset_index(drop=True)

        if verbose:
            print(f"  round {attempt + 1}: {len(combined)} unique candidates")
        if len(combined) >= cfg.target_unique_samples:
            break

    return combined


def train_gan(cfg: GANConfig, verbose: bool = True) -> dict[str, Any]:
    """Run stage 1: train the generator, then emit the augmented datasets."""
    import joblib
    import tensorflow as tf

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(cfg.train_csv)
    df_test = pd.read_csv(cfg.test_csv)

    comp_train = df_train[COMP_COLS].to_numpy(dtype=np.float64)
    prop_train = df_train[TARGET_COLS].to_numpy(dtype=np.float64)
    comp_test = df_test[COMP_COLS].to_numpy(dtype=np.float64)

    # test compositions are read for one purpose only: so that the generator's
    # output cannot coincide with a held-out sample. No test property is used,
    # and the critic never sees a test row.
    forbidden = composition_keys(
        np.vstack([comp_train, comp_test]), step=cfg.snap_step
    )

    rare, counts = rare_elements(comp_train, cfg)
    probabilities = sample_probabilities(comp_train, rare, counts, cfg)
    if verbose:
        print(f"  {len(df_train)} measured rows; oversampling {rare or 'nothing'}")

    raw_train = np.hstack([ilr_transform(comp_train), prop_train])
    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(raw_train)
    X_proxy, y_proxy = scaled_train[:, :N_ILR], scaled_train[:, N_ILR:]
    scaled_train_tf = tf.cast(scaled_train, tf.float32)

    model = x_scaler = y_scaler = thresholds = None
    if cfg.use_surrogate_filter:
        surrogate_dir = Path(cfg.surrogate_model_dir)
        model = tf.keras.models.load_model(
            surrogate_dir / "final_best_model.h5", compile=False
        )
        x_scaler = joblib.load(surrogate_dir / "sx_final.joblib")
        y_scaler = joblib.load(surrogate_dir / "sy_final.joblib")
        thresholds = surrogate_threshold(
            comp_train, prop_train, model, x_scaler, y_scaler, cfg, verbose
        )

    generator = build_generator(cfg)
    critic = build_critic(cfg)
    gen_opt = tf.keras.optimizers.Adam(
        cfg.learning_rate, beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2
    )
    critic_opt = tf.keras.optimizers.Adam(
        cfg.learning_rate, beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2
    )

    @tf.function
    def critic_step(real, noise_std):
        batch = tf.shape(real)[0]
        real_noisy = real + tf.random.normal(tf.shape(real), 0.0, noise_std)
        z = tf.random.normal([batch, cfg.latent_dim])

        with tf.GradientTape() as tape:
            fake = generator(z, training=True)
            fake_noisy = fake + tf.random.normal(tf.shape(fake), 0.0, noise_std)

            wasserstein = tf.reduce_mean(critic(fake_noisy, training=True)) - tf.reduce_mean(
                critic(real_noisy, training=True)
            )

            alpha = tf.random.uniform([batch, 1], 0.0, 1.0)
            interpolated = real_noisy + alpha * (fake_noisy - real_noisy)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                prediction = critic(interpolated, training=True)
            gradients = gp_tape.gradient(prediction, interpolated)
            penalty = tf.reduce_mean(
                (tf.norm(tf.reshape(gradients, [batch, -1]), axis=1) - 1.0) ** 2
            )
            loss = wasserstein + cfg.gradient_penalty_weight * penalty

        critic_opt.apply_gradients(
            zip(tape.gradient(loss, critic.trainable_variables), critic.trainable_variables)
        )
        return loss

    @tf.function
    def generator_step(batch_size):
        z = tf.random.normal([batch_size, cfg.latent_dim])
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(critic(generator(z, training=True), training=True))
        gen_opt.apply_gradients(
            zip(tape.gradient(loss, generator.trainable_variables), generator.trainable_variables)
        )
        return loss

    checkpoint = out_dir / "generator_best.weights.h5"
    checkpoint.unlink(missing_ok=True)

    best_delta = float("inf")
    best_epoch = -1
    waited = 0

    for epoch in range(cfg.epochs):
        # instance noise annealed to zero: heavy early, absent by the end
        noise = tf.constant(
            cfg.instance_noise_std * (1.0 - epoch / cfg.epochs), dtype=tf.float32
        )
        for _ in range(cfg.critic_iterations):
            idx = np.random.choice(
                len(scaled_train), size=cfg.batch_size, replace=True, p=probabilities
            )
            critic_step(tf.gather(scaled_train_tf, idx), noise)
        generator_step(cfg.batch_size)

        if (epoch + 1) % cfg.eval_interval:
            continue

        fake, pool_size = _draw_filtered(
            generator,
            scaler,
            prop_train,
            cfg,
            seed=100000 + epoch + 1,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            thresholds=thresholds,
            use_surrogate=cfg.use_surrogate_filter
            and (epoch + 1) >= cfg.surrogate_warmup_epoch,
        )

        if len(fake) == 0:
            waited += 1
            if verbose:
                print(f"  epoch {epoch + 1}: pool {pool_size} too small to evaluate")
            if waited >= cfg.early_stopping_patience:
                break
            continue

        scores = proxy_cv_delta(X_proxy, y_proxy, fake, cfg)
        improved = (
            scores["delta_mae"] < -cfg.improvement_tolerance
            and scores["control_signal"] > 0
            and scores["delta_mae"] < best_delta
        )

        if verbose:
            print(
                f"  epoch {epoch + 1}: base {scores['base_mae']:.4f} "
                f"aug {scores['aug_mae']:.4f} "
                f"(delta {scores['delta_mae']:+.4f}, "
                f"control {scores['control_signal']:+.4f}, pool {pool_size})"
                + ("  <- kept" if improved else "")
            )

        if improved:
            best_delta = scores["delta_mae"]
            best_epoch = epoch + 1
            generator.save_weights(checkpoint)
            waited = 0
        else:
            waited += 1
            if waited >= cfg.early_stopping_patience:
                break

    if best_epoch <= 0 or not checkpoint.exists():
        raise RuntimeError(
            "no checkpoint met the proxy criterion in this run, so no synthetic "
            "data is emitted. A generator that never improved the proxy has not "
            "been shown to produce anything worth training on."
        )

    generator.load_weights(checkpoint)
    if verbose:
        print(f"  best generator from epoch {best_epoch} (delta {best_delta:+.4f})")

    pool = _build_final_pool(
        generator, scaler, prop_train, forbidden, cfg,
        model, x_scaler, y_scaler, thresholds, verbose,
    )
    if len(pool) == 0:
        raise RuntimeError("no generated composition survived the filters")

    features = np.hstack(
        [ilr_transform(pool[COMP_COLS].to_numpy()), pool[TARGET_COLS].to_numpy()]
    )
    features = RobustScaler().fit_transform(features)
    features[:, N_ILR:] *= float(cfg.kcenter_property_weight)

    written = {}
    for n in cfg.generated_sizes:
        chosen = kcenter_select(features, min(n, len(pool)))
        path = f"{cfg.output_prefix}{n}.csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pool.iloc[chosen].reset_index(drop=True).to_csv(path, index=False)
        written[n] = path
        if verbose:
            print(f"  wrote {len(chosen)} rows to {path}")

    manifest = {
        "best_epoch": best_epoch,
        "best_delta_mae": best_delta,
        "unique_pool_size": int(len(pool)),
        "oversampled_elements": rare,
        "surrogate_filter": cfg.use_surrogate_filter,
        "surrogate_thresholds": thresholds,
        "surrogate_model_dir": cfg.surrogate_model_dir if cfg.use_surrogate_filter else None,
        "written": written,
    }
    with open(out_dir / "gan_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest
