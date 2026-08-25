"""Candidate enumeration and three-objective Pareto ranking (Algorithm S5).

This is the stage that decides what gets synthesized next. It enumerates
compositions on a fixed grid subject to the physical constraints, predicts
kappa and S_ANE for each with the trained deep ensemble, and returns two sets
of five: an exploitation set ranked by predicted S_ANE / kappa, and an
exploration set ranked by ensemble disagreement.

The three objectives are all maximized:

    1 / kappa_mean          low thermal conductivity
    S_ANE_mean              high anomalous Nernst coefficient
    uncertainty             ensemble disagreement

Putting uncertainty on the front rather than in a scalarized acquisition
function is what keeps the exploration set from collapsing onto the
exploitation set: a candidate the ensemble is loudly unsure about survives to
the front even when its predicted performance is unremarkable.

Two conventions here are load-bearing and easy to change by accident.

`uncertainty` is the Euclidean norm of the two member-level standard
deviations after each is divided by its own median across the candidate pool.
The normalization is what makes the two comparable -- they carry different
units and differ by orders of magnitude -- and it means the value is defined
relative to the pool being ranked, not absolutely. Uncertainties from two
different runs are not on the same scale and must not be compared.

The inverse conductivity used for the spread is formed per ensemble member and
then averaged, rather than inverting the mean. Those are different quantities,
and only the first carries information about member disagreement in the
objective actually being maximized. Both are computed and stored: the Pareto
objective uses `Kxx_inv_from_mean` (the inverse of the mean), while the
uncertainty uses the spread of the per-member inverses.

Column names in the emitted tables follow the original notebook -- `Kxx_*` for
kappa and `Syx_*` for S_ANE -- so that artifacts from earlier cycles remain
readable by the same scripts. They map onto `data.PROPERTY_COLUMNS` in order:
the surrogate emits (kxx, S_ANE), so column 0 is Kxx and column 1 is Syx.

`tests/test_select_parity.py` checks the enumeration, the filters, the Pareto
mask and the ranking numerically against the original implementation.
"""

from __future__ import annotations

import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import SelectionConfig
from .elements import ELEMENTS, GROUP_A, GROUP_B
from .features import featurize
from .physics import delta_h_mix

__all__ = [
    "sample_positive_partition",
    "generate_candidates",
    "feasibility_mask",
    "EnsemblePredictor",
    "pareto_mask_3d_max",
    "compute_uncertainty",
    "select_diverse_top_k",
    "rank_candidates",
    "run_selection",
]


# ---------------------------------------------------------------------------
# candidate enumeration
# ---------------------------------------------------------------------------


def sample_positive_partition(
    total: int, k: int, rng: np.random.Generator
) -> tuple[int, ...] | None:
    """Split an integer `total` into `k` strictly positive parts.

    Not a uniform sampler over compositions, despite what an earlier version of
    this docstring said. The k-1 cut points are drawn with replacement and then
    sorted, which favors interior partitions over extreme ones: for total = 6,
    k = 3 the most frequent partition appears about twice as often as the least.
    The behavior is kept because it is what enumerated the candidate pool the
    campaign ranked; it means the pool is a biased sample of the grid, not a
    census of it.

    Returns None when no such partition exists. The k-1 cut points are drawn
    from `total - k` so that every part receives at least one unit after the
    final shift, which is why the space of compositions with a component of
    exactly zero is not reachable here: absent elements are handled by choosing
    which elements are active, not by letting a part go to zero.
    """
    if total < k:
        return None
    if k == 1:
        return (total,)

    rest = total - k
    cuts = np.sort(rng.integers(0, rest + 1, size=k - 1))
    zs = np.empty(k, dtype=int)
    prev = 0
    for i, c in enumerate(cuts):
        zs[i] = c - prev
        prev = c
    zs[-1] = rest - prev
    return tuple(int(z + 1) for z in zs)


def generate_candidates(cfg: SelectionConfig) -> pd.DataFrame:
    """Enumerate candidate compositions by grouped random grid sampling.

    The space is too large to enumerate exhaustively at a 0.01 grid, so it is
    sampled by case: for each split of the total between substitution groups A
    and B, and each choice of which elements are active in each group, up to
    `limit_per_case` random partitions are drawn. Sampling per case rather than
    globally is deliberate -- it keeps sparse compositions (three active
    elements) from being swamped by dense ones, which is where the measured
    alloys actually live.

    The Co and Mn bounds are applied here as well as in `feasibility_mask`,
    because rejecting them during sampling lets the per-case budget be spent on
    compositions that can survive the filter.
    """
    rng = np.random.default_rng(cfg.generation_seed)
    units = int(round(1.0 / cfg.grid_step))

    # convert the x_A / x_B ratio bounds into bounds on the group-A fraction
    a_min = cfg.stoichiometry_min / (1.0 + cfg.stoichiometry_min)
    a_max = cfg.stoichiometry_max / (1.0 + cfg.stoichiometry_max)
    au_min = int(np.ceil(a_min * units - 1e-12))
    au_max = int(np.floor(a_max * units + 1e-12))

    idx_a = [ELEMENTS.index(e) for e in GROUP_A]
    idx_b = [ELEMENTS.index(e) for e in GROUP_B]
    mn_idx = ELEMENTS.index("Mn")
    co_idx = ELEMENTS.index("Co")

    bounds_mn = tuple(cfg.element_bounds.get("Mn", (0.0, 1.0)))
    bounds_co = tuple(cfg.element_bounds.get("Co", (0.0, 1.0)))

    rows: list[np.ndarray] = []

    for au in range(au_min, au_max + 1):
        bu = units - au
        if bu <= 0:
            continue

        for ka in cfg.n_group_a:
            for kb in cfg.n_group_b:
                for act_a in combinations(idx_a, ka):
                    for act_b in combinations(idx_b, kb):
                        produced = 0
                        trials = 0
                        max_trials = cfg.limit_per_case * 20

                        while produced < cfg.limit_per_case and trials < max_trials:
                            trials += 1

                            part_a = sample_positive_partition(au, ka, rng)
                            if part_a is None:
                                break

                            if mn_idx in act_a:
                                mn_frac = part_a[act_a.index(mn_idx)] / units
                                lo, hi = bounds_mn
                                if not (lo - 1e-12 <= mn_frac <= hi + 1e-12):
                                    continue

                            if co_idx in act_a:
                                co_frac = part_a[act_a.index(co_idx)] / units
                                lo, hi = bounds_co
                                if not (lo - 1e-12 <= co_frac <= hi + 1e-12):
                                    continue

                            part_b = sample_positive_partition(bu, kb, rng)
                            if part_b is None:
                                break

                            vec = np.zeros(len(ELEMENTS), dtype=np.float64)
                            for i, u in enumerate(part_a):
                                vec[act_a[i]] = u * cfg.grid_step
                            for j, u in enumerate(part_b):
                                vec[act_b[j]] = u * cfg.grid_step

                            rows.append(vec)
                            produced += 1

    if not rows:
        return pd.DataFrame(columns=list(ELEMENTS))

    df = pd.DataFrame(rows, columns=list(ELEMENTS))
    return df.drop_duplicates().reset_index(drop=True)


def feasibility_mask(df_comp: pd.DataFrame, cfg: SelectionConfig) -> np.ndarray:
    """Hard physical constraints: stoichiometry, element bounds, dH_mix <= 0.

    Applied as a mask rather than a penalty. A composition outside the
    target stoichiometry window or with positive pairwise mixing enthalpy is not
    ranked as a worse candidate but excluded, on the heuristic that it is
    unlikely to form as the intended phase. The screen is a Miedema pairwise sum,
    not an equilibrium phase-stability calculation. The surrogate has no
    training data there to say otherwise.
    """
    if len(df_comp) == 0:
        return np.zeros(0, dtype=bool)

    comp = df_comp[list(ELEMENTS)].to_numpy(dtype=np.float64)
    comp = comp / np.clip(comp.sum(axis=1, keepdims=True), 1e-12, None)

    idx_a = [ELEMENTS.index(e) for e in GROUP_A]
    idx_b = [ELEMENTS.index(e) for e in GROUP_B]

    sum_a = comp[:, idx_a].sum(axis=1)
    sum_b = comp[:, idx_b].sum(axis=1)
    ratio = sum_a / np.clip(sum_b, 1e-12, None)

    valid = (sum_a > 1e-12) & (sum_b > 1e-12)
    valid &= (ratio >= cfg.stoichiometry_min) & (ratio <= cfg.stoichiometry_max)

    for el, bnd in cfg.element_bounds.items():
        if bnd is None:
            continue
        lo, hi = tuple(bnd)
        j = ELEMENTS.index(el)
        valid &= (comp[:, j] >= lo - 1e-12) & (comp[:, j] <= hi + 1e-12)

    hmix = delta_h_mix(comp)
    valid &= hmix <= cfg.h_mix_max + 1e-12

    return valid


# ---------------------------------------------------------------------------
# ensemble prediction
# ---------------------------------------------------------------------------


def find_ensemble_model_files(ensemble_dir: str | Path) -> list[str]:
    """Model files sorted by member index, not by filename.

    Lexicographic order would put `ensemble_10` before `ensemble_2`. The order
    does not affect the mean or the spread, but it does decide which member a
    saved per-member prediction belongs to, so it is fixed here.
    """
    root = Path(ensemble_dir)
    if not root.exists():
        raise FileNotFoundError(f"ensemble directory not found: {ensemble_dir}")

    found: list[tuple[int, str]] = []
    for p in root.iterdir():
        m = re.match(r"ensemble_(\d+)\.h5$", p.name)
        if m:
            found.append((int(m.group(1)), str(p)))

    return [path for _, path in sorted(found, key=lambda x: x[0])]


class EnsemblePredictor:
    """Deep ensemble loaded from disk, applied to compositions.

    Takes compositions rather than features, and calls `ane.features.featurize`
    itself. That is the point: the scalers stored alongside the models were fit
    on features from that function, and letting a caller supply its own feature
    matrix is how a surrogate ends up applied to inputs computed differently
    from those it was trained on.
    """

    def __init__(self, cfg: SelectionConfig, verbose: bool = True) -> None:
        # imported here, not at module scope, so that the enumeration, the
        # filters and the ranking -- and their parity tests -- can be used
        # without TensorFlow installed
        import joblib
        import tensorflow as tf

        self.cfg = cfg
        ensemble_dir = cfg.ensemble_dir
        sx_path = os.path.join(ensemble_dir, "sx_scaler.joblib")
        sy_path = os.path.join(ensemble_dir, "sy_scaler.joblib")

        for path, what in ((sx_path, "input"), (sy_path, "output")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{what} scaler not found: {path}")

        self.sx = joblib.load(sx_path)
        self.sy = joblib.load(sy_path)

        model_paths = find_ensemble_model_files(ensemble_dir)
        if len(model_paths) < cfg.min_ensemble_models:
            raise RuntimeError(
                f"too few ensemble members: {len(model_paths)} found in "
                f"{ensemble_dir}, {cfg.min_ensemble_models} required. The spread "
                f"across members is an objective, so a thin ensemble does not "
                f"just add noise, it changes the ranking."
            )

        self.model_paths = model_paths
        self.models = [tf.keras.models.load_model(p, compile=False) for p in model_paths]

        if verbose:
            print(f"  loaded {len(self.models)} ensemble members from {ensemble_dir}")
            print(f"  input scaler expects {getattr(self.sx, 'n_features_in_', '?')} features")

    def _predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        preds = [m(X_scaled, training=False).numpy() for m in self.models]
        return np.asarray(preds, dtype=np.float64)  # (n_models, batch, 2)

    def predict(
        self, compositions: np.ndarray, batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        """Per-candidate ensemble mean and spread for kappa and S_ANE."""
        batch_size = batch_size or self.cfg.predict_batch_size
        X_feat = featurize(compositions)

        expected = getattr(self.sx, "n_features_in_", None)
        if expected is not None and X_feat.shape[1] != expected:
            raise ValueError(
                f"feature width {X_feat.shape[1]} does not match the "
                f"{expected} the stored scaler was fit on"
            )

        X_scaled = self.sx.transform(X_feat).astype(np.float32)

        batches = []
        for start in range(0, len(X_scaled), batch_size):
            scaled = self._predict_scaled(X_scaled[start : start + batch_size])
            batches.append(
                np.asarray(
                    [self.sy.inverse_transform(p) for p in scaled], dtype=np.float64
                )
            )
        preds = np.concatenate(batches, axis=1)  # (n_models, n_candidates, 2)

        return summarize_predictions(preds, self.cfg)


def summarize_predictions(
    preds: np.ndarray, cfg: SelectionConfig
) -> dict[str, np.ndarray]:
    """Reduce per-member predictions to means and spreads.

    Separated from `EnsemblePredictor` so it can be tested without TensorFlow,
    and so a saved per-member prediction array can be re-summarized without
    re-running the models.

    `preds` has shape (n_models, n_candidates, 2), the last axis ordered as
    `data.PROPERTY_COLUMNS`.
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 3 or preds.shape[2] != 2:
        raise ValueError(f"expected an (n_models, n, 2) array, got {preds.shape}")

    kxx = preds[:, :, 0]
    syx = preds[:, :, 1]

    eps = cfg.kappa_inverse_epsilon
    if cfg.kappa_inverse_mode == "add_eps":
        kxx_inv = 1.0 / (kxx + eps)
    elif cfg.kappa_inverse_mode == "clip_positive":
        kxx_inv = 1.0 / np.clip(kxx, eps, None)
    else:
        raise ValueError(f"unknown kappa_inverse_mode {cfg.kappa_inverse_mode!r}")

    return {
        "Kxx_mean": kxx.mean(axis=0),
        "Kxx_std": kxx.std(axis=0),
        "Syx_mean": syx.mean(axis=0),
        "Syx_std": syx.std(axis=0),
        "Kxx_inv_mean": kxx_inv.mean(axis=0),
        "Kxx_inv_std": kxx_inv.std(axis=0),
        "n_models": np.full(kxx.shape[1], kxx.shape[0], dtype=int),
    }


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


def pareto_mask_3d_max(
    obj0: np.ndarray, obj1: np.ndarray, obj2: np.ndarray
) -> np.ndarray:
    """Non-dominated mask for three maximized objectives, in O(n log n).

    Sorts by obj0 descending and sweeps, keeping the running maximum of obj2
    over the obj1 values already seen in a Fenwick tree indexed by obj1 rank.
    A point is dominated if some earlier point had obj1 at least as large and
    obj2 at least as large. Ties in obj0 are handled as a separate group so
    that equal points do not eliminate one another spuriously; the pairwise
    O(n^2) form is unusable at the ~10^5 candidates this stage produces.
    """
    obj0 = np.asarray(obj0, dtype=np.float64)
    obj1 = np.asarray(obj1, dtype=np.float64)
    obj2 = np.asarray(obj2, dtype=np.float64)

    if not (obj0.shape == obj1.shape == obj2.shape):
        raise ValueError("the three objective arrays must have the same shape")

    finite = np.isfinite(obj0) & np.isfinite(obj1) & np.isfinite(obj2)
    original_indices = np.where(finite)[0]
    if original_indices.size == 0:
        return np.zeros(obj0.size, dtype=bool)

    o0_f, o1_f, o2_f = obj0[finite], obj1[finite], obj2[finite]

    n = o0_f.size
    order = np.lexsort((-o2_f, -o1_f, -o0_f))
    o0, o1, o2 = o0_f[order], o1_f[order], o2_f[order]

    uniq1, inv1 = np.unique(o1, return_inverse=True)
    m = uniq1.size
    rank = (m - inv1).astype(np.int32)      # descending rank, 1-based
    bit = np.full(m + 1, -np.inf, dtype=np.float64)

    def bit_query(p: int) -> float:
        res = -np.inf
        while p > 0:
            if bit[p] > res:
                res = bit[p]
            p -= p & -p
        return res

    def bit_update(p: int, val: float) -> None:
        while p <= m:
            if val > bit[p]:
                bit[p] = val
            p += p & -p

    is_pareto_sorted = np.ones(n, dtype=bool)

    i = 0
    while i < n:
        j = i + 1
        while j < n and o0[j] == o0[i]:
            j += 1

        # dominated by an earlier group, i.e. one with strictly larger obj0
        for k in range(i, j):
            if bit_query(rank[k]) >= o2[k]:
                is_pareto_sorted[k] = False

        # domination inside the current obj0 group
        max_o2_higher_o1 = -np.inf
        k = i
        while k < j:
            curr_o1 = o1[k]
            run_start = k
            run_max_o2 = o2[k]
            k += 1
            while k < j and o1[k] == curr_o1:
                k += 1
            run_end = k

            if max_o2_higher_o1 >= run_max_o2:
                is_pareto_sorted[run_start:run_end] = False
            else:
                for t in range(run_start, run_end):
                    if o2[t] != run_max_o2:
                        is_pareto_sorted[t] = False

            if run_max_o2 > max_o2_higher_o1:
                max_o2_higher_o1 = run_max_o2

        for k in range(i, j):
            bit_update(rank[k], o2[k])

        i = j

    in_sorted_order = np.zeros(n, dtype=bool)
    in_sorted_order[order] = is_pareto_sorted

    mask = np.zeros(obj0.size, dtype=bool)
    mask[original_indices] = in_sorted_order
    return mask


def compute_uncertainty(
    kxx_inv_std: np.ndarray, syx_std: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Median-normalized combination of the two ensemble spreads.

    Each spread is divided by its median over the candidate pool before the two
    are combined, because they carry different units and differ by orders of
    magnitude; without it the objective would be whichever happens to be
    larger. The medians are returned so the run can record them: the resulting
    uncertainty is relative to this pool and is not comparable across runs.
    """
    kstd = np.asarray(kxx_inv_std, dtype=np.float64)
    sstd = np.asarray(syx_std, dtype=np.float64)

    scale_k = float(np.nanmedian(kstd))
    scale_s = float(np.nanmedian(sstd))

    # a degenerate median means the ensemble agrees exactly, which in practice
    # means something upstream is wrong; fall back rather than divide by zero
    if not np.isfinite(scale_k) or scale_k <= 0:
        scale_k = 1e-12
    if not np.isfinite(scale_s) or scale_s <= 0:
        scale_s = 1e-12

    uncertainty = np.sqrt((kstd / scale_k) ** 2 + (sstd / scale_s) ** 2)
    return uncertainty, scale_k, scale_s


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def select_diverse_top_k(
    df_sorted: pd.DataFrame,
    comp_cols: Iterable[str],
    k: int,
    dist_th: float,
    exclude_indices: set[int] | None = None,
) -> list[int]:
    """Take the top k rows, skipping any within `dist_th` of one already taken.

    Distance is the Euclidean norm between composition vectors. Without this
    the five selected candidates come back as five variations on one alloy
    differing by a single grid step, which wastes a synthesis round: the
    surrogate learns almost nothing from five points that close together.

    If diversity cannot be satisfied -- a narrow front, a large threshold --
    the remaining slots are filled by rank order rather than returning fewer
    than k, since the cycle needs a fixed batch size.
    """
    exclude_indices = exclude_indices or set()
    comp_cols = list(comp_cols)
    selected_idx: list[int] = []
    selected_comps: list[np.ndarray] = []

    for idx, row in df_sorted.iterrows():
        if idx in exclude_indices:
            continue

        comp = row[comp_cols].to_numpy(dtype=np.float64)
        if selected_comps:
            dmin = min(float(np.linalg.norm(comp - c)) for c in selected_comps)
            if dmin < dist_th:
                continue

        selected_idx.append(int(idx))
        selected_comps.append(comp)
        if len(selected_idx) >= k:
            return selected_idx

    for idx, _ in df_sorted.iterrows():
        if idx in exclude_indices or idx in selected_idx:
            continue
        selected_idx.append(int(idx))
        if len(selected_idx) >= k:
            break

    return selected_idx


def rank_candidates(
    df_valid: pd.DataFrame, cfg: SelectionConfig, verbose: bool = True
) -> tuple[pd.DataFrame, list[int], list[int], dict[str, float]]:
    """Score, build the Pareto front, and pick the two sets of five.

    `df_valid` must already carry the ensemble summary columns. Returns the
    annotated frame, the two index lists, and the uncertainty scales.

    The exploitation set is drawn from the front by S_ANE / kappa; the
    exploration set is drawn from what remains by uncertainty. Excluding the
    first from the second is what makes the batch ten distinct compositions
    rather than five plus overlap.
    """
    df = df_valid.copy()

    finite = (
        np.isfinite(df["Kxx_mean"].to_numpy(dtype=np.float64))
        & np.isfinite(df["Syx_mean"].to_numpy(dtype=np.float64))
        & np.isfinite(df["Kxx_inv_std"].to_numpy(dtype=np.float64))
        & np.isfinite(df["Syx_std"].to_numpy(dtype=np.float64))
    )
    if cfg.require_positive_kappa:
        # the objective is 1 / kappa_mean; a non-positive mean would rank a
        # physically impossible candidate at the top of the front
        finite &= df["Kxx_mean"].to_numpy(dtype=np.float64) > 0

    dropped = int((~finite).sum())
    if dropped:
        if verbose:
            print(f"  dropped {dropped} candidates with invalid or non-positive kappa")
        df = df.loc[finite].reset_index(drop=True)

    uncertainty, scale_k, scale_s = compute_uncertainty(
        df["Kxx_inv_std"].to_numpy(dtype=np.float64),
        df["Syx_std"].to_numpy(dtype=np.float64),
    )
    df["uncertainty"] = uncertainty
    df["uncert_scale_Kxx_inv_std_median"] = scale_k
    df["uncert_scale_Syx_std_median"] = scale_s

    df["Kxx_inv_from_mean"] = 1.0 / np.clip(
        df["Kxx_mean"].to_numpy(dtype=np.float64), 1e-12, None
    )
    df["Syx_over_Kxx"] = df["Syx_mean"] / np.clip(df["Kxx_mean"], 1e-12, None)

    pf_mask = pareto_mask_3d_max(
        df["Kxx_inv_from_mean"].to_numpy(dtype=np.float64),
        df["Syx_mean"].to_numpy(dtype=np.float64),
        df["uncertainty"].to_numpy(dtype=np.float64),
    )
    df["is_pareto_3d"] = pf_mask
    df_pareto = df.loc[pf_mask].copy()

    if verbose:
        print(f"  Pareto front: {len(df_pareto):,} of {len(df):,} candidates")
    if len(df_pareto) == 0:
        raise RuntimeError("the three-objective Pareto front is empty")

    if cfg.require_positive_s_ane:
        perf_pool = df_pareto.loc[df_pareto["Syx_mean"] > 0].copy()
    else:
        perf_pool = df_pareto.copy()

    if len(perf_pool) == 0:
        # the surrogate predicting no positive S_ANE anywhere on the front is a
        # symptom worth seeing, not a reason to stop the cycle
        if verbose:
            print("  warning: no candidate on the front has positive S_ANE; "
                  "ranking the whole front instead")
        perf_pool = df_pareto.copy()

    top_perf = select_diverse_top_k(
        perf_pool.sort_values(by="Syx_over_Kxx", ascending=False),
        comp_cols=ELEMENTS,
        k=cfg.top_k,
        dist_th=cfg.diversity_distance,
    )

    exclude = set(top_perf)
    top_explore = select_diverse_top_k(
        df_pareto.drop(index=list(exclude), errors="ignore").sort_values(
            by="uncertainty", ascending=False
        ),
        comp_cols=ELEMENTS,
        k=cfg.top_k,
        dist_th=cfg.diversity_distance,
        exclude_indices=exclude,
    )

    df["is_top_performance"] = False
    df.loc[top_perf, "is_top_performance"] = True
    df["is_top_exploration"] = False
    df.loc[top_explore, "is_top_exploration"] = True

    return df, top_perf, top_explore, {"scale_k": scale_k, "scale_s": scale_s}


# ---------------------------------------------------------------------------
# stage entry point
# ---------------------------------------------------------------------------


def run_selection(
    cfg: SelectionConfig, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the whole stage and write the tables. Returns (all, perf, explore)."""
    if verbose:
        print(f"[1/5] enumerating candidates on a {cfg.grid_step} grid")
    df_cand = generate_candidates(cfg)
    if verbose:
        print(f"  {len(df_cand):,} distinct compositions")

    if verbose:
        print("[2/5] applying physical constraints")
    df_valid = (
        df_cand.loc[feasibility_mask(df_cand, cfg)]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if verbose:
        print(f"  {len(df_valid):,} feasible")
    if len(df_valid) == 0:
        raise RuntimeError("no candidate survived the physical constraints")

    if verbose:
        print("[3/5] loading the ensemble and predicting")
    predictor = EnsemblePredictor(cfg, verbose=verbose)
    preds = predictor.predict(df_valid[list(ELEMENTS)].to_numpy(dtype=np.float64))
    for key, value in preds.items():
        df_valid[key] = value if key == "n_models" else np.asarray(value, dtype=np.float64)

    if verbose:
        print("[4/5] ranking")
    df, top_perf_idx, top_explore_idx, scales = rank_candidates(df_valid, cfg, verbose)
    top_perf = df.loc[top_perf_idx].copy()
    top_explore = df.loc[top_explore_idx].copy()

    if verbose:
        print("[5/5] writing results")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = cfg.output_prefix

    df.to_csv(out / f"{prefix}_results.csv", index=False)
    df.loc[df["is_pareto_3d"]].to_csv(out / f"{prefix}_pareto_front.csv", index=False)
    top_perf.to_csv(out / f"{prefix}_top{cfg.top_k}_performance.csv", index=False)
    top_explore.to_csv(out / f"{prefix}_top{cfg.top_k}_exploration.csv", index=False)

    metadata: dict[str, Any] = {
        "ensemble_dir": cfg.ensemble_dir,
        "loaded_models": len(predictor.models),
        "grid_step": cfg.grid_step,
        "limit_per_case": cfg.limit_per_case,
        "generation_seed": cfg.generation_seed,
        "valid_candidates": int(len(df)),
        "pareto_candidates": int(df["is_pareto_3d"].sum()),
        "top_k": cfg.top_k,
        "diversity_distance": cfg.diversity_distance,
        "kappa_inverse_mode": cfg.kappa_inverse_mode,
        "require_positive_kappa": cfg.require_positive_kappa,
        "require_positive_s_ane": cfg.require_positive_s_ane,
        "objectives": [
            "maximize 1 / Kxx_mean",
            "maximize Syx_mean",
            "maximize uncertainty",
        ],
        "uncertainty_definition": (
            "sqrt((Kxx_inv_std / median(Kxx_inv_std))^2 "
            "+ (Syx_std / median(Syx_std))^2)"
        ),
        "uncert_scale_Kxx_inv_std_median": scales["scale_k"],
        "uncert_scale_Syx_std_median": scales["scale_s"],
    }
    with open(out / f"{prefix}_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    if verbose:
        cols: Sequence[str] = ["Kxx_mean", "Syx_mean", "Syx_over_Kxx", "uncertainty"]
        print(f"\ntop {cfg.top_k} by predicted performance")
        print(top_perf[list(cols) + list(ELEMENTS)].to_string(index=False))
        print(f"\ntop {cfg.top_k} by ensemble uncertainty")
        print(top_explore[list(cols) + list(ELEMENTS)].to_string(index=False))
        print(f"\nwritten to {out}/")

    return df, top_perf, top_explore
