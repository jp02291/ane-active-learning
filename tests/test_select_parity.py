"""Numerical parity between `ane.select` and the original notebook code.

The reference block below is transcribed from Pareto_v2.ipynb. Four mechanical
edits were necessary and no others: the notebook's `Config` class is renamed
`_RefConfig`, `Config.` references inside the functions follow that rename,
the domain constants are imported from `ane.elements` and `ane.physics` where
the notebook's copies are character-for-character identical, and the two
functions that took their defaults from `Config` now take them as arguments so
that both sides can be driven with the same values. Do not otherwise tidy this
code -- it is here to disagree with the port, not to be readable.

What is checked, and what is not. Everything downstream of the ensemble is
deterministic and is compared exactly: enumeration, the physical filters, the
Pareto mask, the uncertainty, the diversity-filtered ranking, and the whole
ranking stage end to end on synthetic predictions. The ensemble forward pass
itself is not: it needs the trained weights, and TensorFlow does not promise
bitwise reproducibility across machines anyway. The part of prediction that is
ours -- featurization and the reduction of per-member outputs to means and
spreads -- is compared exactly, so what is untested is Keras, not the port.

Generation is exercised at a coarser grid and a smaller per-case budget than a
production run uses. The random draws are sequence-dependent, so an ordering
difference between the two implementations shows up immediately at any size;
running the full 0.005 grid would take minutes and prove nothing further.

    python -m pytest tests/test_select_parity.py -q
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from ane.config import SelectionConfig
from ane.elements import ELEMENTS as _ANE_ELEMENTS
from ane.features import featurize
from ane.select import (
    compute_uncertainty,
    feasibility_mask,
    generate_candidates,
    pareto_mask_3d_max,
    rank_candidates,
    sample_positive_partition,
    select_diverse_top_k,
    summarize_predictions,
)

# ---------------------------------------------------------------------------
# reference implementation, from Pareto_v2.ipynb
# ---------------------------------------------------------------------------

ELEMENTS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]
GROUP_A = ["Fe", "Co", "Mn"]
GROUP_B = ["Ga", "Al", "Si", "Ge", "Pt"]

BINARY_ENTHALPY = {
    ("Fe", "Co"): -0.847634,
    ("Fe", "Mn"): 0.437463,
    ("Fe", "Ga"): -17.803602,
    ("Fe", "Al"): -32.115644,
    ("Fe", "Si"): -26.345766,
    ("Fe", "Ge"): -8.939757,
    ("Fe", "Pt"): -19.335633,
    ("Co", "Mn"): -7.757938,
    ("Co", "Ga"): -31.302334,
    ("Co", "Al"): -43.255284,
    ("Co", "Si"): -31.191669,
    ("Co", "Ge"): -17.134425,
    ("Co", "Pt"): -10.682789,
    ("Mn", "Ga"): -34.099536,
    ("Mn", "Al"): -43.499266,
    ("Mn", "Si"): -41.506930,
    ("Mn", "Ge"): -31.820084,
    ("Mn", "Pt"): -41.941500,
    ("Ga", "Al"): 1.415135,
    ("Ga", "Si"): 16.499490,
    ("Ga", "Ge"): 8.080138,
    ("Ga", "Pt"): -73.039917,
    ("Al", "Si"): 13.443439,
    ("Al", "Ge"): 9.499498,
    ("Al", "Pt"): -82.880335,
    ("Si", "Ge"): 32.998448,
    ("Si", "Pt"): -55.947653,
    ("Ge", "Pt"): -42.996605,
}


class _RefConfig:
    GRID_STEP = 0.005
    GEN_SEED = 42
    LIMIT_PER_CASE = 1000
    K_A_LIST = (1, 2, 3)
    K_B_LIST = (1, 2, 3, 4, 5)

    STOICH_MIN = 2.2
    STOICH_MAX = 3.8
    HMIX_MAX = 0.0
    ELEMENT_BOUNDS = {
        "Co": (0.00, 0.60),
        "Mn": (0.00, 0.25),
    }

    KXX_INV_MODE = "add_eps"
    KXX_INV_EPS = 1e-8
    REQUIRE_POSITIVE_KXX_MEAN = True

    TOP_K = 5
    DIVERSITY_DIST_TH = 0.10
    REQUIRE_POSITIVE_SYX_FOR_PERFORMANCE = True


def _ref_get_hij(e1: str, e2: str) -> float:
    return BINARY_ENTHALPY.get((e1, e2), BINARY_ENTHALPY.get((e2, e1), 0.0))


def _ref_sample_positive_partition(total, k, rng):
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


def _ref_generate_grouped_grid(step, limit_per_case, seed):
    rng = np.random.default_rng(seed)
    units = int(round(1.0 / step))

    a_min = _RefConfig.STOICH_MIN / (1.0 + _RefConfig.STOICH_MIN)
    a_max = _RefConfig.STOICH_MAX / (1.0 + _RefConfig.STOICH_MAX)
    au_min = int(np.ceil(a_min * units - 1e-12))
    au_max = int(np.floor(a_max * units + 1e-12))

    idx_a = [ELEMENTS.index(e) for e in GROUP_A]
    idx_b = [ELEMENTS.index(e) for e in GROUP_B]
    mn_idx = ELEMENTS.index("Mn")
    co_idx = ELEMENTS.index("Co")

    rows: List[np.ndarray] = []

    for au in range(au_min, au_max + 1):
        bu = units - au
        if bu <= 0:
            continue

        for ka in _RefConfig.K_A_LIST:
            for kb in _RefConfig.K_B_LIST:
                for act_a in combinations(idx_a, ka):
                    for act_b in combinations(idx_b, kb):
                        produced = 0
                        trials = 0
                        max_trials = limit_per_case * 20

                        while produced < limit_per_case and trials < max_trials:
                            trials += 1

                            part_a = _ref_sample_positive_partition(au, ka, rng)
                            if part_a is None:
                                break

                            if mn_idx in act_a:
                                mn_frac = part_a[act_a.index(mn_idx)] / units
                                lo, hi = _RefConfig.ELEMENT_BOUNDS.get("Mn", (0.0, 1.0))
                                if not (lo - 1e-12 <= mn_frac <= hi + 1e-12):
                                    continue

                            if co_idx in act_a:
                                co_frac = part_a[act_a.index(co_idx)] / units
                                lo, hi = _RefConfig.ELEMENT_BOUNDS.get("Co", (0.0, 1.0))
                                if not (lo - 1e-12 <= co_frac <= hi + 1e-12):
                                    continue

                            part_b = _ref_sample_positive_partition(bu, kb, rng)
                            if part_b is None:
                                break

                            vec = np.zeros(len(ELEMENTS), dtype=np.float64)
                            for i, u in enumerate(part_a):
                                vec[act_a[i]] = u * step
                            for j, u in enumerate(part_b):
                                vec[act_b[j]] = u * step

                            rows.append(vec)
                            produced += 1

    if not rows:
        return pd.DataFrame(columns=ELEMENTS)

    df = pd.DataFrame(rows, columns=ELEMENTS)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _ref_calc_delta_h_mix(compositions):
    comp = np.asarray(compositions, dtype=np.float64)
    out = np.zeros(comp.shape[0], dtype=np.float64)
    for i in range(len(ELEMENTS)):
        for j in range(i + 1, len(ELEMENTS)):
            hij = _ref_get_hij(ELEMENTS[i], ELEMENTS[j])
            out += 4.0 * hij * comp[:, i] * comp[:, j]
    return out


def _ref_feasibility_mask_only(df_comp):
    if len(df_comp) == 0:
        return np.zeros(0, dtype=bool)

    comp = df_comp[ELEMENTS].to_numpy(dtype=np.float64)
    comp = comp / np.clip(comp.sum(axis=1, keepdims=True), 1e-12, None)

    idx_a = [ELEMENTS.index(e) for e in GROUP_A]
    idx_b = [ELEMENTS.index(e) for e in GROUP_B]

    sum_a = comp[:, idx_a].sum(axis=1)
    sum_b = comp[:, idx_b].sum(axis=1)
    ratio = sum_a / np.clip(sum_b, 1e-12, None)

    valid = (sum_a > 1e-12) & (sum_b > 1e-12)
    valid &= (ratio >= _RefConfig.STOICH_MIN) & (ratio <= _RefConfig.STOICH_MAX)

    for el, bounds in _RefConfig.ELEMENT_BOUNDS.items():
        if bounds is None:
            continue
        lo, hi = bounds
        j = ELEMENTS.index(el)
        valid &= (comp[:, j] >= lo - 1e-12) & (comp[:, j] <= hi + 1e-12)

    hmix = _ref_calc_delta_h_mix(comp)
    valid &= hmix <= _RefConfig.HMIX_MAX + 1e-12

    return valid


def _ref_pareto_mask_3d_fast_max(obj0, obj1, obj2):
    obj0 = np.asarray(obj0, dtype=np.float64)
    obj1 = np.asarray(obj1, dtype=np.float64)
    obj2 = np.asarray(obj2, dtype=np.float64)

    if not (obj0.shape == obj1.shape == obj2.shape):
        raise ValueError("obj0, obj1, and obj2 must have the same shape.")

    finite = np.isfinite(obj0) & np.isfinite(obj1) & np.isfinite(obj2)
    original_indices = np.where(finite)[0]

    if original_indices.size == 0:
        return np.zeros(obj0.size, dtype=bool)

    obj0_f = obj0[finite]
    obj1_f = obj1[finite]
    obj2_f = obj2[finite]

    n = obj0_f.size
    order = np.lexsort((-obj2_f, -obj1_f, -obj0_f))
    o0 = obj0_f[order]
    o1 = obj1_f[order]
    o2 = obj2_f[order]

    uniq1, inv1 = np.unique(o1, return_inverse=True)
    m = uniq1.size
    rank = (m - inv1).astype(np.int32)
    bit = np.full(m + 1, -np.inf, dtype=np.float64)

    def bit_query(p):
        res = -np.inf
        while p > 0:
            if bit[p] > res:
                res = bit[p]
            p -= p & -p
        return res

    def bit_update(p, val):
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

        for k in range(i, j):
            if bit_query(rank[k]) >= o2[k]:
                is_pareto_sorted[k] = False

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

    is_pareto_finite_order = np.zeros(n, dtype=bool)
    is_pareto_finite_order[order] = is_pareto_sorted

    is_pareto = np.zeros(obj0.size, dtype=bool)
    is_pareto[original_indices] = is_pareto_finite_order
    return is_pareto


def _ref_select_diverse_top_k(df_sorted, comp_cols, k, dist_th, exclude_indices=None):
    exclude_indices = exclude_indices or set()
    selected_idx: List[int] = []
    selected_comps: List[np.ndarray] = []

    for idx, row in df_sorted.iterrows():
        if idx in exclude_indices:
            continue

        comp = row[list(comp_cols)].to_numpy(dtype=np.float64)

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


def _ref_compute_uncertainty(kxx_inv_std, syx_std):
    kstd = np.asarray(kxx_inv_std, dtype=np.float64)
    sstd = np.asarray(syx_std, dtype=np.float64)

    scale_k = float(np.nanmedian(kstd))
    scale_s = float(np.nanmedian(sstd))

    if not np.isfinite(scale_k) or scale_k <= 0:
        scale_k = 1e-12
    if not np.isfinite(scale_s) or scale_s <= 0:
        scale_s = 1e-12

    uncert = np.sqrt((kstd / scale_k) ** 2 + (sstd / scale_s) ** 2)
    return uncert, scale_k, scale_s


def _ref_summarize(preds_real, mode=_RefConfig.KXX_INV_MODE):
    """The body of StandaloneEnsemblePredictor.predict after the forward pass."""
    Kxx_models = preds_real[:, :, 0]
    Syx_models = preds_real[:, :, 1]

    if mode == "clip_positive":
        Kxx_inv_models = 1.0 / np.clip(Kxx_models, _RefConfig.KXX_INV_EPS, None)
    elif mode == "add_eps":
        Kxx_inv_models = 1.0 / (Kxx_models + _RefConfig.KXX_INV_EPS)
    else:
        raise ValueError(f"Unknown KXX_INV_MODE: {mode}")

    return {
        "Kxx_mean": Kxx_models.mean(axis=0),
        "Kxx_std": Kxx_models.std(axis=0),
        "Syx_mean": Syx_models.mean(axis=0),
        "Syx_std": Syx_models.std(axis=0),
        "Kxx_inv_mean": Kxx_inv_models.mean(axis=0),
        "Kxx_inv_std": Kxx_inv_models.std(axis=0),
        "n_models": np.full(Kxx_models.shape[1], Kxx_models.shape[0], dtype=int),
    }


def _ref_rank(df_valid):
    """The body of main() from Step 3 onward, given the prediction columns."""
    df_valid = df_valid.copy()

    finite_pred = (
        np.isfinite(df_valid["Kxx_mean"].to_numpy())
        & np.isfinite(df_valid["Syx_mean"].to_numpy())
        & np.isfinite(df_valid["Kxx_inv_std"].to_numpy())
        & np.isfinite(df_valid["Syx_std"].to_numpy())
    )
    if _RefConfig.REQUIRE_POSITIVE_KXX_MEAN:
        finite_pred &= df_valid["Kxx_mean"].to_numpy(dtype=np.float64) > 0

    if int((~finite_pred).sum()) > 0:
        df_valid = df_valid.loc[finite_pred].reset_index(drop=True)

    uncertainty, scale_k, scale_s = _ref_compute_uncertainty(
        df_valid["Kxx_inv_std"].to_numpy(dtype=np.float64),
        df_valid["Syx_std"].to_numpy(dtype=np.float64),
    )

    df_valid["uncertainty"] = uncertainty
    df_valid["uncert_scale_Kxx_inv_std_median"] = scale_k
    df_valid["uncert_scale_Syx_std_median"] = scale_s

    df_valid["Kxx_inv_from_mean"] = 1.0 / np.clip(
        df_valid["Kxx_mean"].to_numpy(dtype=np.float64), 1e-12, None
    )
    df_valid["Syx_over_Kxx"] = df_valid["Syx_mean"] / np.clip(df_valid["Kxx_mean"], 1e-12, None)

    obj0 = df_valid["Kxx_inv_from_mean"].to_numpy(dtype=np.float64)
    obj1 = df_valid["Syx_mean"].to_numpy(dtype=np.float64)
    obj2 = df_valid["uncertainty"].to_numpy(dtype=np.float64)

    pf_mask = _ref_pareto_mask_3d_fast_max(obj0, obj1, obj2)
    df_valid["is_pareto_3d"] = pf_mask
    df_pareto = df_valid.loc[pf_mask].copy()

    if _RefConfig.REQUIRE_POSITIVE_SYX_FOR_PERFORMANCE:
        df_perf_pool = df_pareto.loc[df_pareto["Syx_mean"] > 0].copy()
    else:
        df_perf_pool = df_pareto.copy()

    if len(df_perf_pool) == 0:
        df_perf_pool = df_pareto.copy()

    df_perf_sorted = df_perf_pool.sort_values(by="Syx_over_Kxx", ascending=False)
    top5_perf_idx = _ref_select_diverse_top_k(
        df_perf_sorted,
        comp_cols=ELEMENTS,
        k=_RefConfig.TOP_K,
        dist_th=_RefConfig.DIVERSITY_DIST_TH,
    )

    exclude_perf = set(top5_perf_idx)
    df_uncert_sorted = df_pareto.drop(index=list(exclude_perf), errors="ignore").sort_values(
        by="uncertainty",
        ascending=False,
    )
    top5_explore_idx = _ref_select_diverse_top_k(
        df_uncert_sorted,
        comp_cols=ELEMENTS,
        k=_RefConfig.TOP_K,
        dist_th=_RefConfig.DIVERSITY_DIST_TH,
        exclude_indices=exclude_perf,
    )

    return df_valid, top5_perf_idx, top5_explore_idx, scale_k, scale_s


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

#: (grid_step, limit_per_case). Coarse enough to run in seconds, fine enough
#: that every branch of the sampler is taken: the 0.01 case is the production
#: grid, and the 0.05 case forces partitions small enough that the Mn and Co
#: bound rejections fire often.
GRID_CASES = [(0.01, 3), (0.02, 8), (0.05, 20)]


def _cfg(**overrides) -> SelectionConfig:
    cfg = SelectionConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _fake_predictions(n, n_models=25, seed=7, pathological=True):
    """Per-member predictions shaped like the ensemble's, (n_models, n, 2).

    Deliberately includes non-positive and non-finite kappa. Those rows are
    what the `add_eps` inverse and the positivity filter exist for, and a port
    that quietly reorders or drops them differently would otherwise pass.
    """
    rng = np.random.default_rng(seed)
    kxx = rng.gamma(shape=4.0, scale=2.0, size=(n_models, n))
    syx = rng.normal(loc=1.5, scale=1.2, size=(n_models, n))
    if pathological and n >= 10:
        kxx[:, 0] = -0.5                     # every member negative
        kxx[0, 1] = -1e-9                    # one member barely negative
        kxx[:, 2] = np.nan
        syx[:, 3] = -np.abs(syx[:, 3])       # no positive S_ANE
        kxx[:, 4] = kxx[:, 5]                # exact ties in the objectives
        syx[:, 4] = syx[:, 5]
    return np.stack([kxx, syx], axis=2)


# ---------------------------------------------------------------------------
# enumeration and filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step,limit", GRID_CASES)
def test_generation_matches_reference(step, limit):
    ours = generate_candidates(_cfg(grid_step=step, limit_per_case=limit))
    ref = _ref_generate_grouped_grid(step=step, limit_per_case=limit, seed=42)

    assert list(ours.columns) == list(ref.columns)
    assert len(ours) == len(ref)
    assert np.array_equal(ours.to_numpy(), ref.to_numpy())


def test_partition_sampler_matches_reference():
    for total, k in [(1, 1), (5, 1), (5, 5), (3, 5), (100, 3), (7, 2), (0, 1)]:
        ours = sample_positive_partition(total, k, np.random.default_rng(11))
        ref = _ref_sample_positive_partition(total, k, np.random.default_rng(11))
        assert ours == ref, (total, k)


@pytest.mark.parametrize("step,limit", GRID_CASES)
def test_feasibility_matches_reference(step, limit):
    cfg = _cfg(grid_step=step, limit_per_case=limit)
    df = generate_candidates(cfg)
    assert np.array_equal(feasibility_mask(df, cfg), _ref_feasibility_mask_only(df))


def test_delta_h_mix_matches_reference():
    from ane.physics import delta_h_mix

    df = generate_candidates(_cfg(grid_step=0.02, limit_per_case=8))
    comp = df.to_numpy(dtype=np.float64)
    assert np.array_equal(delta_h_mix(comp), _ref_calc_delta_h_mix(comp))


def test_element_order_matches_reference():
    """A silent reordering here would invalidate every saved scaler."""
    assert list(_ANE_ELEMENTS) == ELEMENTS


# ---------------------------------------------------------------------------
# featurization as the predictor calls it
# ---------------------------------------------------------------------------


def test_featurize_matches_notebook_ilr_concat():
    """The predictor's featurization, against the notebook's own copy of it.

    Independent of `test_features_parity.py`: that file compares against a
    transcription from ensemble_v3.ipynb, this one against Pareto_v2.ipynb, and
    the point of the exercise is that four notebooks each carried their own
    copy. Constants are hard-coded rather than imported from `ane.elements` --
    importing them would make the two agree by construction, which is how a
    truncated R_GAS survived in `elements.py` unnoticed.

    Note that this comparison is in float32, where that truncation is invisible;
    the float64 check in `test_features_parity.test_descriptors_match_reference`
    is the one that catches it.
    """
    import math

    R_GAS = 8.31446261815324
    ELEMENT_PROPS = {
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

    def multiplicative_replacement(A, delta=1e-3):
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

    def helmert_submatrix(D):
        H = np.zeros((D, D - 1), dtype=np.float64)
        for i in range(1, D):
            a = 1.0 / math.sqrt(i * (i + 1))
            H[:i, i - 1] = a
            H[i, i - 1] = -i * a
        return H

    class ILR:
        def __init__(self, D):
            self.D = D
            self.H = helmert_submatrix(D)

        def transform(self, X):
            X = multiplicative_replacement(X)
            logx = np.log(closure(X))
            clr = logx - logx.mean(axis=1, keepdims=True)
            return (clr @ self.H).astype(np.float64)

    def batch_atomic_properties(C):
        X = closure(np.asarray(C, dtype=np.float64))
        radius = np.array([ELEMENT_PROPS[e]["radius"] for e in ELEMENTS], dtype=np.float64)
        vec = np.array([ELEMENT_PROPS[e]["vec"] for e in ELEMENTS], dtype=np.float64)
        weight = np.array([ELEMENT_PROPS[e]["weight"] for e in ELEMENTS], dtype=np.float64)
        en = np.array([ELEMENT_PROPS[e]["en"] for e in ELEMENTS], dtype=np.float64)

        r_avg = X @ radius
        atomic_size_diff = np.sqrt(np.sum(X * (1.0 - radius[None, :] / r_avg[:, None]) ** 2, axis=1))
        vec_avg = X @ vec
        vec_std = np.sqrt(np.sum(X * (vec[None, :] - vec_avg[:, None]) ** 2, axis=1))
        weight_avg = X @ weight
        X_safe = np.clip(X, 1e-12, None)
        smix = -R_GAS * np.sum(X_safe * np.log(X_safe), axis=1)
        en_avg = X @ en
        en_std = np.sqrt(np.sum(X * (en[None, :] - en_avg[:, None]) ** 2, axis=1))
        return np.column_stack(
            [r_avg, atomic_size_diff, vec_avg, vec_std, weight_avg, smix, en_avg, en_std]
        ).astype(np.float64)

    def ilr_concat(X_comp8):
        ilr = ILR(D=8).transform(X_comp8)
        calc = batch_atomic_properties(X_comp8)
        return np.hstack([ilr, calc]).astype(np.float32)

    comp = generate_candidates(_cfg(grid_step=0.02, limit_per_case=8)).to_numpy(np.float64)
    assert np.array_equal(featurize(comp), ilr_concat(comp))


# ---------------------------------------------------------------------------
# prediction reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["add_eps", "clip_positive"])
def test_summarize_predictions_matches_reference(mode):
    preds = _fake_predictions(300)
    ours = summarize_predictions(preds, _cfg(kappa_inverse_mode=mode))
    ref = _ref_summarize(preds, mode=mode)

    assert set(ours) == set(ref)
    for key in ref:
        assert np.array_equal(ours[key], ref[key], equal_nan=True), key


def test_inverse_modes_actually_differ():
    """Confirms the mode is a real choice and not a dead configuration knob."""
    preds = _fake_predictions(300)
    a = summarize_predictions(preds, _cfg(kappa_inverse_mode="add_eps"))
    b = summarize_predictions(preds, _cfg(kappa_inverse_mode="clip_positive"))
    assert not np.array_equal(a["Kxx_inv_std"], b["Kxx_inv_std"], equal_nan=True)


def test_summarize_rejects_unknown_mode():
    with pytest.raises(ValueError):
        summarize_predictions(_fake_predictions(20), _cfg(kappa_inverse_mode="nope"))


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pareto_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 800
    # coarse rounding forces the tie handling to be exercised, which is where
    # the two group-wise branches of the sweep live
    obj0 = np.round(rng.normal(size=n), 1)
    obj1 = np.round(rng.normal(size=n), 1)
    obj2 = np.round(rng.normal(size=n), 1)
    obj0[rng.integers(0, n, 20)] = np.nan
    obj1[rng.integers(0, n, 10)] = np.inf

    assert np.array_equal(
        pareto_mask_3d_max(obj0, obj1, obj2),
        _ref_pareto_mask_3d_fast_max(obj0, obj1, obj2),
    )


def test_pareto_matches_brute_force():
    """Independent check that the reference itself is right, not just copied."""
    rng = np.random.default_rng(5)
    n = 300
    o = np.round(rng.normal(size=(n, 3)), 1)

    brute = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(o[j] >= o[i]) and np.any(o[j] > o[i]):
                brute[i] = False
                break

    fast = pareto_mask_3d_max(o[:, 0], o[:, 1], o[:, 2])
    assert np.array_equal(fast, brute)


def test_pareto_all_nan_returns_empty():
    nan = np.full(10, np.nan)
    assert not pareto_mask_3d_max(nan, nan, nan).any()


def test_pareto_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        pareto_mask_3d_max(np.zeros(3), np.zeros(4), np.zeros(3))


# ---------------------------------------------------------------------------
# uncertainty and ranking
# ---------------------------------------------------------------------------


def test_uncertainty_matches_reference():
    rng = np.random.default_rng(3)
    kstd = np.abs(rng.normal(scale=1e-3, size=500))
    sstd = np.abs(rng.normal(scale=10.0, size=500))
    kstd[:5] = np.nan

    ours = compute_uncertainty(kstd, sstd)
    ref = _ref_compute_uncertainty(kstd, sstd)

    assert np.array_equal(ours[0], ref[0], equal_nan=True)
    assert ours[1] == ref[1] and ours[2] == ref[2]


def test_uncertainty_degenerate_scale_matches_reference():
    zeros = np.zeros(20)
    ours = compute_uncertainty(zeros, zeros)
    ref = _ref_compute_uncertainty(zeros, zeros)
    assert np.array_equal(ours[0], ref[0]) and ours[1:] == ref[1:]


def test_diverse_top_k_matches_reference():
    cfg = _cfg(grid_step=0.02, limit_per_case=8)
    df = generate_candidates(cfg)
    df = df.loc[feasibility_mask(df, cfg)].reset_index(drop=True)
    rng = np.random.default_rng(2)
    df["score"] = rng.normal(size=len(df))
    df_sorted = df.sort_values(by="score", ascending=False)

    for dist_th in (0.0, 0.10, 0.35, 5.0):     # 5.0 forces the fallback path
        ours = select_diverse_top_k(df_sorted, ELEMENTS, k=5, dist_th=dist_th)
        ref = _ref_select_diverse_top_k(df_sorted, ELEMENTS, k=5, dist_th=dist_th)
        assert ours == ref, dist_th


def test_diverse_top_k_exclusion_matches_reference():
    cfg = _cfg(grid_step=0.02, limit_per_case=8)
    df = generate_candidates(cfg)
    df = df.loc[feasibility_mask(df, cfg)].reset_index(drop=True)
    rng = np.random.default_rng(4)
    df["score"] = rng.normal(size=len(df))
    df_sorted = df.sort_values(by="score", ascending=False)

    exclude = set(int(i) for i in df_sorted.index[:7])
    ours = select_diverse_top_k(df_sorted, ELEMENTS, 5, 0.10, exclude_indices=exclude)
    ref = _ref_select_diverse_top_k(df_sorted, ELEMENTS, 5, 0.10, exclude_indices=exclude)
    assert ours == ref
    assert not (set(ours) & exclude)


# ---------------------------------------------------------------------------
# the ranking stage end to end
# ---------------------------------------------------------------------------


def test_rank_candidates_matches_reference():
    cfg = _cfg(grid_step=0.02, limit_per_case=8)
    df = generate_candidates(cfg)
    df = df.loc[feasibility_mask(df, cfg)].drop_duplicates().reset_index(drop=True)
    assert len(df) > 50, "too few candidates for the comparison to mean anything"

    preds = _fake_predictions(len(df))
    for key, value in _ref_summarize(preds).items():
        df[key] = value

    ours, top_perf, top_explore, scales = rank_candidates(df, cfg, verbose=False)
    ref_df, ref_perf, ref_explore, ref_k, ref_s = _ref_rank(df)

    assert top_perf == ref_perf
    assert top_explore == ref_explore
    assert scales["scale_k"] == ref_k and scales["scale_s"] == ref_s
    assert np.array_equal(
        ours["is_pareto_3d"].to_numpy(), ref_df["is_pareto_3d"].to_numpy()
    )
    for col in ("uncertainty", "Kxx_inv_from_mean", "Syx_over_Kxx", "Syx_mean", "Kxx_mean"):
        assert np.array_equal(
            ours[col].to_numpy(), ref_df[col].to_numpy(), equal_nan=True
        ), col


def test_selected_batches_are_disjoint_and_full():
    cfg = _cfg(grid_step=0.02, limit_per_case=8)
    df = generate_candidates(cfg)
    df = df.loc[feasibility_mask(df, cfg)].drop_duplicates().reset_index(drop=True)
    for key, value in _ref_summarize(_fake_predictions(len(df))).items():
        df[key] = value

    _, top_perf, top_explore, _ = rank_candidates(df, cfg, verbose=False)
    assert not set(top_perf) & set(top_explore)
    assert len(top_perf) == cfg.top_k and len(top_explore) == cfg.top_k


def test_config_defaults_match_the_committed_yaml():
    """The defaults must be the file a cycle is actually run with."""
    from pathlib import Path

    from ane.config import PipelineConfig

    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    from_file = PipelineConfig.from_yaml(yaml_path)
    assert from_file.as_dict() == PipelineConfig().as_dict()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
