"""Numerical parity between `ane.augment` and the original notebook code.

The reference block is transcribed from GAN_v2.ipynb. Module-level constants
that the notebook's functions read directly are passed as arguments here so
that both sides can be driven identically; nothing else is changed.

What is checked. Every deterministic step between the generator's output and
the emitted CSV: the invertible ILR and its inverse, the grid snapping, the
composition keys, the physical and property filters, the k-center selection,
the rare-element weighting, and the surrogate threshold and mask. Also that
the two ILR transforms in this package are genuinely different and neither has
been quietly swapped for the other.

What is not. The adversarial training. Two networks, ten thousand steps, and a
result that is not reproducible across machines even with the seeds fixed.
What can be said is that everything deciding which of the generator's samples
survive is compared exactly, and that is where the augmented dataset is
actually determined -- an unfiltered WGAN-GP on fifty rows produces very
little that passes.

    python -m pytest tests/test_augment_parity.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ane.augment import (
    N_ILR,
    composition_keys,
    ilr_inverse,
    ilr_transform,
    kcenter_select,
    physical_mask,
    property_mask,
    quantize_on_grid,
    rare_elements,
    sample_probabilities,
    surrogate_consistency_mask,
)
from ane.config import GANConfig

# ---------------------------------------------------------------------------
# reference implementation, from GAN_v2.ipynb
# ---------------------------------------------------------------------------

group_A = ["Fe", "Co", "Mn"]
group_B = ["Ga", "Al", "Si", "Ge", "Pt"]
all_elements = group_A + group_B
property_names = ["kxx", "S_ANE"]

binary_enthalpy = {
    ("Fe", "Co"): -0.847634, ("Fe", "Mn"): 0.437463, ("Fe", "Ga"): -17.803602,
    ("Fe", "Al"): -32.115644, ("Fe", "Si"): -26.345766, ("Fe", "Ge"): -8.939757,
    ("Fe", "Pt"): -19.335633, ("Co", "Mn"): -7.757938, ("Co", "Ga"): -31.302334,
    ("Co", "Al"): -43.255284, ("Co", "Si"): -31.191669, ("Co", "Ge"): -17.134425,
    ("Co", "Pt"): -10.682789, ("Mn", "Ga"): -34.099536, ("Mn", "Al"): -43.499266,
    ("Mn", "Si"): -41.506930, ("Mn", "Ge"): -31.820084, ("Mn", "Pt"): -41.941500,
    ("Ga", "Al"): 1.415135, ("Ga", "Si"): 16.499490, ("Ga", "Ge"): 8.080138,
    ("Ga", "Pt"): -73.039917, ("Al", "Si"): 13.443439, ("Al", "Ge"): 9.499498,
    ("Al", "Pt"): -82.880335, ("Si", "Ge"): 32.998448, ("Si", "Pt"): -55.947653,
    ("Ge", "Pt"): -42.996605,
}


class _RefGANConfig:
    """Module-level constants of GAN_v2.ipynb, as the surviving copy has them."""

    INPUT_DIM = 9
    LATENT_DIM = 8
    BATCH_SIZE = 16
    EPOCHS = 10000
    N_CRITIC = 4
    GP_WEIGHT = 10.0
    LR = 1e-4
    BETA1, BETA2 = 0.0, 0.9

    LOG_INTERVAL = 100
    EARLYSTOP_PATIENCE = 50
    NOISE_STD0 = 0.01

    KFOLD_SPLITS = 5
    PROXY_HIDDEN = (16, 16)
    PROXY_MAXITER = 1000

    FAKE_POOL_MIN = 100
    FAKE_SELECT_K = 20
    FAKE_DRAW_BATCH = 2000
    FAKE_MAX_ROUNDS = 60
    KCC_WEIGHT_PROP = 1.0

    TARGET_UNIQUE_SAMPLES = 800
    FINAL_SIZES = [100, 200, 300, 400, 500]
    FINAL_DRAW_PER_ROUND = 10000
    FINAL_MAX_ROUNDS = 100

    RARE_MIN_COUNT = 2
    RARE_MAX_RATIO = 0.25
    WEIGHT_BOOST_MAX = 5.0
    WEIGHT_CAP = 9.0

    CO_MAX_FILTER = 0.60
    SNAP_STEP = 0.01
    SNAP_DIST_MAX = 0.015

    SURROGATE_THRESHOLD_Q = 0.99
    SURROGATE_THRESHOLD_SAFETY = 2.0
    SURROGATE_WARMUP_EPOCH = 1000

    STOICH_MIN, STOICH_MAX = 2.2, 3.8
    MN_MAX = 0.25
    HMIX_MAX = 0.0


def _ref_helmert_submatrix(D):
    H = np.zeros((D - 1, D), dtype=np.float64)
    for i in range(1, D):
        scale = np.sqrt(i * (i + 1))
        H[i - 1, :i] = 1.0 / scale
        H[i - 1, i] = -i / scale
    return H


def _ref_ilr_transform(comp):
    comp = np.asarray(comp, dtype=np.float64)
    comp = np.clip(comp, 1e-12, None)
    comp = comp / comp.sum(axis=1, keepdims=True)
    H = _ref_helmert_submatrix(comp.shape[1])
    return np.log(comp) @ H.T


def _ref_ilr_inverse_to_unit(ilr_comp):
    ilr_comp = np.asarray(ilr_comp, dtype=np.float64)
    D = ilr_comp.shape[1] + 1
    H = _ref_helmert_submatrix(D)
    X = np.exp(ilr_comp @ H)
    X = np.maximum(X, 1e-12)
    return X / X.sum(axis=1, keepdims=True)


def _ref_quantize_on_grid(comp, step=0.01):
    X = np.asarray(comp, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    n, d = X.shape
    RES = int(round(1.0 / float(step)))
    out_int = np.zeros((n, d), dtype=int)
    for i in range(n):
        row = np.maximum(0.0, X[i].copy())
        row = row / (row.sum() + 1e-12)
        raw = row * RES
        base = np.floor(raw).astype(int)
        residual = raw - base
        need = RES - int(base.sum())
        if need > 0:
            order = np.argsort(-residual)
            base[order[:need]] += 1
        elif need < 0:
            order = np.argsort(residual)
            for idx in order[: (-need)]:
                if base[idx] > 0:
                    base[idx] -= 1
        out_int[i] = base
    return out_int.astype(np.float64) / float(RES)


def _ref_composition_keys_from_array(comp, step=0.01):
    comp_q = _ref_quantize_on_grid(comp, step=step)
    RES = int(round(1.0 / step))
    keys = (comp_q * RES + 1e-9).astype(int)
    return {tuple(row) for row in keys}


def _ref_calc_delta_h_mix(compositions, elements, enthalpy_dict):
    comps = np.asarray(compositions, dtype=np.float64)
    out = np.zeros(comps.shape[0], dtype=np.float64)
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            h_ij = enthalpy_dict.get(
                (elements[i], elements[j]),
                enthalpy_dict.get((elements[j], elements[i]), 0.0),
            )
            out += 4.0 * h_ij * comps[:, i] * comps[:, j]
    return out


def _ref_physical_mask_from_raw(ilr7, props2):
    comp8 = _ref_ilr_inverse_to_unit(ilr7)
    df = pd.DataFrame(comp8, columns=all_elements)
    sumA = df[group_A].sum(axis=1).values
    sumB = df[group_B].sum(axis=1).values
    eps = 1e-12
    ratio = sumA / np.maximum(sumB, eps)
    mask = (
        (sumB > eps)
        & (ratio >= _RefGANConfig.STOICH_MIN)
        & (ratio <= _RefGANConfig.STOICH_MAX)
        & (df["Mn"].values <= _RefGANConfig.MN_MAX)
    )
    hmix = _ref_calc_delta_h_mix(df[all_elements].values, all_elements, binary_enthalpy)
    mask &= hmix <= _RefGANConfig.HMIX_MAX
    return mask, comp8


def _ref_property_mask(props2, prop_tr_raw, margin=0.15):
    kxx, sane = props2[:, 0], props2[:, 1]
    finite = np.isfinite(kxx) & np.isfinite(sane)
    positive_kxx = kxx > 0.0
    lo = np.quantile(prop_tr_raw, 0.01, axis=0)
    hi = np.quantile(prop_tr_raw, 0.99, axis=0)
    width = hi - lo
    within = (
        (kxx >= lo[0] - margin * width[0])
        & (kxx <= hi[0] + margin * width[0])
        & (sane >= lo[1] - margin * width[1])
        & (sane <= hi[1] + margin * width[1])
    )
    return finite & positive_kxx & within


def _ref_kcenter_select(X, n_select):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n_select >= n:
        return np.arange(n, dtype=int)
    center = X.mean(axis=0)
    d2 = np.sum((X - center) ** 2, axis=1)
    sel = [int(np.argmax(d2))]
    min_d2 = np.sum((X - X[sel[0]]) ** 2, axis=1)
    for _ in range(1, n_select):
        cand = int(np.argmax(min_d2))
        sel.append(cand)
        min_d2 = np.minimum(min_d2, np.sum((X - X[cand]) ** 2, axis=1))
    return np.array(sel, dtype=int)


def _ref_auto_select_rare_elements(comp_tr, elements, eps=1e-12, min_count=2, max_ratio=0.25):
    N, _ = comp_tr.shape
    counts = {el: int(np.sum(comp_tr[:, j] > eps)) for j, el in enumerate(elements)}
    rare = [el for el, c in counts.items() if min_count <= c <= int(np.floor(max_ratio * N))]
    return rare, counts


def _ref_build_sample_probs(comp_tr, elements, rare_els, counts, eps=1e-12, boost_max=5.0, cap=9.0):
    N = comp_tr.shape[0]
    w = np.ones(N, dtype=np.float64)
    for el in rare_els:
        j = elements.index(el)
        boost = min(boost_max, np.sqrt(N / max(counts[el], 1)))
        w[comp_tr[:, j] > eps] *= boost
    w = np.minimum(w, cap)
    return w / w.sum()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> GANConfig:
    cfg = GANConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _compositions(n=400, seed=0, sparse=True) -> np.ndarray:
    """Compositions with the sparsity the pipeline actually sees."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        k = int(rng.integers(3, 6)) if sparse else 8
        idx = rng.choice(8, size=k, replace=False)
        x = np.zeros(8)
        x[idx] = rng.dirichlet(np.ones(k))
        rows.append(x)
    return np.array(rows)


def _realistic_alloys(n=60, seed=1) -> np.ndarray:
    """Compositions inside the stoichiometry window, so the filters see work."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        x = np.zeros(8)
        a_total = rng.uniform(2.2, 3.8)
        a_total = a_total / (1.0 + a_total)
        n_a = int(rng.integers(1, 4))
        n_b = int(rng.integers(1, 4))
        a_idx = rng.choice(3, size=n_a, replace=False)
        b_idx = rng.choice(5, size=n_b, replace=False) + 3
        x[a_idx] = rng.dirichlet(np.ones(n_a)) * a_total
        x[b_idx] = rng.dirichlet(np.ones(n_b)) * (1.0 - a_total)
        rows.append(x)
    return np.array(rows)


def _properties(n, seed=2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(3.0, 18.0, n), rng.normal(1.0, 0.5, n)])


class _StubModel:
    """A surrogate that predicts a fixed offset from the truth."""

    def __init__(self, offset):
        self.offset = np.asarray(offset, dtype=np.float64)

    def __call__(self, X, training=False):
        class _Out:
            def __init__(self, value):
                self.value = value

            def numpy(self):
                return self.value

        return _Out(np.tile(self.offset, (len(X), 1)))


class _IdentityScaler:
    def __init__(self, n_features_in=15):
        self.n_features_in_ = n_features_in

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------------------
# the two ILR transforms are different, on purpose
# ---------------------------------------------------------------------------


def test_generator_ilr_matches_reference():
    comp = _compositions(400, seed=3)
    assert np.array_equal(ilr_transform(comp), _ref_ilr_transform(comp))


def test_generator_ilr_inverse_matches_reference():
    coords = ilr_transform(_compositions(400, seed=4))
    assert np.array_equal(ilr_inverse(coords), _ref_ilr_inverse_to_unit(coords))


def test_generator_ilr_round_trips():
    """The property the surrogate's transform does not have and does not need."""
    comp = _compositions(300, seed=5)
    comp = comp / comp.sum(axis=1, keepdims=True)
    recovered = ilr_inverse(ilr_transform(comp))
    assert np.allclose(recovered, comp, atol=1e-9)


def test_generator_ilr_is_not_the_surrogate_ilr():
    """Two transforms, two jobs. Swapping them would be silent and wrong.

    They differ only in how an absent component is handled -- clipped to 1e-12
    here, replaced by 1e-3 there -- but every real composition has absent
    components, so the difference is on every row.
    """
    from ane.features import ILR

    comp = _compositions(200, seed=6)
    ours = ilr_transform(comp)
    surrogate = ILR().transform(comp)

    assert ours.shape == surrogate.shape
    assert not np.allclose(ours, surrogate)
    assert np.abs(ours - surrogate).max() > 1.0


def test_dense_compositions_agree_between_the_two_ilrs():
    """With no zeros the zero handling never fires and they must coincide,
    which is what shows the difference is the zero rule and not the basis."""
    from ane.features import ILR

    comp = _compositions(100, seed=7, sparse=False)
    assert np.allclose(ilr_transform(comp), ILR().transform(comp))


# ---------------------------------------------------------------------------
# grid snapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", [0.005, 0.01, 0.02])
def test_quantize_matches_reference(step):
    comp = _compositions(300, seed=8)
    assert np.array_equal(
        quantize_on_grid(comp, step=step), _ref_quantize_on_grid(comp, step=step)
    )


@pytest.mark.parametrize("step", [0.005, 0.01])
def test_quantized_compositions_sum_to_one_exactly(step):
    """Independent-rounding would not, and the error would ride through the
    whole pipeline as a composition that does not close."""
    q = quantize_on_grid(_compositions(300, seed=9), step=step)
    resolution = int(round(1.0 / step))
    units = np.rint(q * resolution).astype(int)
    assert np.all(units.sum(axis=1) == resolution)
    assert np.all(units >= 0)


def test_quantize_handles_a_single_row():
    row = np.array([0.751, 0.0, 0.0, 0.249, 0.0, 0.0, 0.0, 0.0])
    q = quantize_on_grid(row, step=0.01)
    assert q.shape == (1, 8)
    assert np.array_equal(q, _ref_quantize_on_grid(row, step=0.01))


@pytest.mark.parametrize("step", [0.005, 0.01])
def test_composition_keys_match_reference(step):
    comp = _compositions(200, seed=10)
    assert composition_keys(comp, step=step) == _ref_composition_keys_from_array(comp, step=step)


def test_composition_keys_collapse_near_identical_compositions():
    """Keys exist so that a re-proposal is caught despite float arithmetic."""
    base = _realistic_alloys(20, seed=11)
    jittered = base + np.random.default_rng(12).normal(0, 1e-9, base.shape)
    assert composition_keys(base, 0.01) == composition_keys(jittered, 0.01)


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_physical_mask_matches_reference():
    comp = _realistic_alloys(300, seed=13)
    ilr7 = ilr_transform(comp)
    props = _properties(len(comp), seed=14)

    ours, our_comp = physical_mask(ilr7, _cfg())
    ref, ref_comp = _ref_physical_mask_from_raw(ilr7, props)

    assert np.array_equal(ours, ref)
    assert np.allclose(our_comp, ref_comp)


def test_physical_mask_actually_rejects():
    comp = _compositions(400, seed=15)     # unconstrained: most should fail
    mask, _ = physical_mask(ilr_transform(comp), _cfg())
    assert 0 < mask.sum() < len(mask)


def test_property_mask_matches_reference_where_s_ane_is_physical():
    """Parity with the campaign filter holds on samples the target permits.

    The campaign filter is reproduced by `_ref_property_mask`. The current
    filter adds one term it did not have -- |S_ANE| >= 0 -- so the two agree
    exactly on generated samples with a non-negative |S_ANE| and differ only
    where the campaign filter admitted a negative magnitude. That divergence is
    the subject of the next test.
    """
    measured = _properties(60, seed=16)
    generated = _properties(400, seed=17)
    generated[:5, 0] = -1.0                      # non-positive kappa
    generated[5:10, 0] = np.nan
    generated[10:15, 1] = 500.0                  # far outside the measured range

    ours = property_mask(generated, measured, _cfg())
    ref = _ref_property_mask(generated, measured, margin=0.15)

    physical = generated[:, 1] >= 0.0
    assert np.array_equal(ours[physical], ref[physical])
    assert not ours[:15].any()


def test_property_mask_rejects_negative_s_ane():
    """|S_ANE| is a magnitude, so a negative generated value is not a property.

    The percentile window does not exclude these on its own: widening the
    measured 1-99% span by the margin puts its lower edge below zero, so the
    campaign filter admitted them. Whether any reached a historical training set
    cannot be established -- the generated CSVs were not retained -- so this is
    a forward-looking correction, and the test pins it.
    """
    measured = _properties(60, seed=16)
    lo = np.quantile(measured, 0.01, axis=0)
    hi = np.quantile(measured, 0.99, axis=0)
    width = hi - lo
    assert lo[1] - 0.15 * width[1] < 0.0, "the window must reach below zero for this to matter"

    generated = _properties(20, seed=17)
    generated[:, 0] = float(np.median(measured[:, 0]))
    generated[:5, 1] = [-0.5, -0.2, -1e-6, 0.0, 0.5]

    ours = property_mask(generated, measured, _cfg())
    ref = _ref_property_mask(generated, measured, margin=0.15)

    assert not ours[:3].any(), "negative |S_ANE| must be rejected"
    assert ours[3] and ours[4], "zero and positive |S_ANE| must survive"
    assert ref[:3].any(), "the campaign filter did admit them; that is the point"


def test_property_mask_allows_modest_extrapolation():
    """The search is looking for something better than anything measured, so
    the range check must not be a hard box around the training data."""
    measured = _properties(60, seed=18)
    hi = np.quantile(measured, 0.99, axis=0)
    lo = np.quantile(measured, 0.01, axis=0)
    just_beyond = np.array([[hi[0] + 0.10 * (hi[0] - lo[0]), measured[:, 1].mean()]])
    far_beyond = np.array([[hi[0] + 0.50 * (hi[0] - lo[0]), measured[:, 1].mean()]])

    cfg = _cfg()
    assert property_mask(just_beyond, measured, cfg)[0]
    assert not property_mask(far_beyond, measured, cfg)[0]


# ---------------------------------------------------------------------------
# diversity selection and weighting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_select", [1, 5, 20, 50])
def test_kcenter_matches_reference(n_select):
    X = np.random.default_rng(19).normal(size=(200, 9))
    assert np.array_equal(kcenter_select(X, n_select), _ref_kcenter_select(X, n_select))


def test_kcenter_returns_everything_when_asked_for_too_much():
    X = np.random.default_rng(20).normal(size=(10, 9))
    assert np.array_equal(kcenter_select(X, 25), np.arange(10))
    assert np.array_equal(kcenter_select(X, 25), _ref_kcenter_select(X, 25))


def test_kcenter_is_deterministic_and_spreads_out():
    """Compared against random subsampling, which is what it replaces."""
    rng = np.random.default_rng(21)
    X = np.vstack([rng.normal(0, 0.1, (400, 2)), rng.normal(5, 0.1, (20, 2))])

    chosen = kcenter_select(X, 20)
    assert np.array_equal(chosen, kcenter_select(X, 20))

    n_far = int((X[chosen][:, 0] > 2.5).sum())
    assert n_far >= 5, "k-center should reach the sparse cluster, unlike a random draw"


def test_rare_elements_match_reference():
    comp = _realistic_alloys(60, seed=22)
    cfg = _cfg()
    ours_rare, ours_counts = rare_elements(comp, cfg)
    ref_rare, ref_counts = _ref_auto_select_rare_elements(
        comp, all_elements,
        min_count=_RefGANConfig.RARE_MIN_COUNT,
        max_ratio=_RefGANConfig.RARE_MAX_RATIO,
    )
    assert ours_rare == ref_rare
    assert ours_counts == ref_counts


def test_sample_probabilities_match_reference():
    comp = _realistic_alloys(60, seed=23)
    cfg = _cfg()
    rare, counts = rare_elements(comp, cfg)

    ours = sample_probabilities(comp, rare, counts, cfg)
    ref = _ref_build_sample_probs(
        comp, all_elements, rare, counts,
        boost_max=_RefGANConfig.WEIGHT_BOOST_MAX,
        cap=_RefGANConfig.WEIGHT_CAP,
    )
    assert np.array_equal(ours, ref)
    assert np.isclose(ours.sum(), 1.0)


def test_sample_probabilities_boost_rows_holding_rare_elements():
    comp = np.zeros((40, 8))
    comp[:, 0] = 0.75                    # Fe everywhere
    comp[:, 3] = 0.25                    # Ga everywhere
    comp[:3, 3] = 0.20
    comp[:3, 7] = 0.05                   # Pt in three rows only

    cfg = _cfg()
    rare, counts = rare_elements(comp, cfg)
    assert "Pt" in rare

    probs = sample_probabilities(comp, rare, counts, cfg)
    assert probs[:3].min() > probs[3:].max()


def test_sample_probability_boost_is_capped():
    """Uncapped inverse-frequency weighting would let one row own the batch."""
    comp = np.zeros((100, 8))
    comp[:, 0] = 0.75
    comp[:, 3] = 0.25
    comp[0, 3] = 0.20
    comp[0, 7] = 0.05

    cfg = _cfg()
    rare, counts = rare_elements(comp, cfg)
    probs = sample_probabilities(comp, rare, counts, cfg)
    assert probs.max() / probs.min() <= cfg.weight_cap + 1e-9


# ---------------------------------------------------------------------------
# surrogate consistency
# ---------------------------------------------------------------------------


def test_surrogate_mask_keeps_agreement_and_drops_disagreement():
    comp = _realistic_alloys(50, seed=24)
    props = _properties(len(comp), seed=25)

    # a surrogate that always predicts the middle of the measured range: rows
    # whose generated properties sit near it agree, the rest do not
    stub = _StubModel([props[:, 0].mean(), props[:, 1].mean()])
    scaler = _IdentityScaler()
    thresholds = [3.0, 0.4]

    mask, predicted, diff = surrogate_consistency_mask(
        comp, props, stub, scaler, scaler, thresholds
    )
    assert np.array_equal(diff, np.abs(props - predicted))
    assert np.array_equal(
        mask, (diff[:, 0] <= thresholds[0]) & (diff[:, 1] <= thresholds[1])
    )
    assert 0 < mask.sum() < len(mask), "the filter should split the sample, not clear it"


def test_surrogate_mask_is_a_no_op_without_a_model():
    comp = _realistic_alloys(30, seed=26)
    props = _properties(len(comp), seed=27)

    mask, predicted, diff = surrogate_consistency_mask(comp, props, None, None, None, None)
    assert mask.all()
    assert np.isnan(predicted).all() and np.isnan(diff).all()


def test_surrogate_mask_does_not_relabel():
    """The filter vetoes; it never replaces the generated properties."""
    comp = _realistic_alloys(30, seed=28)
    props = _properties(len(comp), seed=29)
    before = props.copy()

    surrogate_consistency_mask(
        comp, props, _StubModel([9.0, 9.0]), _IdentityScaler(), _IdentityScaler(), [1e9, 1e9]
    )
    assert np.array_equal(props, before)


def test_surrogate_threshold_uses_safety_factor_without_validation_data(tmp_path):
    """In-sample residuals are optimistic; the factor is what compensates."""
    from ane.augment import surrogate_threshold

    comp = _realistic_alloys(60, seed=30)
    props = _properties(len(comp), seed=31)
    model, scaler = _StubModel([8.0, 1.0]), _IdentityScaler()

    strict = surrogate_threshold(
        comp, props, model, scaler, scaler, _cfg(surrogate_safety_factor=1.0), verbose=False
    )
    lenient = surrogate_threshold(
        comp, props, model, scaler, scaler, _cfg(surrogate_safety_factor=2.0), verbose=False
    )
    assert np.allclose(np.array(lenient), 2.0 * np.array(strict))


def test_surrogate_threshold_calibration_subset_is_seeded():
    from ane.augment import surrogate_threshold

    comp = _realistic_alloys(60, seed=32)
    props = _properties(len(comp), seed=33)
    model, scaler = _StubModel([8.0, 1.0]), _IdentityScaler()
    cfg = _cfg()

    a = surrogate_threshold(comp, props, model, scaler, scaler, cfg, verbose=False)
    b = surrogate_threshold(comp, props, model, scaler, scaler, cfg, verbose=False)
    assert a == b


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_gan_config_values_agreeing_with_the_notebook():
    cfg = _cfg()
    assert cfg.gradient_penalty_weight == _RefGANConfig.GP_WEIGHT
    assert cfg.learning_rate == _RefGANConfig.LR
    assert (cfg.adam_beta_1, cfg.adam_beta_2) == (_RefGANConfig.BETA1, _RefGANConfig.BETA2)
    assert cfg.instance_noise_std == _RefGANConfig.NOISE_STD0
    assert cfg.eval_interval == _RefGANConfig.LOG_INTERVAL
    assert cfg.early_stopping_patience == _RefGANConfig.EARLYSTOP_PATIENCE
    assert cfg.proxy_kfold_splits == _RefGANConfig.KFOLD_SPLITS
    assert cfg.fake_draw_batch == _RefGANConfig.FAKE_DRAW_BATCH
    assert cfg.fake_max_rounds == _RefGANConfig.FAKE_MAX_ROUNDS
    assert cfg.kcenter_property_weight == _RefGANConfig.KCC_WEIGHT_PROP
    assert cfg.target_unique_samples == _RefGANConfig.TARGET_UNIQUE_SAMPLES
    assert tuple(cfg.generated_sizes) == tuple(_RefGANConfig.FINAL_SIZES)
    assert cfg.final_draw_per_round == _RefGANConfig.FINAL_DRAW_PER_ROUND
    assert cfg.final_max_rounds == _RefGANConfig.FINAL_MAX_ROUNDS
    assert cfg.rare_min_count == _RefGANConfig.RARE_MIN_COUNT
    assert cfg.rare_max_ratio == _RefGANConfig.RARE_MAX_RATIO
    assert cfg.weight_boost_max == _RefGANConfig.WEIGHT_BOOST_MAX
    assert cfg.weight_cap == _RefGANConfig.WEIGHT_CAP
    assert cfg.co_max == _RefGANConfig.CO_MAX_FILTER
    assert cfg.mn_max == _RefGANConfig.MN_MAX
    assert cfg.snap_distance_max == _RefGANConfig.SNAP_DIST_MAX
    assert cfg.snap_step == _RefGANConfig.SNAP_STEP
    assert cfg.surrogate_warmup_epoch == _RefGANConfig.SURROGATE_WARMUP_EPOCH
    assert (cfg.stoichiometry_min, cfg.stoichiometry_max) == (
        _RefGANConfig.STOICH_MIN, _RefGANConfig.STOICH_MAX
    )
    assert cfg.h_mix_max == _RefGANConfig.HMIX_MAX


def test_gan_config_matches_the_notebook_everywhere():
    """The GAN block follows the notebook, and every value is pinned.

    Settled by the saved generator checkpoint: its first layer has shape
    (8, 64), so `latent_dim` is 8. An earlier transcription of this block had
    16, along with nine other values that disagreed with both the notebook and
    Algorithm S2; the checkpoint ruled that reading out and the rest were
    brought into line.

    Worth stating because the DNN block went the other way -- there the
    notebook was the edited copy and the recorded optima proved it. Neither
    source is trustworthy throughout, which is why each value is pinned rather
    than inherited from a policy.
    """
    cfg = _cfg()
    assert cfg.latent_dim == _RefGANConfig.LATENT_DIM == 8
    assert cfg.critic_iterations == _RefGANConfig.N_CRITIC == 4
    assert cfg.epochs == _RefGANConfig.EPOCHS == 10000
    assert cfg.batch_size == _RefGANConfig.BATCH_SIZE == 16
    assert tuple(cfg.proxy_hidden) == _RefGANConfig.PROXY_HIDDEN == (16, 16)
    assert cfg.proxy_max_iter == _RefGANConfig.PROXY_MAXITER == 1000
    assert cfg.fake_pool_min == _RefGANConfig.FAKE_POOL_MIN == 100
    assert cfg.fake_select_k == _RefGANConfig.FAKE_SELECT_K == 20
    assert cfg.surrogate_quantile == _RefGANConfig.SURROGATE_THRESHOLD_Q == 0.99
    assert cfg.surrogate_safety_factor == _RefGANConfig.SURROGATE_THRESHOLD_SAFETY == 2.0
    assert cfg.snap_step == _RefGANConfig.SNAP_STEP == 0.01


def test_generator_architecture_matches_the_released_checkpoint():
    """Weight shapes the saved checkpoint has: (8, 64), (64, 64), (64, 9).

    Pinned as plain arithmetic rather than by loading TensorFlow, so the check
    runs everywhere. If any of these three change, the released weights stop
    loading -- which is a different and worse failure than merely training a
    different model.
    """
    from ane.augment import HIDDEN_UNITS, INPUT_DIM

    cfg = _cfg()
    assert (cfg.latent_dim, HIDDEN_UNITS) == (8, 64)
    assert (HIDDEN_UNITS, HIDDEN_UNITS) == (64, 64)
    assert (HIDDEN_UNITS, INPUT_DIM) == (64, 9)


def test_generator_output_width_is_ilr_plus_properties():
    from ane.augment import INPUT_DIM

    assert N_ILR == 7
    assert INPUT_DIM == _RefGANConfig.INPUT_DIM == 9


def test_snap_distance_is_meaningful_relative_to_the_grid():
    """A snap tolerance well above the grid spacing would admit everything.

    With `snap_step` 0.01 the largest a snap can move a single component is
    0.005, so a tolerance of 0.015 on the eight-dimensional norm allows a
    couple of components to move by the maximum. It is pinned rather than
    adjusted -- the value belongs to the campaign, and changing it would change
    which samples are kept.
    """
    cfg = _cfg()
    max_single_component_shift = cfg.snap_step / 2.0
    assert cfg.snap_distance_max > max_single_component_shift
    assert cfg.snap_distance_max == 0.015


def test_gan_and_selection_grids_agree():
    """Both stages work on a 0.01 grid, and the settings are still separate.

    They are separate fields because generation and enumeration are different
    decisions that a cycle could legitimately make differently. They currently
    coincide, and that is worth pinning: the two were once transposed in the
    configuration, which is the kind of error that produces synthetic data
    lying off the grid the candidates are drawn from.
    """
    from ane.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.gan.snap_step == 0.01
    assert cfg.selection.grid_step == 0.01


def test_config_defaults_match_the_committed_yaml():
    from pathlib import Path

    from ane.config import PipelineConfig

    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert PipelineConfig.from_yaml(yaml_path).as_dict() == PipelineConfig().as_dict()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
