"""Numerical parity between `ane.features` and the original notebook code.

Refactoring a featurization routine is exactly the kind of change that can
alter results without raising an error, so the ported implementation is checked
against a verbatim copy of the original functions rather than trusted by
inspection. The reference block below is transcribed unchanged from
ensemble_v3.ipynb; do not tidy it.

    python -m pytest tests/test_features_parity.py -q
"""

from __future__ import annotations

import math

import numpy as np

from ane.elements import ELEMENT_PROPS, ELEMENTS
from ane.features import ILR, atomic_descriptors, featurize

#: Hard-coded rather than imported from `ane.elements`, on purpose. Importing
#: the constant would make the reference agree with the package by
#: construction and hide exactly the drift this file exists to catch:
#: `elements.R_GAS` was once truncated to 8.314462618 and the test still passed.
_REF_R_GAS = 8.31446261815324

# ---------------------------------------------------------------------------
# reference implementation, copied verbatim from the original notebooks
# ---------------------------------------------------------------------------
_COMP_COLS = list(ELEMENTS)


def _ref_closure(A, axis=-1):
    A = np.asarray(A, dtype=np.float64)
    s = A.sum(axis=axis, keepdims=True)
    s[s == 0] = 1.0
    return A / s


def _ref_multiplicative_replacement(A, delta=1e-3):
    A = _ref_closure(A)
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
    return _ref_closure(B)


def _ref_helmert_submatrix(D):
    H = np.zeros((D, D - 1), dtype=np.float64)
    for i in range(1, D):
        a = 1.0 / math.sqrt(i * (i + 1))
        H[:i, i - 1] = a
        H[i, i - 1] = -i * a
    return H


class _RefILR:
    def __init__(self, D):
        self.D = D
        self.H = _ref_helmert_submatrix(D)

    def transform(self, X):
        X = _ref_multiplicative_replacement(X)
        logx = np.log(_ref_closure(X))
        clr = logx - logx.mean(axis=1, keepdims=True)
        return (clr @ self.H).astype(np.float64)


def _ref_batch_atomic_properties(C):
    X = _ref_closure(np.asarray(C, dtype=np.float64))
    radius = np.array([ELEMENT_PROPS[e]["radius"] for e in _COMP_COLS], dtype=np.float64)
    vec = np.array([ELEMENT_PROPS[e]["vec"] for e in _COMP_COLS], dtype=np.float64)
    weight = np.array([ELEMENT_PROPS[e]["weight"] for e in _COMP_COLS], dtype=np.float64)
    en = np.array([ELEMENT_PROPS[e]["en"] for e in _COMP_COLS], dtype=np.float64)

    r_avg = X @ radius
    atomic_size_diff = np.sqrt(np.sum(X * (1.0 - radius[None, :] / r_avg[:, None]) ** 2, axis=1))
    vec_avg = X @ vec
    vec_std = np.sqrt(np.sum(X * (vec[None, :] - vec_avg[:, None]) ** 2, axis=1))
    weight_avg = X @ weight
    X_safe = np.clip(X, 1e-12, None)
    smix = -_REF_R_GAS * np.sum(X_safe * np.log(X_safe), axis=1)
    en_avg = X @ en
    en_std = np.sqrt(np.sum(X * (en[None, :] - en_avg[:, None]) ** 2, axis=1))
    return np.column_stack(
        [r_avg, atomic_size_diff, vec_avg, vec_std, weight_avg, smix, en_avg, en_std]
    ).astype(np.float64)


def _ref_ilr_concat(X_comp8):
    ilr = _RefILR(D=8).transform(X_comp8)
    calc = _ref_batch_atomic_properties(X_comp8)
    return np.hstack([ilr, calc]).astype(np.float32)


# ---------------------------------------------------------------------------
# test compositions
# ---------------------------------------------------------------------------
def _sample_compositions(n=400, seed=0):
    """Mixtures spanning the sparsity patterns the pipeline actually sees.

    Real alloys use three to five of the eight elements, so most entries are
    exactly zero. Dense random rows would exercise none of the zero-replacement
    logic, which is where a port is most likely to go wrong.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        k = rng.integers(2, len(ELEMENTS) + 1)
        idx = rng.choice(len(ELEMENTS), size=k, replace=False)
        x = np.zeros(len(ELEMENTS))
        x[idx] = rng.dirichlet(np.ones(k))
        rows.append(x)
    # the two discovered alloys and the reference composition, exactly
    known = np.zeros((3, len(ELEMENTS)))
    known[0, [0, 3, 7]] = [0.74, 0.24, 0.02]        # Fe0.74Ga0.24Pt0.02
    known[1, [0, 3, 4]] = [0.75, 0.13, 0.12]        # Fe0.75Ga0.13Al0.12
    known[2, [0, 3]] = [0.75, 0.25]                 # Fe0.75Ga0.25
    return np.vstack([np.array(rows), known])


X = _sample_compositions()


def test_ilr_matches_reference():
    assert np.array_equal(ILR().transform(X), _RefILR(D=8).transform(X))


def test_descriptors_match_reference():
    assert np.array_equal(atomic_descriptors(X), _ref_batch_atomic_properties(X))


def test_featurize_matches_reference():
    assert np.array_equal(featurize(X), _ref_ilr_concat(X))


def test_featurize_shape_and_dtype():
    F = featurize(X)
    assert F.shape == (len(X), 15)
    assert F.dtype == np.float32


def test_ilr_is_scale_invariant():
    """Compositions differing only by an overall factor must map identically."""
    assert np.allclose(ILR().transform(X), ILR().transform(X * 7.3))


def test_rejects_wrong_width():
    import pytest

    with pytest.raises(ValueError):
        featurize(np.ones((4, 5)))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("\nall parity checks passed")
