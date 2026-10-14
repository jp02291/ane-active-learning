"""Composition featurization: 7 ILR coordinates + 8 physicochemical descriptors.

This is the only place in the pipeline where a composition becomes a model
input. The generator, the surrogate ensemble and the candidate ranker all call
`featurize` here, which guarantees that a model is applied to features computed
exactly as they were during training.

The transform is a faithful port of the routine used in the original
notebooks; `tests/test_features_parity.py` checks it numerically against that
implementation.

Pipeline
--------
1. closure                    -- rescale each row to sum to one
2. multiplicative_replacement -- replace zeros by `delta`, shrinking the
                                 non-zero parts so the ratios among present
                                 components are preserved
3. isometric log-ratio        -- centered log-ratio projected onto a Helmert
                                 basis, giving 7 orthonormal coordinates
4. atomic descriptors         -- 8 composition-weighted quantities

Steps 1-3 exist because compositions are compositional data: they live on a
simplex, not in Euclidean space, and feeding raw fractions to a neural network
imposes a spurious geometry. The zero replacement is needed because the
log-ratio transform is undefined when a component is absent, which is the
common case here.
"""

from __future__ import annotations

import math

import numpy as np

from .elements import N_ELEMENTS, R_GAS, property_vector

__all__ = [
    "closure",
    "multiplicative_replacement",
    "helmert_submatrix",
    "ILR",
    "atomic_descriptors",
    "featurize",
    "FEATURE_NAMES",
]

FEATURE_NAMES: tuple[str, ...] = (
    "ilr_1", "ilr_2", "ilr_3", "ilr_4", "ilr_5", "ilr_6", "ilr_7",
    "radius_mean",
    "atomic_size_difference",
    "vec_mean",
    "vec_std",
    "atomic_weight_mean",
    "mixing_entropy",
    "electronegativity_mean",
    "electronegativity_std",
)

ZERO_REPLACEMENT_DELTA: float = 1e-3


def closure(A: np.ndarray, axis: int = -1) -> np.ndarray:
    """Rescale so that the components of each composition sum to one."""
    A = np.asarray(A, dtype=np.float64)
    s = A.sum(axis=axis, keepdims=True)
    s[s == 0] = 1.0
    return A / s


def multiplicative_replacement(
    A: np.ndarray, delta: float = ZERO_REPLACEMENT_DELTA
) -> np.ndarray:
    """Replace zero components by `delta` without distorting the others.

    Absent components are set to `delta` and the present ones are shrunk by a
    common factor, so the ratios among the components that are actually there
    survive the substitution. A row of all zeros is mapped to the uniform
    composition rather than raising, since such a row can only arise from a
    malformed input and failing loudly here would abort a long generation run.
    """
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


def helmert_submatrix(D: int) -> np.ndarray:
    """Orthonormal D x (D-1) Helmert basis for the ILR transform."""
    H = np.zeros((D, D - 1), dtype=np.float64)
    for i in range(1, D):
        a = 1.0 / math.sqrt(i * (i + 1))
        H[:i, i - 1] = a
        H[i, i - 1] = -i * a
    return H


class ILR:
    """Isometric log-ratio transform on a fixed Helmert basis."""

    def __init__(self, D: int = N_ELEMENTS) -> None:
        self.D = D
        self.H = helmert_submatrix(D)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"expected an (n, {self.D}) array, got {X.shape}")
        X = multiplicative_replacement(X)
        logx = np.log(closure(X))
        clr = logx - logx.mean(axis=1, keepdims=True)
        return (clr @ self.H).astype(np.float64)


def atomic_descriptors(C: np.ndarray) -> np.ndarray:
    """Eight composition-weighted physicochemical descriptors.

    Returned in the order given by ``FEATURE_NAMES[7:]``: mean metallic radius,
    atomic size difference, mean and standard deviation of the valence electron
    concentration, mean atomic weight, ideal mixing entropy, and mean and
    standard deviation of the electronegativity.
    """
    X = closure(np.asarray(C, dtype=np.float64))
    if X.ndim != 2 or X.shape[1] != N_ELEMENTS:
        raise ValueError(f"expected an (n, {N_ELEMENTS}) array, got {X.shape}")

    radius = property_vector("radius")
    vec = property_vector("vec")
    weight = property_vector("weight")
    en = property_vector("en")

    r_avg = X @ radius
    atomic_size_diff = np.sqrt(
        np.sum(X * (1.0 - radius[None, :] / r_avg[:, None]) ** 2, axis=1)
    )

    vec_avg = X @ vec
    vec_std = np.sqrt(np.sum(X * (vec[None, :] - vec_avg[:, None]) ** 2, axis=1))

    weight_avg = X @ weight

    # clipped so that absent components contribute 0 log 0 -> 0 rather than nan
    X_safe = np.clip(X, 1e-12, None)
    smix = -R_GAS * np.sum(X_safe * np.log(X_safe), axis=1)

    en_avg = X @ en
    en_std = np.sqrt(np.sum(X * (en[None, :] - en_avg[:, None]) ** 2, axis=1))

    return np.column_stack(
        [r_avg, atomic_size_diff, vec_avg, vec_std, weight_avg, smix, en_avg, en_std]
    ).astype(np.float64)


def featurize(X_comp: np.ndarray) -> np.ndarray:
    """Map compositions to the 15-dimensional surrogate input.

    Returns float32 to match the dtype the Keras models were trained on.
    """
    ilr = ILR(D=N_ELEMENTS).transform(X_comp)
    desc = atomic_descriptors(X_comp)
    return np.hstack([ilr, desc]).astype(np.float32)
