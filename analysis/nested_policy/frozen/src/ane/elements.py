"""Element inventory and atomic property table.

Single source of truth for the eight-element design space. Every stage of the
pipeline -- generative augmentation, surrogate training, and candidate
selection -- imports the ordering and the property values from here.

That matters more than it looks. In the original notebooks this table was
copied into four files; had one copy drifted, the surrogate would have been
trained on features computed differently from those it was later applied to,
and nothing would have raised an error. Keeping one definition makes that
class of bug impossible.

The property values are those used for the machine-learning descriptors. The
Callaway lattice-thermal-conductivity model in `physics.py` uses a separate
radius set taken from a different source; the two are deliberately not shared,
and both are reported in the supplementary material.
"""

from __future__ import annotations

import numpy as np

#: Composition columns, in the order assumed by every array in the pipeline.
#: Changing this order silently invalidates saved scalers and trained models.
ELEMENTS: tuple[str, ...] = ("Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt")

#: Substitution groups used by the stoichiometry constraint x_A / x_B.
GROUP_A: tuple[str, ...] = ("Fe", "Co", "Mn")
GROUP_B: tuple[str, ...] = ("Ga", "Al", "Si", "Ge", "Pt")

#: radius : metallic radius [A]
#: vec    : valence electron concentration
#: weight : standard atomic weight [u]
#: en     : Pauling electronegativity
ELEMENT_PROPS: dict[str, dict[str, float]] = {
    "Fe": {"radius": 1.26, "vec": 8.0, "weight": 55.845, "en": 1.83},
    "Co": {"radius": 1.25, "vec": 9.0, "weight": 58.933195, "en": 1.88},
    "Mn": {"radius": 1.37, "vec": 7.0, "weight": 54.938044, "en": 1.55},
    "Ga": {"radius": 1.408, "vec": 3.0, "weight": 69.723, "en": 1.81},
    "Al": {"radius": 1.429, "vec": 3.0, "weight": 26.9815385, "en": 1.61},
    "Si": {"radius": 1.316, "vec": 4.0, "weight": 28.085, "en": 1.90},
    "Ge": {"radius": 1.366, "vec": 4.0, "weight": 72.63, "en": 2.01},
    "Pt": {"radius": 1.39, "vec": 10.0, "weight": 195.084, "en": 2.28},
}

#: Molar gas constant [J / (mol K)], CODATA exact.
#:
#: This was briefly truncated to 8.314462618 here while the notebooks carried
#: the full value, which shifted the mixing-entropy descriptor by ~1e-10 in
#: float64. Nothing downstream saw it -- `featurize` casts to float32, where a
#: relative difference of 1e-11 rounds away entirely, so no trained model or
#: fitted scaler was ever affected. It is written out in full anyway: a
#: constant that disagrees with the one the reference implementation uses makes
#: every parity claim conditional on where the rounding happens to land.
R_GAS: float = 8.31446261815324

N_ELEMENTS: int = len(ELEMENTS)
N_ILR: int = N_ELEMENTS - 1          # 7 isometric log-ratio coordinates
N_DESCRIPTORS: int = 8               # physicochemical descriptors
N_FEATURES: int = N_ILR + N_DESCRIPTORS  # 15-dimensional surrogate input


def property_vector(key: str) -> np.ndarray:
    """Return one atomic property as an array ordered like ``ELEMENTS``."""
    if key not in {"radius", "vec", "weight", "en"}:
        raise KeyError(f"unknown atomic property {key!r}")
    return np.array([ELEMENT_PROPS[e][key] for e in ELEMENTS], dtype=np.float64)


def element_index(symbol: str) -> int:
    """Column index of an element in the composition arrays."""
    return ELEMENTS.index(symbol)


def _self_check() -> None:
    missing = [e for e in ELEMENTS if e not in ELEMENT_PROPS]
    if missing:
        raise ValueError(f"ELEMENT_PROPS is missing entries for {missing}")
    if set(GROUP_A) | set(GROUP_B) != set(ELEMENTS):
        raise ValueError("GROUP_A and GROUP_B do not partition ELEMENTS")
    if set(GROUP_A) & set(GROUP_B):
        raise ValueError("GROUP_A and GROUP_B overlap")


_self_check()
