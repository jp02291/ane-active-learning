"""Physics used outside the learned model.

Two independent pieces live here.

`delta_h_mix` screens generated and enumerated compositions for thermodynamic
plausibility. It is a regular-solution sum over binary pairs with enthalpies
from a Miedema-model calculation, and it is applied as a hard filter
(dH_mix <= 0) rather than being fed to the surrogate.

`kappa_lattice` reconstructs the lattice thermal conductivity of the literature
compositions in the initial dataset, for which only |S_ANE| and electrical
conductivity were reported. It is not part of the active-learning loop; it was
used once, to complete the training table. The radii it uses differ from those
in `elements.py` because they come from a different source, and both sets are
reported in the supplementary material rather than reconciled.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .elements import ELEMENTS

__all__ = ["delta_h_mix", "kappa_lattice", "kappa_electronic", "CallawayParams"]

# ---------------------------------------------------------------------------
# Miedema mixing enthalpy
# ---------------------------------------------------------------------------

#: Binary mixing enthalpies at equiatomic composition [kJ/mol], from a Miedema
#: model calculation. Keys are stored in the element order of `ELEMENTS`, not
#: alphabetically, so lookups must try both orderings.
BINARY_ENTHALPY: dict[tuple[str, str], float] = {
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


def binary_enthalpy(e1: str, e2: str) -> float:
    """Look up a pair enthalpy in either key order, defaulting to zero."""
    return BINARY_ENTHALPY.get((e1, e2), BINARY_ENTHALPY.get((e2, e1), 0.0))


def delta_h_mix(compositions: np.ndarray) -> np.ndarray:
    """Regular-solution mixing enthalpy [kJ/mol], one value per composition.

    dH_mix = sum over pairs of 4 * H_ij * c_i * c_j, where the factor of four
    normalizes H_ij, which is tabulated at equiatomic composition.
    """
    comp = np.asarray(compositions, dtype=np.float64)
    if comp.ndim != 2 or comp.shape[1] != len(ELEMENTS):
        raise ValueError(f"expected an (n, {len(ELEMENTS)}) array, got {comp.shape}")
    out = np.zeros(comp.shape[0], dtype=np.float64)
    for i in range(len(ELEMENTS)):
        for j in range(i + 1, len(ELEMENTS)):
            hij = binary_enthalpy(ELEMENTS[i], ELEMENTS[j])
            if hij:
                out += 4.0 * hij * comp[:, i] * comp[:, j]
    return out


# ---------------------------------------------------------------------------
# Callaway lattice thermal conductivity
# ---------------------------------------------------------------------------

K_B = 1.380649e-23        # J/K
H_BAR = 1.0545718e-34     # J s
N_AVOGADRO = 6.02214076e23

#: Atomic data for the Callaway model. Deliberately separate from
#: `elements.ELEMENT_PROPS`: these radii come from the source used for the
#: phonon-scattering calculation and differ from the descriptor radii, most
#: noticeably for Ga (1.53 vs 1.408 A).
CALLAWAY_ELEMENTS: tuple[str, ...] = ("Fe", "Ga", "Al", "Si", "Ge")
CALLAWAY_MASS = np.array([55.84, 69.723, 26.98154, 28.085, 72.63])     # u
CALLAWAY_RADIUS = np.array([1.27, 1.53, 1.43, 1.316, 1.366])           # A


class CallawayParams:
    """Model parameters, as used to complete the literature entries.

    Two parameterizations, one calculation. The defaults below are the standard
    physical form: the Umklapp exponential carries a negative exponent and
    GAMMA is 3.1192, which is what Supplementary Table S6 reports (rounded
    there to 3.12). The campaign itself ran the algebraically equivalent
    positive-exponent form with gamma = 2.0, kept here as LEGACY_UMKLAPP_SIGN
    and LEGACY_GAMMA. The two give identical kappa_L to floating-point
    precision, and `test_callaway_reconstruction.py` asserts it.

    They can, because the reconstruction is evaluated at one temperature. At
    fixed T the exponential is a constant, not a function, so

        exp(+theta_D / 3T) = exp(-theta_D / 3T) * exp(2 theta_D / 3T)

    and the ratio exp(2 theta_D / 3T) = 2.4324 at theta_D = 400 K, T = 300 K is
    absorbed into the Umklapp prefactor B. Since B is proportional to gamma^2,
    the equivalent Grueneisen parameter is 2.0 * sqrt(2.4324) = 3.1192.

    The positive exponent is not the physical form -- it makes Umklapp
    scattering diverge rather than freeze out as T falls -- which is why it is
    no longer the default. Nothing here is evaluated away from 300 K, so the
    choice does not enter the reported numbers; both gamma and epsilon are
    fitted rather than measured. The legacy pair is retained so that the
    campaign's own parameterization stays runnable and the equivalence stays
    testable.
    """

    V_S = 3000.0          # sound velocity [m/s]
    THETA_D = 400.0       # Debye temperature [K]
    VOLUME = 1.217e-29    # volume per atom [m^3]
    L_BOUNDARY = 5e-6     # boundary scattering length [m]
    #: Strain-field parameter used for the deposited literature reconstruction.
    #: This parameter is adjustable, as stated in Supplementary Note S1.
    EPSILON = 0.71

    #: The parameterization the campaign ran. Not the physical sign convention;
    #: kept so the historical calculation remains reproducible.
    LEGACY_GAMMA = 2.0
    LEGACY_UMKLAPP_SIGN = +1

    #: Defaults: the standard form quoted in Supplementary Note S1 and
    #: Supplementary Table S6. GAMMA is derived from the legacy pair rather
    #: than written out, so the equivalence is exact rather than exact to the
    #: two decimals the table prints.
    UMKLAPP_SIGN = -1
    GAMMA = LEGACY_GAMMA * np.sqrt(np.exp(2.0 * THETA_D / (3.0 * 300.0)))

    #: Retained name for the same quantity, used where the text refers to the
    #: value as reported rather than as configured.
    GAMMA_SI_EQUIVALENT = GAMMA


def _fluctuation_parameters(comp: np.ndarray) -> tuple[float, float]:
    """Mass and strain fluctuation parameters Gamma_M and Gamma_S.

    Single-lattice Klemens form: each element contributes its atomic fraction
    times the squared fractional deviation from the composition average. There
    is no per-element (M_i / M_bar)^2 factor. The mass-ratio weight that does
    appear in the multi-sublattice treatment of Abeles is a per-sublattice
    factor (M_bar_s / M_bar)^2, a different quantity that does not apply to the
    single-lattice reconstruction used for the literature entries. See
    `data/literature_reconstruction.csv` and the note in README.
    """
    comp = np.asarray(comp, dtype=np.float64)
    m_bar = float(comp @ CALLAWAY_MASS)
    r_bar = float(comp @ CALLAWAY_RADIUS)
    gamma_m = float((comp * (1.0 - CALLAWAY_MASS / m_bar) ** 2).sum())
    gamma_s = float((comp * (1.0 - CALLAWAY_RADIUS / r_bar) ** 2).sum())
    return gamma_m, gamma_s


def kappa_lattice(
    comp: np.ndarray,
    T: float = 300.0,
    epsilon: float = CallawayParams.EPSILON,
    umklapp_sign: int = CallawayParams.UMKLAPP_SIGN,
    v_s: float = CallawayParams.V_S,
    theta_d: float = CallawayParams.THETA_D,
    gamma: float = CallawayParams.GAMMA,
    volume: float = CallawayParams.VOLUME,
    L: float = CallawayParams.L_BOUNDARY,
) -> float:
    """Callaway integral for kappa_L [W m^-1 K^-1] of one composition.

    `comp` is ordered as `CALLAWAY_ELEMENTS`, not as `elements.ELEMENTS`.

    Defaults are the standard negative-exponent form with the corresponding
    gamma. To reproduce the campaign's own parameterization instead, pass
    `umklapp_sign=CallawayParams.LEGACY_UMKLAPP_SIGN` together with
    `gamma=CallawayParams.LEGACY_GAMMA`; the two give the same kappa_L at
    300 K. The sign and gamma must be changed together -- flipping the sign
    alone moves kappa_L by tens of percent.
    """
    comp = np.asarray(comp, dtype=np.float64)
    if comp.shape[-1] != len(CALLAWAY_ELEMENTS):
        raise ValueError(f"expected {len(CALLAWAY_ELEMENTS)} components, got {comp.shape}")

    m_bar = float(comp @ CALLAWAY_MASS)
    gamma_m, gamma_s = _fluctuation_parameters(comp)

    A = volume / (4.0 * np.pi * v_s**3) * (gamma_m + epsilon * gamma_s)
    B = H_BAR * gamma**2 / (m_bar / N_AVOGADRO / 1000.0) / v_s**2 / theta_d
    boundary = v_s / L

    def integrand(x: float) -> float:
        w = K_B * T / H_BAR * x
        rate = (
            A * w**4
            + B * T * w**2 * np.exp(umklapp_sign * theta_d / 3.0 / T)
            + boundary
        )
        return x**4 * np.exp(x) / rate / (np.exp(x) - 1.0) ** 2

    value, _ = quad(integrand, 1e-9, theta_d / T)
    return K_B / (2.0 * np.pi**2 * v_s) * (K_B * T / H_BAR) ** 3 * value


def kappa_electronic(sigma: float, T: float = 300.0, L0: float = 2.44e-8) -> float:
    """Wiedemann-Franz electronic contribution from electrical conductivity."""
    return L0 * sigma * T
