"""Thermodynamic state and reduced-unit parameters for two-component hydrogen plasma."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlasmaState:
    """Immutable container for two-component hydrogen plasma state parameters.

    All quantities are in reduced units:
    - length: ion-sphere radius a = (3 / (4 * pi * n_i))^(1/3) = 1 [Eq. (1)]
    - mass: electron mass m_e = 1
    - time: inverse electron plasma frequency 1 / omega_pe = 1 [Eq. (17)]

    Parameters
    ----------
    gamma : float
        Coulomb coupling parameter Gamma = e^2 / (a k_B T) [Eq. (4)] [dimensionless].
    r_s : float
        Brueckner density parameter r_s = a / a_0 [Eq. (6)] [dimensionless].
    n_pairs : int
        Number of proton-electron pairs N (total particles = 2 * n_pairs) [dimensionless].
    mass_ratio : float, optional
        Proton-to-electron mass ratio M = m_i / m_e [Eq. (59)] [dimensionless].
    e_sq : float, optional
        Squared elementary charge e^2 in reduced units [m_e a^3 omega_pe^2].
    """

    gamma: float
    r_s: float
    n_pairs: int = 125
    mass_ratio: float = 1836.15
    e_sq: float = 1.0 / 3.0

    @property
    def n_total(self) -> int:
        """Total number of particles 2N [dimensionless]."""
        return 2 * self.n_pairs

    @property
    def k_t(self) -> float:
        """Thermal energy k_B T = 1 / (3 * Gamma) [m_e a^2 omega_pe^2]."""
        return self.e_sq / self.gamma

    @property
    def volume(self) -> float:
        """Simulation cubic box volume V = (4 * pi / 3) * N [a^3]."""
        return (4.0 * math.pi / 3.0) * self.n_pairs

    @property
    def box(self) -> float:
        """Simulation cubic box side length L = V^(1/3) [a]."""
        return self.volume ** (1.0 / 3.0)

    @property
    def omega_pi(self) -> float:
        """Ion plasma frequency omega_pi = (m_e / m_i)^(1/2) [omega_pe]."""
        return 1.0 / math.sqrt(self.mass_ratio)

    @property
    def masses_by_species(self) -> NDArray[np.float64]:
        """Species masses array [m_i, m_e] in units of [m_e]."""
        return np.array([self.mass_ratio, 1.0], dtype=np.float64)

    @property
    def charges_by_species(self) -> NDArray[np.float64]:
        """Species charge numbers [+1.0, -1.0] in units of [e]."""
        return np.array([1.0, -1.0], dtype=np.float64)

    @property
    def lam(self) -> NDArray[np.float64]:
        """2x2 matrix of thermal de Broglie wavelengths lambda_ab [Eq. (15)] in units of [a]."""
        m_i = self.mass_ratio
        m_e = 1.0
        mu_ii = 0.5 * m_i
        mu_ie = (m_i * m_e) / (m_i + m_e)
        mu_ee = 0.5 * m_e

        pref = self.gamma / (2.0 * math.pi * self.r_s)
        lam_ii = math.sqrt(pref / mu_ii)
        lam_ie = math.sqrt(pref / mu_ie)
        lam_ee = math.sqrt(pref / mu_ee)
        return np.array([[lam_ii, lam_ie], [lam_ie, lam_ee]], dtype=np.float64)
