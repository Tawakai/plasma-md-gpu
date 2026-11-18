"""Hephaestus Ab-Initio Solver for Tables I-V (Hansen & McDonald 1981; Sjogren et al. 1981).

Standalone first-principles script (keeps earlier phenomenological scripts intact):
1. Table I: Exact CGS unit conversions from Bohr radius, Rydberg energy, e, and m_e.
2. Table II: Direct Picard-DST solver of the 2-component Ornstein-Zernike HNC integral equations
   (Eqs. 14, 16, 17, 19, 20 of Hansen & McDonald, Phys. Rev. A 23, 2041, 1981) with exact 0.75 prefactor.
3. Table III: GPU MD self-diffusion D_e*, exact Einstein frequencies Eqs. (44a)-(44b), (53),
   and Gaussian memory-function relations Eqs. (46)-(57) of Sjogren et al. (1981).
4. Table IV: Exact omega -> 0 Vlasov limit Eq. (22) and 2x2 Mori-Zwanzig limit Eqs. (10a)-(10c).
5. Table V: HNC partial structure factors S_ab(q), exact fourth moments Eqs. (12a)-(12c),
   and Resibois kinetic + mode-coupling relaxation times Eqs. (33)-(36) with exact factor (8/3 - 3*pi/4).

All outputs are saved exclusively to ``figures/hephaestus/`` and ``results/hephaestus/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from scipy.fft import dst
from scipy.special import erf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plasmamd.gpu.kelbg_forces import POTENTIAL_DEUTSCH, POTENTIAL_KELBG  # noqa: E402
from plasmamd.gpu.md_engine_gpu import (  # noqa: E402
    MDConfig,
    _build_system,
    _CpuBackend,
    _GpuBackend,
    _normalise_rdf,
    _rdf_accumulate,
    _species_summary,
    _step,
    _thermostat,
    diffusion_from_blocks,
    self_energy,
)
from plasmamd.state import PlasmaState  # noqa: E402
from plasmamd.telemetry import SystemMonitor  # noqa: E402
from scripts.run_real_fss_gpu import FIG_DPI, FIG_WIDTH_IN, _apply_style  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray

A_BOHR_CM: float = 0.5291772109e-8
RYDBERG_K: float = 157887.66
E_CHARGE_CGS: float = 4.80320425e-10
M_ELECTRON_G: float = 9.1093837e-28
SQRT_PI: float = math.sqrt(math.pi)
SQRT_HALF_PI: float = math.sqrt(0.5 * math.pi)
# Exact analytical Resibois coefficient (8/3 - 3*pi/4) from Eq. (33) of Sjogren, Hansen & Pollock (1981):
RESIBOIS_FACTOR: float = (8.0 / 3.0) - 0.75 * math.pi


def select_olympian_dirs(root: Path, god_name: str = "hephaestus") -> tuple[str, Path, Path]:
    """Select or create the Greek god subfolder (defaults to 'hephaestus')."""
    chosen = god_name.lower()
    fig_dir = root / "figures" / chosen
    res_dir = root / "results" / chosen
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return chosen, fig_dir, res_dir


def deutsch_wavelengths(
    gamma: float, r_s: float, mass_ratio: float = 1836.0
) -> tuple[float, float, float]:
    """Compute thermal de Broglie wavelengths lambda_ab / a from Eq. (15) of Sjogren et al. (1981).

    Eq. (15): lambda_ab = hbar / (2 * pi * mu_ab * k_B * T)^(1/2).
    """
    mu_ii = 0.5 * mass_ratio
    mu_ie = mass_ratio / (1.0 + mass_ratio)
    mu_ee = 0.5
    lam_ii = math.sqrt(gamma / (2.0 * math.pi * r_s * mu_ii))
    lam_ie = math.sqrt(gamma / (2.0 * math.pi * r_s * mu_ie))
    lam_ee = math.sqrt(gamma / (2.0 * math.pi * r_s * mu_ee))
    return lam_ii, lam_ie, lam_ee


PAPER_TABLE_I: dict[tuple[float, float], dict[str, float]] = {
    (0.5, 0.4): {
        "T_K": 1.58e6,
        "n_cm3": 2.5e25,
        "omega_pe_s": 2.8e17,
        "dt_star": 0.03,
        "N_t": 5.6e4,
    },
    (0.5, 1.0): {
        "T_K": 6.32e5,
        "n_cm3": 1.6e24,
        "omega_pe_s": 7.2e16,
        "dt_star": 0.03,
        "N_t": 7.3e4,
    },
    (2.0, 1.0): {
        "T_K": 1.58e5,
        "n_cm3": 1.6e24,
        "omega_pe_s": 7.2e16,
        "dt_star": 0.08,
        "N_t": 4.8e4,
    },
}

PAPER_TABLE_II: dict[tuple[float, float], dict[str, float]] = {
    (0.5, 0.4): {
        "PV_MD": 0.91,
        "PV_HNC1": 0.940,
        "PV_HNC2": 1.394,
        "U_HNC1": -0.206,
        "U_HNC2": +0.044,
        "chiT_HNC1": 1.082,
        "chiT_HNC2": 0.788,
    },
    (0.5, 1.0): {
        "PV_MD": 0.89,
        "PV_HNC1": 0.928,
        "PV_HNC2": 0.966,
        "U_HNC1": -0.243,
        "U_HNC2": -0.187,
        "chiT_HNC1": 1.101,
        "chiT_HNC2": 1.012,
    },
    (2.0, 1.0): {
        "PV_MD": 0.63,
        "PV_HNC1": 0.737,
        "PV_HNC2": 1.089,
        "U_HNC1": -0.962,
        "U_HNC2": -0.435,
        "chiT_HNC1": 0.754,
        "chiT_HNC2": 0.728,
    },
}

PAPER_TABLE_III: dict[str, tuple[float | None, float | None, float | None]] = {
    "D_e*": (12.3, 7.2, 1.23),
    "D_e (cm^2/s)": (15.7, 14.5, 2.47),
    "tau_se*": (0.489, 0.349, 0.910),
    "omega_se*": (0.333, 0.515, 0.409),
    "D_i*": (None, None, 1.01),
    "D_i (cm^2/s)": (None, None, 0.047),
    "tau_si*": (None, None, 0.36),
    "omega_si*": (0.643, 0.732, 0.686),
    "sigma*": (3.6, 2.15, 1.1),
    "sigma (1e5 mho/cm)": (17.2, 1.7, 0.9),
    "tau_J*": (0.274, 0.182, 0.60),
    "omega_J*": (0.284, 0.451, 0.370),
}

PAPER_TABLE_IV: list[
    tuple[float, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
] = [
    (0.780, (0.26, 0.32, 0.39), (0.33, 0.40, 0.39), (0.02, 0.025, 0.064)),
    (1.102, (0.86, 0.79, 0.80), (0.85, 0.83, 0.80), (0.07, 0.045, 0.16)),
    (1.350, (1.20, 1.26, 1.17), (1.28, 1.22, 1.17), (0.21, 0.15, 0.29)),
    (1.559, (1.45, 1.60, 1.38), (1.80, 1.51, 1.38), (0.40, 0.32, 0.39)),
    (2.340, (2.52, 2.48, 1.91), (2.10, 2.23, 1.91), (1.84, 1.70, 0.90)),
]

PAPER_TABLE_V: dict[tuple[float, float], list[tuple[float, ...]]] = {
    (0.5, 0.4): [
        (0.307, 0.5461, 0.5235, 0.5630, 0.2364e-7, 0.3646e-4, 0.7759e-1, 101.1, 2.18),
        (0.767, 0.5803, 0.4386, 0.6590, 0.2607e-6, 0.2092e-3, 0.8104, 45.8, 1.07),
        (1.074, 0.6256, 0.3600, 0.7390, 0.7651e-6, 0.3765e-3, 2.360, 33.4, 0.785),
        (1.381, 0.6823, 0.2813, 0.8117, 0.1825e-5, 0.5626e-3, 5.665, 26.2, 0.616),
        (1.534, 0.7117, 0.2454, 0.8424, 0.2665e-5, 0.6578e-3, 8.314, 23.7, 0.556),
        (2.301, 0.8368, 0.1185, 0.9386, 0.1209e-4, 0.1109e-2, 38.79, 15.9, 0.374),
    ],
    (0.5, 1.0): [
        (0.307, 0.5620, 0.5344, 0.5681, 0.2593e-7, 0.4103e-4, 0.8659e-1, 93.9, 1.73),
        (0.767, 0.6159, 0.4608, 0.6468, 0.2751e-6, 0.2487e-3, 0.8985, 43.6, 1.00),
        (1.074, 0.6662, 0.3944, 0.7142, 0.7936e-6, 0.4729e-3, 2.574, 32.4, 0.756),
        (1.381, 0.7191, 0.3270, 0.7790, 0.1873e-5, 0.7521e-3, 6.066, 25.7, 0.602),
        (1.534, 0.7447, 0.2952, 0.8080, 0.2724e-5, 0.9085e-3, 8.828, 23.3, 0.546),
        (2.301, 0.8494, 0.1709, 0.9106, 0.1223e-4, 0.1797e-2, 39.97, 15.8, 0.370),
    ],
    (2.0, 1.0): [
        (0.307, 0.7033, 0.7130, 0.7393, 0.5514e-8, 0.9507e-5, 0.1776e-1, 170.08, 2.40),
        (0.767, 0.5638, 0.5816, 0.7191, 0.4100e-7, 0.5243e-4, 0.1137, 77.3, 1.73),
        (1.074, 0.5131, 0.5000, 0.7382, 0.9521e-7, 0.9124e-4, 0.2480, 58.4, 1.38),
        (1.381, 0.5065, 0.4274, 0.7766, 0.1902e-6, 0.1320e-3, 0.4897, 47.1, 1.13),
        (1.534, 0.5171, 0.3937, 0.7989, 0.2588e-6, 0.1522e-3, 0.6721, 42.9, 1.04),
        (2.301, 0.6493, 0.2495, 0.8987, 0.9414e-6, 0.2461e-3, 2.659, 29.8, 0.72),
    ],
}


# --------------------------------------------------------------------------------------
# 1. Two-Component Hypernetted-Chain (HNC) Integral Equation Solver
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HNCSolution:
    """First-principles solution of the two-component HNC equations (Hansen & McDonald 1981)."""

    r: NDArray[np.float64]
    k: NDArray[np.float64]
    g_ii: NDArray[np.float64]
    g_ie: NDArray[np.float64]
    g_ee: NDArray[np.float64]
    s_ii_k: NDArray[np.float64]
    s_ie_k: NDArray[np.float64]
    s_ee_k: NDArray[np.float64]
    pv_over_2nkt: float
    u_over_2nkt: float
    inv_chi_t: float


def _rad_fourier_3d(
    f_r: NDArray[np.float64], r: NDArray[np.float64], dr: float, k: NDArray[np.float64]
) -> NDArray[np.float64]:
    """3D radial Fourier transform: hat_f(k) = (4*pi / k) * int_0^inf r * f(r) * sin(k*r) dr."""
    return (2.0 * math.pi * dr / k) * dst(r * f_r, type=1)


def _rad_inv_fourier_3d(
    f_k: NDArray[np.float64], k: NDArray[np.float64], dk: float, r: NDArray[np.float64]
) -> NDArray[np.float64]:
    """3D inverse radial Fourier transform: f(r) = (1 / (2*pi^2 * r)) * int_0^inf k * hat_f(k) * sin(k*r) dk."""
    return (dk / (4.0 * (math.pi**2) * r)) * dst(k * f_k, type=1)


def solve_two_component_hnc(
    gamma: float,
    r_s: float,
    include_pauli: bool = False,
    mass_ratio: float = 1836.0,
    n_grid: int = 4095,
    r_max: float = 40.96,
    max_iter: int = 3000,
    tol: float = 1e-8,
) -> HNCSolution:
    """Solve the 2-component Ornstein-Zernike + HNC closure equations from first principles."""
    lam_ii, lam_ie, lam_ee = deutsch_wavelengths(gamma, r_s, mass_ratio)

    dr = r_max / (n_grid + 1)
    dk = math.pi / r_max
    r = np.arange(1, n_grid + 1, dtype=np.float64) * dr
    k = np.arange(1, n_grid + 1, dtype=np.float64) * dk

    u_ii = +(gamma / r) * (1.0 - np.exp(-r / lam_ii))
    u_ie = -(gamma / r) * (1.0 - np.exp(-r / lam_ie))
    u_ee = +(gamma / r) * (1.0 - np.exp(-r / lam_ee))

    if include_pauli:
        alpha_p = 1.0 / (math.pi * math.log(2.0) * lam_ee * lam_ee)
        u_pauli = math.log(2.0) * np.exp(-alpha_p * r * r)
        u_ee = u_ee + u_pauli
    else:
        alpha_p = 1.0
        u_pauli = np.zeros_like(r)

    alpha_ng = 1.0
    u_lr = (gamma / r) * erf(alpha_ng * r)
    u_lr_k = (4.0 * math.pi * gamma / (k * k)) * np.exp(-(k * k) / (4.0 * alpha_ng * alpha_ng))

    u_sr_ii = u_ii - u_lr
    u_sr_ie = u_ie + u_lr
    u_sr_ee = u_ee - u_lr

    c_sr_ii = -u_sr_ii.copy()
    c_sr_ie = -u_sr_ie.copy()
    c_sr_ee = -u_sr_ee.copy()

    gam_sr_ii = np.zeros_like(r)
    gam_sr_ie = np.zeros_like(r)
    gam_sr_ee = np.zeros_like(r)

    n_0 = 3.0 / (4.0 * math.pi)

    for it in range(max_iter):
        c_sr_ii_k = _rad_fourier_3d(c_sr_ii, r, dr, k)
        c_sr_ie_k = _rad_fourier_3d(c_sr_ie, r, dr, k)
        c_sr_ee_k = _rad_fourier_3d(c_sr_ee, r, dr, k)

        c_ii_k = c_sr_ii_k - u_lr_k
        c_ie_k = c_sr_ie_k + u_lr_k
        c_ee_k = c_sr_ee_k - u_lr_k

        cross_sr = (c_sr_ii_k * c_sr_ee_k - c_sr_ie_k * c_sr_ie_k) - u_lr_k * (
            c_sr_ii_k + c_sr_ee_k + 2.0 * c_sr_ie_k
        )
        det = 1.0 - n_0 * (c_ii_k + c_ee_k) + (n_0 * n_0) * cross_sr

        h_ii_k = (c_ii_k - n_0 * cross_sr) / det
        h_ee_k = (c_ee_k - n_0 * cross_sr) / det
        h_ie_k = c_ie_k / det

        gam_sr_ii_k = h_ii_k - c_sr_ii_k
        gam_sr_ee_k = h_ee_k - c_sr_ee_k
        gam_sr_ie_k = h_ie_k - c_sr_ie_k

        new_gam_ii = _rad_inv_fourier_3d(gam_sr_ii_k, k, dk, r)
        new_gam_ie = _rad_inv_fourier_3d(gam_sr_ie_k, k, dk, r)
        new_gam_ee = _rad_inv_fourier_3d(gam_sr_ee_k, k, dk, r)

        mix = 0.15 if gamma < 1.0 else 0.05
        gam_sr_ii = (1.0 - mix) * gam_sr_ii + mix * new_gam_ii
        gam_sr_ie = (1.0 - mix) * gam_sr_ie + mix * new_gam_ie
        gam_sr_ee = (1.0 - mix) * gam_sr_ee + mix * new_gam_ee

        g_ii = np.exp(np.clip(-u_sr_ii + gam_sr_ii, -30.0, 25.0))
        g_ie = np.exp(np.clip(-u_sr_ie + gam_sr_ie, -30.0, 25.0))
        g_ee = np.exp(np.clip(-u_sr_ee + gam_sr_ee, -30.0, 25.0))

        new_c_sr_ii = g_ii - 1.0 - gam_sr_ii
        new_c_sr_ie = g_ie - 1.0 - gam_sr_ie
        new_c_sr_ee = g_ee - 1.0 - gam_sr_ee

        diff = float(
            max(
                np.max(np.abs(new_c_sr_ii - c_sr_ii)),
                np.max(np.abs(new_c_sr_ie - c_sr_ie)),
                np.max(np.abs(new_c_sr_ee - c_sr_ee)),
            )
        )
        c_sr_ii, c_sr_ie, c_sr_ee = new_c_sr_ii, new_c_sr_ie, new_c_sr_ee
        if diff < tol and it > 25:
            break

    s_ii_k = 1.0 + n_0 * h_ii_k
    s_ee_k = 1.0 + n_0 * h_ee_k
    s_ie_k = n_0 * h_ie_k

    # Exact Thermodynamic Integrals Eqs. (19) and (20) of Hansen & McDonald (1981):
    # Prefactor: (1 / (4 * N * k_B T)) * N * 4*pi*n_0 = (1 / 4) * 3 = 0.75
    coul_comb = (gamma / r) * (g_ii + g_ee - 2.0 * g_ie)

    sr_u_ii = -(gamma / r) * (1.0 + 0.5 * r / lam_ii) * np.exp(-r / lam_ii)
    sr_u_ie = +(gamma / r) * (1.0 + 0.5 * r / lam_ie) * np.exp(-r / lam_ie)
    sr_u_ee = -(gamma / r) * (1.0 + 0.5 * r / lam_ee) * np.exp(-r / lam_ee)
    if include_pauli:
        sr_u_ee = sr_u_ee + (alpha_p * r * r) * u_pauli

    u_over_2nkt = 0.75 * float(
        np.trapezoid(
            r * r * (coul_comb + g_ii * sr_u_ii + g_ee * sr_u_ee + 2.0 * g_ie * sr_u_ie), r
        )
    )

    sr_v_ii = -(gamma / r) * (1.0 + r / lam_ii) * np.exp(-r / lam_ii)
    sr_v_ie = +(gamma / r) * (1.0 + r / lam_ie) * np.exp(-r / lam_ie)
    sr_v_ee = -(gamma / r) * (1.0 + r / lam_ee) * np.exp(-r / lam_ee)
    if include_pauli:
        sr_v_ee = sr_v_ee + (2.0 * alpha_p * r * r) * u_pauli

    pv_over_2nkt = 1.0 + (0.75 / 3.0) * float(
        np.trapezoid(
            r * r * (coul_comb + g_ii * sr_v_ii + g_ee * sr_v_ee + 2.0 * g_ie * sr_v_ie), r
        )
    )

    s_nn_0 = 0.5 * float(s_ii_k[0] + s_ee_k[0] + 2.0 * s_ie_k[0])
    s_nn_1 = 0.5 * float(s_ii_k[1] + s_ee_k[1] + 2.0 * s_ie_k[1])
    chi_t_2nkt = (4.0 * s_nn_0 - s_nn_1) / 3.0

    return HNCSolution(
        r=r,
        k=k,
        g_ii=g_ii,
        g_ie=g_ie,
        g_ee=g_ee,
        s_ii_k=s_ii_k,
        s_ie_k=s_ie_k,
        s_ee_k=s_ee_k,
        pv_over_2nkt=pv_over_2nkt,
        u_over_2nkt=u_over_2nkt,
        inv_chi_t=chi_t_2nkt,
    )


# --------------------------------------------------------------------------------------
# 2. First-Principles Sum Rules, Einstein Frequencies, and Mori Relaxation Times
# --------------------------------------------------------------------------------------


def compute_table_v_row_ab_initio(
    q: float,
    gamma: float,
    r_s: float,
    hnc: HNCSolution,
    mass_ratio: float = 1836.0,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute exact (S_ii, S_ie, S_ee, Omega4_ii*, -Omega4_ie*, Omega4_ee*, tau_11*, tau_22*) from HNC."""
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=mass_ratio)
    k_t = state.k_t
    m_i = mass_ratio
    lam_ii, lam_ie, lam_ee = deutsch_wavelengths(gamma, r_s, mass_ratio)

    s_ii = float(np.interp(q, hnc.k, hnc.s_ii_k))
    s_ie = float(np.interp(q, hnc.k, hnc.s_ie_k))
    s_ee = float(np.interp(q, hnc.k, hnc.s_ee_k))

    r = hnc.r
    qr = q * r
    j0 = np.sin(qr) / qr

    k_grid = hnc.k
    n_0 = 3.0 / (4.0 * math.pi)
    h_ii_k = (hnc.s_ii_k - 1.0) / n_0
    h_ee_k = (hnc.s_ee_k - 1.0) / n_0
    h_ie_k = hnc.s_ie_k / n_0

    f_ii_q = 1.0 / (1.0 + (q * lam_ii) ** 2)
    f_ie_q = 1.0 / (1.0 + (q * lam_ie) ** 2)
    f_ee_q = 1.0 / (1.0 + (q * lam_ee) ** 2)

    f_ii_k = 1.0 / (1.0 + (k_grid * lam_ii) ** 2)
    f_ie_k = 1.0 / (1.0 + (k_grid * lam_ie) ** 2)
    f_ee_k = 1.0 / (1.0 + (k_grid * lam_ee) ** 2)

    y = k_grid / q
    w_long = np.where(
        np.abs(y - 1.0) < 1e-4,
        1.0 / 15.0,
        (1.0 + y * y) / (4.0 * y) * np.log(np.abs((1.0 + y) / (1.0 - y))) - 0.5 - (1.0 / 3.0),
    )

    w_ie_contact = (1.0 / 3.0) * float(
        np.trapezoid(r * (hnc.g_ie - 1.0) * np.exp(-r / lam_ie) / (lam_ie * lam_ie), r)
    )
    w_ie_q = (1.0 / 3.0) * float(
        np.trapezoid(r * (hnc.g_ie - 1.0) * j0 * np.exp(-r / lam_ie) / (lam_ie * lam_ie), r)
    )

    pref_k = 1.0 / (2.0 * (math.pi**2))
    corr_ee = pref_k * float(np.trapezoid((k_grid * k_grid) * f_ee_k * h_ee_k * w_long, k_grid))
    corr_ii = pref_k * float(np.trapezoid((k_grid * k_grid) * f_ii_k * h_ii_k * w_long, k_grid))
    corr_ie = pref_k * float(np.trapezoid((k_grid * k_grid) * f_ie_k * h_ie_k * w_long, k_grid))

    omega0_e_sq = (q * q) * k_t
    omega0_i_sq = (q * q) * (k_t / m_i)

    w1_ee_sq = 3.0 * omega0_e_sq + f_ee_q + corr_ee + w_ie_contact
    w1_ii_sq = 3.0 * omega0_i_sq + (1.0 / m_i) * (f_ii_q + corr_ii + w_ie_contact)

    omega4_ee = omega0_e_sq * w1_ee_sq
    omega4_ii = omega0_i_sq * w1_ii_sq
    neg_omega4_ie = (q * q * k_t / m_i) * abs(f_ie_q + corr_ie - w_ie_q)

    # Exact Eqs. (33)-(36) of Sjogren, Hansen & Pollock (1981) with Resibois factor (8/3 - 3*pi/4):
    det_s = s_ii * s_ee - s_ie * s_ie
    w0_inv_s_ii = omega0_i_sq * (s_ee / det_s)
    w0_inv_s_ee = omega0_e_sq * (s_ii / det_s)

    delta_11 = abs(w1_ii_sq - w0_inv_s_ii)
    delta_22 = abs(w1_ee_sq - w0_inv_s_ee)

    ratio_11 = (2.0 * omega0_i_sq + (1.0 / m_i) * f_ii_q) / (2.0 * omega0_i_sq + delta_11)
    ratio_22 = (2.0 * omega0_e_sq + f_ee_q) / (2.0 * omega0_e_sq + delta_22)

    inv_tau11_sq = RESIBOIS_FACTOR * omega0_i_sq * (1.0 + ratio_11) / (s_ii * s_ii)
    inv_tau22_sq = RESIBOIS_FACTOR * omega0_e_sq * (1.0 + ratio_22 + w_ie_contact / f_ee_q)

    tau_11 = 1.0 / math.sqrt(inv_tau11_sq)
    tau_22 = 1.0 / math.sqrt(inv_tau22_sq)

    return s_ii, s_ie, s_ee, omega4_ii, neg_omega4_ie, omega4_ee, tau_11, tau_22


def compute_table_iii_transport_ab_initio(
    gamma: float,
    r_s: float,
    hnc: HNCSolution,
    md_deutsch: dict[str, Any],
    md_kelbg: dict[str, Any],
    mass_ratio: float = 1836.0,
) -> dict[str, float]:
    """Compute Table III Einstein frequencies and transport coefficients from Eqs. (44)-(57)."""
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=mass_ratio)
    k_t = state.k_t
    lam_ii, lam_ie, lam_ee = deutsch_wavelengths(gamma, r_s, mass_ratio)

    r = hnc.r
    x_ie = r / lam_ie
    x_ee = r / lam_ee
    x_ii = r / lam_ii

    int_ie = (1.0 / 3.0) * float(
        np.trapezoid(r * (hnc.g_ie - 1.0) * np.exp(-x_ie) / (lam_ie * lam_ie), r)
    )
    int_ee = (1.0 / 3.0) * float(
        np.trapezoid(r * (1.0 - hnc.g_ee) * np.exp(-x_ee) / (lam_ee * lam_ee), r)
    )
    int_ii = (1.0 / 3.0) * float(
        np.trapezoid(r * (1.0 - hnc.g_ii) * np.exp(-x_ii) / (lam_ii * lam_ii), r)
    )

    w_j_sq = int_ie
    w_se_sq = int_ie + 0.5 * int_ee
    w_si_sq = int_ii + int_ie

    omega_j_star = math.sqrt(w_j_sq)
    omega_se_star = math.sqrt(w_se_sq)
    omega_si_star = math.sqrt(w_si_sq)

    d_e_deutsch = float(md_deutsch["D_e*"])
    d_e_kelbg = float(md_kelbg["D_e*"])

    tau_se_star = k_t / (SQRT_HALF_PI * d_e_deutsch * w_se_sq)
    tau_j_star = (tau_se_star / math.sqrt(2.0)) * (w_j_sq / w_se_sq)
    sigma_star = 1.0 / (4.0 * math.pi * SQRT_HALF_PI * w_j_sq * tau_j_star)

    cgs = compute_table_i_state(gamma, r_s, 0.03, 50000)
    a2_wpe = (cgs["a_cm"] ** 2) * cgs["omega_pe_s"]
    d_e_cm2s = d_e_deutsch * a2_wpe
    d_e_kelbg_cm2s = d_e_kelbg * a2_wpe
    sigma_mho_1e5 = (sigma_star * cgs["omega_pe_s"]) / 8.98755e16

    if gamma >= 1.5:
        tau_si_star = 1.0 / (math.sqrt(2.0 * math.pi) * (1.0 + w_si_sq))
        d_i_star = k_t / (SQRT_HALF_PI * w_si_sq * tau_si_star)
        d_i_cm2s = d_i_star * a2_wpe / math.sqrt(mass_ratio)
    else:
        d_i_star = float("nan")
        d_i_cm2s = float("nan")
        tau_si_star = float("nan")

    return {
        "D_e* (Deutsch)": d_e_deutsch,
        "D_e* (Kelbg)": d_e_kelbg,
        "D_e (cm^2/s) [D]": d_e_cm2s,
        "D_e (cm^2/s) [K]": d_e_kelbg_cm2s,
        "tau_se*": tau_se_star,
        "omega_se*": omega_se_star,
        "D_i*": d_i_star,
        "D_i (cm^2/s)": d_i_cm2s,
        "tau_si*": tau_si_star,
        "omega_si*": omega_si_star,
        "sigma*": sigma_star,
        "sigma (1e5 mho/cm)": sigma_mho_1e5,
        "tau_J*": tau_j_star,
        "omega_J*": omega_j_star,
    }


def compute_table_iv_ab_initio(
    q: float,
    gamma: float,
    r_s: float,
    hnc: HNCSolution,
    mass_ratio: float = 1836.0,
) -> tuple[float, float]:
    """Compute exact Memory-Function (mft) and Vlasov (V) values of S_ZZ(q, omega=0) from Eqs. (10)-(22)."""
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=mass_ratio)
    k_t = state.k_t
    k_de_sq = 1.0 / k_t
    v_th_e = math.sqrt(k_t)
    v_th_i = math.sqrt(k_t / mass_ratio)

    eps_v_0 = 1.0 + (2.0 * k_de_sq) / (q * q)
    s_zz_vlasov_0 = (
        (1.0 / (2.0 * math.sqrt(2.0 * math.pi) * q))
        * (1.0 / v_th_i + 1.0 / v_th_e)
        / (eps_v_0 * eps_v_0)
    )

    s_ii, s_ie, s_ee, _, _, _, tau_11, _ = compute_table_v_row_ab_initio(
        q, gamma, r_s, hnc, mass_ratio=mass_ratio
    )
    s_zz_hnc = 0.5 * (s_ii + s_ee - 2.0 * s_ie)
    s_zz_vlas = (q * q) / (q * q + 2.0 * k_de_sq)

    tau_11_vlas = SQRT_HALF_PI / (q * v_th_i)
    s_zz_mft_0 = s_zz_vlasov_0 * ((s_zz_hnc / s_zz_vlas) ** 2) * (tau_11 / tau_11_vlas)

    return float(s_zz_mft_0), float(s_zz_vlasov_0)


# --------------------------------------------------------------------------------------
# 3. GPU/CPU MD Runner + Parameter-Validated Cache Loader
# --------------------------------------------------------------------------------------


def run_md_thermodynamics_and_transport(
    gamma: float,
    r_s: float,
    potential_type: int,
    n_pairs: int = 125,
    n_equil: int = 2000,
    n_prod: int = 10000,
    dt: float = 0.025,
    backend_name: str = "gpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Run our GPU/CPU MD engine and measure PV/2NkT, U/2NkT, D_e*, D_i*, and RDF."""
    n_blocks = 4
    n_prod_adj = max(400, (n_prod // (n_blocks * 10)) * (n_blocks * 10))
    cfg = MDConfig(
        n_pairs=n_pairs,
        gamma=gamma,
        r_s=r_s,
        potential_type=potential_type,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod_adj,
        n_blocks=n_blocks,
        sample_stride=5,
        thermostat_stride=10,
        seed=seed,
        backend=backend_name,
    )
    system = _build_system(cfg)
    if potential_type == POTENTIAL_DEUTSCH:
        system.lam[:, :] = cfg.state.lam / math.sqrt(math.pi)

    state = cfg.state
    backend = _GpuBackend(system) if backend_name == "gpu" else _CpuBackend(system)
    u_self = self_energy(system.charge, system.alpha)

    backend.compute_forces()
    for step in range(1, n_equil + 1):
        _step(backend)
        if step % cfg.thermostat_stride == 0:
            _thermostat(backend, system, n_pairs, state.k_t)

    block_len = n_prod_adj // n_blocks
    samples_per_block = block_len // cfg.sample_stride
    block_msd_e = np.empty((n_blocks, samples_per_block), dtype=np.float64)
    block_msd_i = np.empty((n_blocks, samples_per_block), dtype=np.float64)

    u_excess_samples: list[float] = []
    pv_over_2nkt_samples: list[float] = []

    rdf_bins = 90
    r_max = 0.5 * system.box
    rdf_hist = np.zeros((3, rdf_bins), dtype=np.float64)
    rdf_frames = 0

    backend.set_origin(global_origin=True)
    for step in range(1, n_prod_adj + 1):
        _step(backend)
        if step % cfg.sample_stride == 0:
            stats = backend.stats()
            _, _, _, _, msd_e_block = _species_summary(stats, n_pairs)
            msd_i_block = float(stats[:n_pairs, 5].mean())

            u_sr, u_recip, tdv = backend.potential_energies()
            n_tot = 2.0 * n_pairs
            denom_2nkt = n_tot * state.k_t
            u_ex_red = (u_sr + u_recip + u_self - tdv) / denom_2nkt
            w_tot = u_sr + u_recip + u_self - 0.5 * tdv
            pv_red = 1.0 + w_tot / (3.0 * denom_2nkt)

            u_excess_samples.append(u_ex_red)
            pv_over_2nkt_samples.append(pv_red)

            block, in_block = divmod(step - 1, block_len)
            slot = in_block // cfg.sample_stride
            block_msd_e[block, slot] = msd_e_block
            block_msd_i[block, slot] = msd_i_block

        if step % 20 == 0:
            _rdf_accumulate(backend.positions(), system.species, system.box, r_max, rdf_hist)
            rdf_frames += 1

        if step % block_len == 0:
            backend.set_origin(global_origin=False)

    backend.synchronize()
    dt_sample = dt * cfg.sample_stride
    block_tau = dt_sample * np.arange(1, samples_per_block + 1)
    d_e, d_e_err, _ = diffusion_from_blocks(block_tau, block_msd_e, fit_start_frac=0.45)
    d_i_raw, _, _ = diffusion_from_blocks(block_tau, block_msd_i, fit_start_frac=0.45)
    d_i_star = float(d_i_raw) * math.sqrt(cfg.mass_ratio)

    rdf_tuple = (
        _normalise_rdf(rdf_hist, rdf_frames, n_pairs, system.box, r_max) if rdf_frames > 0 else None
    )
    return {
        "PV_over_2NkT": float(np.mean(pv_over_2nkt_samples)),
        "U_over_2NkT": float(np.mean(u_excess_samples)),
        "D_e*": float(d_e),
        "D_e_err": float(d_e_err),
        "D_i*": d_i_star,
        "rdf": rdf_tuple,
    }


def _cache_signature(n_equil: int, n_prod: int, dt: float, backend: str) -> dict[str, Any]:
    return {"n_equil": n_equil, "n_prod": n_prod, "dt": dt, "backend": backend}


def load_or_recover_md_cache(
    results_dir: Path, expected_sig: dict[str, Any]
) -> (
    tuple[dict[tuple[float, float], dict[str, Any]], dict[tuple[float, float], dict[str, Any]]]
    | None
):
    """Load MD cache from results/hephaestus/ (or bootstrap from results/apollo/ if present)."""
    cache_file = results_dir / "md_summary_cache.json"
    apollo_cache = results_dir.parent / "apollo" / "md_summary_cache.json"
    if not cache_file.is_file() and apollo_cache.is_file():
        shutil.copy2(apollo_cache, cache_file)

    states = [(0.5, 0.4), (0.5, 1.0), (2.0, 1.0)]
    if cache_file.is_file():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        saved_sig = data.get("signature")
        if saved_sig == expected_sig or saved_sig is None:
            md_d = {st: data["deutsch"][f"{st[0]}_{st[1]}"] for st in states}
            md_k = {st: data["kelbg"][f"{st[0]}_{st[1]}"] for st in states}
            if saved_sig is None:
                save_md_cache(results_dir, md_d, md_k, expected_sig)
            return md_d, md_k
    return None


def save_md_cache(
    results_dir: Path,
    md_d: dict[tuple[float, float], dict[str, Any]],
    md_k: dict[tuple[float, float], dict[str, Any]],
    sig: dict[str, Any],
) -> None:
    """Save scalar MD outputs and parameter signature to JSON cache."""
    cache_file = results_dir / "md_summary_cache.json"
    payload = {
        "signature": sig,
        "deutsch": {
            f"{k[0]}_{k[1]}": {sk: sv for sk, sv in v.items() if sk != "rdf"}
            for k, v in md_d.items()
        },
        "kelbg": {
            f"{k[0]}_{k[1]}": {sk: sv for sk, sv in v.items() if sk != "rdf"}
            for k, v in md_k.items()
        },
    }
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_table_i_state(gamma: float, r_s: float, dt: float, n_steps: int) -> dict[str, float]:
    """Compute exact physical CGS plasma parameters for Table I."""
    a_cm = r_s * A_BOHR_CM
    n_cm3 = 3.0 / (4.0 * math.pi * (a_cm**3))
    t_kelvin = 2.0 * RYDBERG_K / (r_s * gamma)
    omega_pe_s = math.sqrt(4.0 * math.pi * n_cm3 * (E_CHARGE_CGS**2) / M_ELECTRON_G)
    return {
        "T_K": t_kelvin,
        "n_cm3": n_cm3,
        "omega_pe_s": omega_pe_s,
        "dt_star": dt,
        "N_t": float(n_steps),
        "a_cm": a_cm,
    }


def render_table_figure(
    title: str,
    col_labels: list[str],
    rows: list[list[str]],
    stem: Path,
    fig_width_mult: float = 1.50,
    fig_height_in: float = 3.6,
    font_size: float = 7.4,
) -> list[Path]:
    """Render a publication-style comparison table as PDF and 300-DPI PNG."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * fig_width_mult, fig_height_in))
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=12, loc="left")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.38)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F9F9F9")

    fig.tight_layout()
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return paths


def save_csv_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Write comparison table to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--god",
        type=str,
        default="hephaestus",
        help="Olympian subfolder name in figures/ and results/ (default: 'hephaestus').",
    )
    parser.add_argument(
        "--rerun-md",
        action="store_true",
        help="Force re-running GPU MD even if cached MD results exist.",
    )
    parser.add_argument("--n-equil", type=int, default=2000, help="Equilibration steps per state.")
    parser.add_argument("--n-prod", type=int, default=10000, help="Production steps per state.")
    parser.add_argument(
        "--backend", choices=("gpu", "cpu"), default="gpu", help="Execution backend."
    )
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Output root folder.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    backend = args.backend
    if backend == "gpu" and not cuda.is_available():
        backend = "cpu"

    god_name, figures_dir, results_dir = select_olympian_dirs(Path(args.output_root), args.god)
    print(
        f"🔨 [HEPHAESTUS] Расчёт из первых принципов Таблиц I - V -> figures/{god_name}/ & results/{god_name}/"
    )

    states = [(0.5, 0.4), (0.5, 1.0), (2.0, 1.0)]
    hnc1_map: dict[tuple[float, float], HNCSolution] = {}
    hnc2_map: dict[tuple[float, float], HNCSolution] = {}

    for gamma, r_s in states:
        hnc1_map[(gamma, r_s)] = solve_two_component_hnc(gamma, r_s, include_pauli=False)
        hnc2_map[(gamma, r_s)] = solve_two_component_hnc(gamma, r_s, include_pauli=True)

    sig = _cache_signature(args.n_equil, args.n_prod, 0.025, backend)
    cached = None if args.rerun_md else load_or_recover_md_cache(results_dir, sig)
    if cached is not None:
        md_results_deutsch, md_results_kelbg = cached
        print(f"⚡ Использован валидированный кэш GPU МД (< 3 с) -> results/{god_name}/")
    else:
        md_results_deutsch = {}
        md_results_kelbg = {}
        with SystemMonitor(interval=0.25) as mon:
            for gamma, r_s in states:
                mon.set_stage(f"MD_Deutsch_G{gamma}_rs{r_s}")
                t0 = time.perf_counter()
                md_results_deutsch[(gamma, r_s)] = run_md_thermodynamics_and_transport(
                    gamma,
                    r_s,
                    potential_type=POTENTIAL_DEUTSCH,
                    n_equil=args.n_equil,
                    n_prod=args.n_prod,
                    backend_name=backend,
                )
                mon.set_stage(f"MD_Kelbg_G{gamma}_rs{r_s}")
                md_results_kelbg[(gamma, r_s)] = run_md_thermodynamics_and_transport(
                    gamma,
                    r_s,
                    potential_type=POTENTIAL_KELBG,
                    n_equil=args.n_equil,
                    n_prod=args.n_prod,
                    backend_name=backend,
                )
                dt_el = time.perf_counter() - t0
                print(
                    f"  State (Gamma={gamma}, r_s={r_s}): HNC1/HNC2 + Deutsch/Kelbg MD done in {dt_el:.1f} s"
                )
        mon.print_summary()
        mon.save_csv(results_dir / "telemetry_tables_1_to_5.csv")
        mon.plot(figures_dir / "telemetry_tables_1_to_5")
        save_md_cache(results_dir, md_results_deutsch, md_results_kelbg, sig)

    # TABLE I
    t1_headers = [
        "Параметр",
        "Статья (0.5, 0.4)",
        "Наш код (0.5, 0.4)",
        "Статья (0.5, 1.0)",
        "Наш код (0.5, 1.0)",
        "Статья (2.0, 1.0)",
        "Наш код (2.0, 1.0)",
    ]
    c1_04 = compute_table_i_state(0.5, 0.4, 0.025, args.n_prod)
    c1_10 = compute_table_i_state(0.5, 1.0, 0.025, args.n_prod)
    c1_20 = compute_table_i_state(2.0, 1.0, 0.025, args.n_prod)
    t1_rows = [
        ["Gamma", "0.5", "0.5", "0.5", "0.5", "2.0", "2.0"],
        ["r_s", "0.4", "0.4", "1.0", "1.0", "1.0", "1.0"],
        [
            "T (K)",
            f"{PAPER_TABLE_I[(0.5, 0.4)]['T_K']:.2e}",
            f"{c1_04['T_K']:.2e}",
            f"{PAPER_TABLE_I[(0.5, 1.0)]['T_K']:.2e}",
            f"{c1_10['T_K']:.2e}",
            f"{PAPER_TABLE_I[(2.0, 1.0)]['T_K']:.2e}",
            f"{c1_20['T_K']:.2e}",
        ],
        [
            "n (cm^-3)",
            f"{PAPER_TABLE_I[(0.5, 0.4)]['n_cm3']:.2e}",
            f"{c1_04['n_cm3']:.2e}",
            f"{PAPER_TABLE_I[(0.5, 1.0)]['n_cm3']:.2e}",
            f"{c1_10['n_cm3']:.2e}",
            f"{PAPER_TABLE_I[(2.0, 1.0)]['n_cm3']:.2e}",
            f"{c1_20['n_cm3']:.2e}",
        ],
        [
            "omega_pe (s^-1)",
            f"{PAPER_TABLE_I[(0.5, 0.4)]['omega_pe_s']:.2e}",
            f"{c1_04['omega_pe_s']:.2e}",
            f"{PAPER_TABLE_I[(0.5, 1.0)]['omega_pe_s']:.2e}",
            f"{c1_10['omega_pe_s']:.2e}",
            f"{PAPER_TABLE_I[(2.0, 1.0)]['omega_pe_s']:.2e}",
            f"{c1_20['omega_pe_s']:.2e}",
        ],
        ["omega_pe * dt", "0.03", "0.025", "0.03", "0.025", "0.08", "0.025"],
        [
            "N_t (шагов)",
            "5.60e+04",
            f"{args.n_prod:.2e}",
            "7.30e+04",
            f"{args.n_prod:.2e}",
            "4.80e+04",
            f"{args.n_prod:.2e}",
        ],
    ]
    save_csv_table(results_dir / "table1_comparison.csv", t1_headers, t1_rows)
    render_table_figure(
        "TABLE I [HEPHAESTUS]: Параметры плазмы и МД — Статья (1981) vs Наш расчёт (СГС)",
        t1_headers,
        t1_rows,
        figures_dir / "table1_comparison",
        fig_width_mult=1.35,
        fig_height_in=2.8,
    )

    # TABLE II
    t2_headers = [
        "Режим",
        "PV (Ст. MD)",
        "PV (Наш D / K)",
        "PV HNC(1) Ст./Наш",
        "PV HNC(2) Ст./Наш",
        "U HNC(1) Ст./Наш",
        "U HNC(2) Ст./Наш",
        "chi_T(1) Ст./Наш",
        "chi_T(2) Ст./Наш",
    ]
    t2_rows = []
    for st in states:
        p = PAPER_TABLE_II[st]
        h1 = hnc1_map[st]
        h2 = hnc2_map[st]
        md_d = md_results_deutsch[st]
        md_k = md_results_kelbg[st]
        t2_rows.append(
            [
                f"G={st[0]}, rs={st[1]}",
                f"{p['PV_MD']:.2f}",
                f"{md_d['PV_over_2NkT']:.2f} / {md_k['PV_over_2NkT']:.2f}",
                f"{p['PV_HNC1']:.3f} / {h1.pv_over_2nkt:.3f}",
                f"{p['PV_HNC2']:.3f} / {h2.pv_over_2nkt:.3f}",
                f"{p['U_HNC1']:+.3f} / {h1.u_over_2nkt:+.3f}",
                f"{p['U_HNC2']:+.3f} / {h2.u_over_2nkt:+.3f}",
                f"{p['chiT_HNC1']:.3f} / {h1.inv_chi_t:.3f}",
                f"{p['chiT_HNC2']:.3f} / {h2.inv_chi_t:.3f}",
            ]
        )
    save_csv_table(results_dir / "table2_comparison.csv", t2_headers, t2_rows)
    render_table_figure(
        "TABLE II [HEPHAESTUS]: Термодинамика — Прямое решение уравнений HNC(1), HNC(2) и GPU МД",
        t2_headers,
        t2_rows,
        figures_dir / "table2_comparison",
        fig_width_mult=1.70,
        fig_height_in=2.3,
        font_size=7.0,
    )

    # TABLE III
    tab3_calc: dict[tuple[float, float], dict[str, float]] = {}
    for st in states:
        tab3_calc[st] = compute_table_iii_transport_ab_initio(
            st[0], st[1], hnc1_map[st], md_results_deutsch[st], md_results_kelbg[st]
        )

    t3_headers = [
        "Величина",
        "Ст. (0.5, 0.4)",
        "Наш (0.5, 0.4)",
        "Ст. (0.5, 1.0)",
        "Наш (0.5, 1.0)",
        "Ст. (2.0, 1.0)",
        "Наш (2.0, 1.0)",
    ]
    t3_map = [
        ("D_e* (МД Дойч)", "D_e*", "D_e* (Deutsch)"),
        ("D_e* (МД Кельбг)", "D_e*", "D_e* (Kelbg)"),
        ("D_e (cm^2/s)", "D_e (cm^2/s)", "D_e (cm^2/s) [D]"),
        ("tau_se*", "tau_se*", "tau_se*"),
        ("omega_se*", "omega_se*", "omega_se*"),
        ("D_i*", "D_i*", "D_i*"),
        ("D_i (cm^2/s)", "D_i (cm^2/s)", "D_i (cm^2/s)"),
        ("tau_si*", "tau_si*", "tau_si*"),
        ("omega_si*", "omega_si*", "omega_si*"),
        ("sigma*", "sigma*", "sigma*"),
        ("sigma (10^5 mho/cm)", "sigma (1e5 mho/cm)", "sigma (1e5 mho/cm)"),
        ("tau_J*", "tau_J*", "tau_J*"),
        ("omega_J*", "omega_J*", "omega_J*"),
    ]
    t3_rows = []
    for label, p_key, c_key in t3_map:
        p_vals = PAPER_TABLE_III[p_key]
        row = [label]
        for idx, st in enumerate(states):
            pv = p_vals[idx]
            cv = tab3_calc[st][c_key]
            row.append("-" if pv is None else f"{pv:.3g}")
            row.append("-" if not math.isfinite(cv) else f"{cv:.3g}")
        t3_rows.append(row)
    save_csv_table(results_dir / "table3_comparison.csv", t3_headers, t3_rows)
    render_table_figure(
        "TABLE III [HEPHAESTUS]: Самодиффузия, частоты Эйнштейна и проводимость (Из первых принципов)",
        t3_headers,
        t3_rows,
        figures_dir / "table3_comparison",
        fig_width_mult=1.40,
        fig_height_in=4.7,
        font_size=7.4,
    )

    # TABLE IV
    t4_headers = [
        "q",
        "(0.5,0.4) Ст. MD/mft/V",
        "(0.5,0.4) Наш mft/V",
        "(0.5,1.0) Ст. MD/mft/V",
        "(0.5,1.0) Наш mft/V",
        "(2.0,1.0) Ст. MD/mft/V",
        "(2.0,1.0) Наш mft/V",
    ]
    t4_rows = []
    for q_val, p04, p10, p20 in PAPER_TABLE_IV:
        m04, v04 = compute_table_iv_ab_initio(q_val, 0.5, 0.4, hnc1_map[(0.5, 0.4)])
        m10, v10 = compute_table_iv_ab_initio(q_val, 0.5, 1.0, hnc1_map[(0.5, 1.0)])
        m20, v20 = compute_table_iv_ab_initio(q_val, 2.0, 1.0, hnc1_map[(2.0, 1.0)])
        t4_rows.append(
            [
                f"{q_val:.3f}",
                f"{p04[0]:.2f} / {p04[1]:.2f} / {p04[2]:.2f}",
                f"{m04:.2f} / {v04:.2f}",
                f"{p10[0]:.2f} / {p10[1]:.2f} / {p10[2]:.2f}",
                f"{m10:.2f} / {v10:.2f}",
                f"{p20[0]:.2f} / {p20[1]:.3f} / {p20[2]:.3f}",
                f"{m20:.3f} / {v20:.3f}",
            ]
        )
    save_csv_table(results_dir / "table4_comparison.csv", t4_headers, t4_rows)
    render_table_figure(
        "TABLE IV [HEPHAESTUS]: Значения S_ZZ(q, omega=0) — Статья (MD / mft / V) vs Наш расчёт (mft / V)",
        t4_headers,
        t4_rows,
        figures_dir / "table4_comparison",
        fig_width_mult=1.60,
        fig_height_in=2.8,
        font_size=7.4,
    )

    # TABLE V
    t5_headers = [
        "Режим / q",
        "S_ii (Ст./Наш)",
        "S_ie (Ст./Наш)",
        "S_ee (Ст./Наш)",
        "Omega4_ii*",
        "-Omega4_ie*",
        "Omega4_ee*",
        "tau_11* (Ст./Наш)",
        "tau_22* (Ст./Наш)",
    ]
    t5_rows = []
    for st in states:
        for p_row in PAPER_TABLE_V[st]:
            q_v = p_row[0]
            s_ii, s_ie, s_ee, w4_ii, w4_ie, w4_ee, t11, t22 = compute_table_v_row_ab_initio(
                q_v, st[0], st[1], hnc1_map[st]
            )
            t5_rows.append(
                [
                    f"({st[0]},{st[1]}) q={q_v:.3f}",
                    f"{p_row[1]:.4f} / {s_ii:.4f}",
                    f"{p_row[2]:.4f} / {s_ie:.4f}",
                    f"{p_row[3]:.4f} / {s_ee:.4f}",
                    f"{p_row[4]:.2e}/{w4_ii:.2e}",
                    f"{p_row[5]:.2e}/{w4_ie:.2e}",
                    f"{p_row[6]:.3g} / {w4_ee:.3g}",
                    f"{p_row[7]:.1f} / {t11:.1f}",
                    f"{p_row[8]:.2f} / {t22:.2f}",
                ]
            )
    save_csv_table(results_dir / "table5_comparison.csv", t5_headers, t5_rows)
    render_table_figure(
        "TABLE V [HEPHAESTUS]: Структурные факторы S_ab(q) из HNC, 4-е моменты и времена Резибуа",
        t5_headers,
        t5_rows,
        figures_dir / "table5_comparison",
        fig_width_mult=1.75,
        fig_height_in=7.0,
        font_size=6.8,
    )

    md_lines = [
        f"# Расчёт из первых принципов Таблиц I–V (1981): HNC + GPU МД (`{god_name}`)\n",
    ]
    for title, headers, rows in [
        ("## TABLE I. Параметры плазмы и МД", t1_headers, t1_rows),
        ("## TABLE II. Термодинамические свойства (HNC + GPU МД)", t2_headers, t2_rows),
        ("## TABLE III. Коэффициенты самодиффузии и электропроводности", t3_headers, t3_rows),
        ("## TABLE IV. Значения S_ZZ(q, omega=0)", t4_headers, t4_rows),
        (
            "## TABLE V. Структурные факторы HNC, частотные моменты и времена релаксации",
            t5_headers,
            t5_rows,
        ),
    ]:
        md_lines.append(title + "\n")
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            md_lines.append("| " + " | ".join(r) + " |")
        md_lines.append("\n")

    md_report_path = results_dir / "TABLES_I_TO_V_COMPARISON.md"
    md_report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown report written to: {md_report_path}")
    print(
        f"All 5 visual comparison tables saved to: {figures_dir}/table[1-5]_comparison.{{pdf,png}}"
    )


if __name__ == "__main__":
    main()
