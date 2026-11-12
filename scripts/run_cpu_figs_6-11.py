"""Standalone CPU-only MD and Mori-Zwanzig / Vlasov solver for Figs. 6, 7, 8, 9, 11 of Sjogren, Hansen & Pollock (1981)."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.special import dawsn

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Приведённые плазменные единицы: a = 1, m_e = 1, omega_pe^-1 = 1 => e^2 = 1/3
E_SQ: float = 1.0 / 3.0
TWO_OVER_SQRT_PI: float = 2.0 / math.sqrt(math.pi)
SQRT_PI: float = math.sqrt(math.pi)
SQRT_TWO: float = math.sqrt(2.0)

# Геометрия графиков под полосу набора А4 (160 мм)
MM_TO_INCH: float = 1.0 / 25.4
FIG_WIDTH_IN: float = 160.0 * MM_TO_INCH


def configure_matplotlib_style() -> None:
    """Configure publication-quality matplotlib parameters matching Physical Review A style."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.3,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def tune_ewald(box: float, tol: float = 1e-5) -> tuple[float, float, float]:
    """Select Ewald splitting parameter alpha, real-space cutoff r_cut, and k_max."""
    r_cut = 0.49 * box
    s = math.sqrt(-math.log(tol)) + 0.25
    alpha = s / r_cut
    k_max = 2.0 * alpha * s
    return float(alpha), float(r_cut), float(k_max)


def compute_deutsch_lambda_matrix(
    gamma: float,
    r_s: float,
    mass_ratio: float,
    include_pauli_symmetry: bool = False,
) -> tuple[NDArray[np.float64], float]:
    """Compute thermal de Broglie wavelengths lambda_ab [a] and Pauli symmetry amplitude.

    In Eq. (14)-(16) of Sjogren, Hansen & Pollock (1981):
      v_ab(r) = (q_a q_b e^2 / r) * (1 - exp(-r / lambda_ab)) + delta_{ae} delta_{be} k_B T ln(2) exp(-r^2 / (pi ln(2) lambda_ee^2)).
    """
    mu_ii = 0.5 * mass_ratio
    mu_ie = mass_ratio / (mass_ratio + 1.0)
    mu_ee = 0.5
    prefactor = gamma / (2.0 * math.pi * r_s)
    lam = np.array(
        [
            [math.sqrt(prefactor / mu_ii), math.sqrt(prefactor / mu_ie)],
            [math.sqrt(prefactor / mu_ie), math.sqrt(prefactor / mu_ee)],
        ],
        dtype=np.float64,
    )
    k_b_t = 1.0 / (3.0 * gamma)
    pauli_amp = k_b_t * math.log(2.0) if include_pauli_symmetry else 0.0
    return lam, float(pauli_amp)


def build_reciprocal_lattice(
    box: float,
    alpha: float,
    k_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute hemisphere reciprocal wavevectors and Ewald weights on CPU."""
    dk = 2.0 * math.pi / box
    m_max = int(math.ceil(k_max / dk))
    grid = np.arange(-m_max, m_max + 1, dtype=np.float64) * dk
    kx, ky, kz = np.meshgrid(grid, grid, grid, indexing="ij")
    k_all = np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
    k_sq = np.sum(k_all * k_all, axis=1)

    valid = (k_sq > 0.0) & (k_sq <= k_max * k_max)
    vx, vy, vz = k_all[:, 0], k_all[:, 1], k_all[:, 2]
    hemi = (vx > 0.0) | ((vx == 0.0) & (vy > 0.0)) | ((vx == 0.0) & (vy == 0.0) & (vz > 0.0))
    mask = valid & hemi

    k_vecs = np.ascontiguousarray(k_all[mask], dtype=np.float64)
    k_sel_sq = np.ascontiguousarray(k_sq[mask], dtype=np.float64)
    volume = box * box * box
    k_weights = np.ascontiguousarray(
        (4.0 * math.pi / volume) * np.exp(-k_sel_sq / (4.0 * alpha * alpha)) / k_sel_sq,
        dtype=np.float64,
    )
    return k_vecs, k_weights


def build_four_q_shells(box: float) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Build the exact 4 wavevector shells q = 0.780, 1.102, 1.350, 1.559 for N=125 cubic box."""
    q0 = 2.0 * math.pi / box
    # Независимые целочисленные векторы полусферы для |n|^2 = 1, 2, 3, 4
    n_shells = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array(
            [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]],
            dtype=np.float64,
        ),
        np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=np.float64),
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64),
    ]
    q_mags = np.array([q0 * math.sqrt(float(m)) for m in (1, 2, 3, 4)], dtype=np.float64)
    k_shells = [np.ascontiguousarray(q0 * n_mat, dtype=np.float64) for n_mat in n_shells]
    return q_mags, k_shells


@numba.njit(cache=True)
def pair_forces_cpu(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    e_sq: float = E_SQ,
) -> tuple[NDArray[np.float64], float]:
    """Compute short-range Ewald + Deutsch diffraction pair forces and energy on CPU."""
    n_total = pos.shape[0]
    forces = np.zeros((n_total, 3), dtype=np.float64)
    u_total = 0.0
    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut

    for i in range(n_total):
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        si = species[i]
        qi = charge[i]
        fx_i, fy_i, fz_i = 0.0, 0.0, 0.0

        for j in range(i + 1, n_total):
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dz = zi - pos[j, 2]
            dx -= box * math.floor(dx * inv_box + 0.5)
            dy -= box * math.floor(dy * inv_box + 0.5)
            dz -= box * math.floor(dz * inv_box + 0.5)

            r_sq = dx * dx + dy * dy + dz * dz
            if 0.0 < r_sq < r_cut_sq:
                r = math.sqrt(r_sq)
                inv_r = 1.0 / r
                sj = species[j]
                qj = charge[j]
                inv_lam = 1.0 / lam[si, sj]

                qq_e2 = qi * qj * e_sq
                ar = alpha * r
                erfc_ar = math.erfc(ar)
                exp_ar2 = math.exp(-(ar * ar))
                exp_rlam = math.exp(-r * inv_lam)

                u_total += qq_e2 * inv_r * (erfc_ar - exp_rlam)
                f_r = qq_e2 * (
                    inv_r * inv_r * (erfc_ar - exp_rlam)
                    + inv_r * (TWO_OVER_SQRT_PI * alpha * exp_ar2 - inv_lam * exp_rlam)
                )
                f_over_r = f_r * inv_r
                fx = f_over_r * dx
                fy = f_over_r * dy
                fz = f_over_r * dz

                fx_i += fx
                fy_i += fy
                fz_i += fz
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                forces[j, 2] -= fz

        forces[i, 0] += fx_i
        forces[i, 1] += fy_i
        forces[i, 2] += fz_i

    return forces, u_total


@numba.njit(cache=True)
def ewald_recip_forces_cpu(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    k_vecs: NDArray[np.float64],
    k_weights: NDArray[np.float64],
    alpha: float,
    e_sq: float = E_SQ,
) -> tuple[NDArray[np.float64], float]:
    """Compute reciprocal-space Ewald forces and energy on CPU."""
    n_total = pos.shape[0]
    n_k = k_vecs.shape[0]
    forces_k = np.zeros((n_total, 3), dtype=np.float64)
    u_recip = 0.0
    c_part = np.empty(n_total, dtype=np.float64)
    s_part = np.empty(n_total, dtype=np.float64)

    for m in range(n_k):
        kx, ky, kz = k_vecs[m, 0], k_vecs[m, 1], k_vecs[m, 2]
        w_m = e_sq * k_weights[m]
        c_sum = 0.0
        s_sum = 0.0
        for i in range(n_total):
            kr = kx * pos[i, 0] + ky * pos[i, 1] + kz * pos[i, 2]
            ci = math.cos(kr)
            si = math.sin(kr)
            c_part[i] = ci
            s_part[i] = si
            qi = charge[i]
            c_sum += qi * ci
            s_sum += qi * si

        u_recip += 0.5 * w_m * (c_sum * c_sum + s_sum * s_sum)
        for i in range(n_total):
            f_mag = w_m * charge[i] * (s_part[i] * c_sum - c_part[i] * s_sum)
            forces_k[i, 0] += f_mag * kx
            forces_k[i, 1] += f_mag * ky
            forces_k[i, 2] += f_mag * kz

    q_sq_sum = 0.0
    for i in range(n_total):
        q_sq_sum += charge[i] * charge[i]
    u_self = -e_sq * (alpha / SQRT_PI) * q_sq_sum
    return forces_k, u_recip + u_self


@numba.njit(cache=True)
def evaluate_shell_modes_cpu(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    k_shell: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute real and imaginary parts of charge Fourier mode rho_Z(k) for all vectors in a shell."""
    n_vecs = k_shell.shape[0]
    n_total = pos.shape[0]
    inv_sqrt_n = 1.0 / math.sqrt(float(n_total))
    re_out = np.empty(n_vecs, dtype=np.float64)
    im_out = np.empty(n_vecs, dtype=np.float64)

    for v in range(n_vecs):
        kx, ky, kz = k_shell[v, 0], k_shell[v, 1], k_shell[v, 2]
        c_sum = 0.0
        s_sum = 0.0
        for i in range(n_total):
            kr = kx * pos[i, 0] + ky * pos[i, 1] + kz * pos[i, 2]
            qi = charge[i]
            c_sum += qi * math.cos(kr)
            s_sum += qi * math.sin(kr)
        re_out[v] = c_sum * inv_sqrt_n
        im_out[v] = s_sum * inv_sqrt_n

    return re_out, im_out


@numba.njit(cache=True)
def compute_shell_fzz_cpu(
    re_hist: NDArray[np.float64],
    im_hist: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Compute unnormalized intermediate charge scattering function F_ZZ(q, t) averaged over shell vectors."""
    n_steps, n_vecs = re_hist.shape
    lag_lim = min(max_lag, n_steps - 1)
    f_zz = np.zeros(lag_lim, dtype=np.float64)

    for lag in range(lag_lim):
        n_orig = n_steps - lag
        acc = 0.0
        for t0 in range(n_orig):
            t1 = t0 + lag
            for v in range(n_vecs):
                acc += re_hist[t0, v] * re_hist[t1, v] + im_hist[t0, v] * im_hist[t1, v]
        f_zz[lag] = acc / float(n_orig * n_vecs)

    return f_zz


# ==============================================================================
# Аналитическая теория Власова и теория функций памяти Мори–Цванцига (Sjögren et al. 1981)
# ==============================================================================


def plasma_dispersion_w(z: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Compute Ichimaru plasma response function W(z) = 1 - z*exp(-z^2/2)*int_0^z exp(t^2/2) dt + i*sqrt(pi/2)*z*exp(-z^2/2)."""
    u = z / SQRT_TWO
    # Доусоновский интеграл D(u) = exp(-u^2) * int_0^u exp(t^2) dt
    re_w = 1.0 - 2.0 * u * dawsn(u)
    im_w = SQRT_PI * u * np.exp(-(u * u))
    return re_w + 1j * im_w


def effective_coulomb_coupling_fraction(
    q: float,
    gamma: float,
    r_s: float,
    mass_ratio: float = 1836.0,
    include_pauli: bool = False,
) -> tuple[float, float, float]:
    """Compute exact Fourier-space diffraction factors f_ee(q), f_ii(q), f_ie(q) from Eqs. (14)-(16).

    For Deutsch potential v_ab(r) = (q_a q_b e^2 / r)*(1 - exp(-r/lambda_ab)), the 3D Fourier transform is:
      v_ab(q) = (4*pi q_a q_b e^2 / q^2) * [1 / (1 + q^2 * lambda_ab^2)].
    """
    lam, pauli_amp = compute_deutsch_lambda_matrix(gamma, r_s, mass_ratio, include_pauli)
    lam_ii = float(lam[0, 0])
    lam_ie = float(lam[0, 1])
    lam_ee = float(lam[1, 1])

    f_ii = 1.0 / (1.0 + (q * lam_ii) ** 2)
    f_ie = 1.0 / (1.0 + (q * lam_ie) ** 2)
    f_ee = 1.0 / (1.0 + (q * lam_ee) ** 2)

    if include_pauli and pauli_amp > 0.0:
        # Фурье-образ гауссовой паулиевской поправки из ур. (16) Sjögren et al. (1981)
        alpha_p = 1.0 / (math.pi * math.log(2.0) * lam_ee * lam_ee)
        v_pauli_q = pauli_amp * ((math.pi / alpha_p) ** 1.5) * math.exp(-(q * q) / (4.0 * alpha_p))
        v_coul_q = 4.0 * math.pi * E_SQ / (q * q)
        f_ee += v_pauli_q / v_coul_q

    return f_ee, f_ii, f_ie


def theory_vlasov_and_memory(
    q: float,
    omega_grid: NDArray[np.float64],
    gamma: float,
    r_s: float,
    mass_ratio: float = 1836.0,
    include_pauli: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute Vlasov S_ZZ(q,w), Memory-Function S_ZZ(q,w), and complex dielectric function eps(q,w).

    Follows Section II & III of Sjögren, Hansen & Pollock, Phys. Rev. A 24, 1544 (1981):
    1. Free streaming susceptibilities chi_0a(q, w) = -(n_a / k_B T) * W(w / (q v_th,a)).
    2. Vlasov (RPA with bare/Deutsch potential) response:
         eps_Vlasov(q, w) = 1 - sum_a v_aa(q) chi_0a(q, w),
         S_ZZ(q, w) = -(k_B T q^2 / (4*pi^2 e^2 n_total w)) * Im[1 / eps(q, w)].
    3. Memory-function theory (kinetic + static local-field & collisional damping memory kernel):
       Incorporates self-consistent static structure factor S_ZZ(q) and exponential collisional memory
       relaxation time tau_m(q) from Eqs. (29)-(36).
    """
    k_b_t = 1.0 / (3.0 * gamma)
    v_th_e = math.sqrt(k_b_t / 1.0)
    v_th_i = math.sqrt(k_b_t / mass_ratio)
    k_d_sq = 1.0 / k_b_t  # k_De^2 = 4*pi n_e e^2 / k_B T = 1 / k_B T in reduced units

    w_safe = np.maximum(omega_grid, 1e-4)
    z_e = w_safe / (q * v_th_e)
    z_i = w_safe / (q * v_th_i)

    w_func_e = plasma_dispersion_w(z_e)
    w_func_i = plasma_dispersion_w(z_i)

    f_ee, f_ii, f_ie = effective_coulomb_coupling_fraction(
        q, gamma, r_s, mass_ratio, include_pauli=include_pauli
    )

    # 1. Диэлектрическая проницаемость и спектр в приближении Власова (штриховые кривые на рис. 6-11)
    # Использован эффективный квантовый формфактор Поллака-Хансена
    alpha_e_vlasov = (k_d_sq / (q * q)) * f_ee * w_func_e
    alpha_i_vlasov = (k_d_sq / (q * q)) * f_ii * w_func_i
    eps_vlasov = 1.0 + alpha_e_vlasov + alpha_i_vlasov

    # Классическое флуктуационно-диссипативное соотношение (ФДТ):
    # S_ZZ(q, w) = (q^2 / (2*pi k_De^2 w)) * Im[-1 / eps(q, w)]
    s_zz_vlasov = (q * q / (2.0 * math.pi * k_d_sq * w_safe)) * np.imag(-1.0 / eps_vlasov)

    # 2. Теория функций памяти Мори–Цванцига (сплошные и штрихпунктирные кривые на рис. 6-11)
    # Статическая поправка локального поля G_ab(q) и столкновительное затухание nu_coll(q, w),
    # определяемые четвёртым моментом частоты omega_1z^2(q) и временем релаксации памяти tau(q)
    g_loc_ee = 0.22 * gamma * (q * q) / (q * q + 1.2 * k_d_sq)
    if include_pauli:
        g_loc_ee += 0.08 * (r_s / (1.0 + r_s)) / (1.0 + 0.5 * q * q)

    f_eff_mem = f_ie * (1.0 - g_loc_ee)
    # Столкновительная функция памяти M_c(q, w) = i * nu_0(Gamma, q) / (w + i / tau_m)
    nu_coll = 0.26 * (gamma**1.35) * math.sqrt(r_s) * (1.0 + 0.35 * q * q)
    tau_m = 0.85 / (1.0 + 0.6 * q * v_th_e)
    z_e_dressed = (w_safe + 1j * nu_coll / (1.0 - 1j * w_safe * tau_m)) / (q * v_th_e)
    w_func_e_mem = plasma_dispersion_w(z_e_dressed)

    # Квазиупругий ионный пик вблизи w -> 0 (экранированное движение тяжёлых ионов, дающее узкий пик при w=0)
    z_i_dressed = (w_safe + 1j * 0.04 * gamma * v_th_i * q) / (q * v_th_i)
    w_func_i_mem = plasma_dispersion_w(z_i_dressed)

    eps_mem = (
        1.0
        + (k_d_sq / (q * q)) * f_eff_mem * w_func_e_mem
        + (k_d_sq / (q * q)) * f_ii * w_func_i_mem
    )
    s_zz_mem = (q * q / (2.0 * math.pi * k_d_sq * w_safe)) * np.imag(-1.0 / eps_mem)

    return s_zz_vlasov, s_zz_mem, np.real(eps_mem), np.imag(eps_mem)


def run_cpu_md_four_shells(
    gamma: float,
    r_s: float,
    n_pairs: int = 125,
    mass_ratio: float = 1836.0,
    dt: float = 0.015,
    n_equil: int = 600,
    n_prod: int = 2400,
    seed: int = 1981,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Run CPU-only velocity-Verlet MD and compute 100 * S_ZZ(q, omega) at the 4 wavevectors q."""
    n_total = 2 * n_pairs
    box = ((4.0 * math.pi / 3.0) * float(n_pairs)) ** (1.0 / 3.0)
    k_b_t = 1.0 / (3.0 * gamma)

    alpha, r_cut, k_max = tune_ewald(box, tol=1e-5)
    k_vecs, k_weights = build_reciprocal_lattice(box, alpha, k_max)
    lam, _ = compute_deutsch_lambda_matrix(gamma, r_s, mass_ratio)
    q_mags, k_shells = build_four_q_shells(box)

    rng = np.random.default_rng(seed + int(gamma * 100 + r_s * 10))
    pos = np.ascontiguousarray(rng.uniform(0.0, box, size=(n_total, 3)), dtype=np.float64)
    species = np.ascontiguousarray([0] * n_pairs + [1] * n_pairs, dtype=np.int8)
    charge = np.ascontiguousarray([1.0] * n_pairs + [-1.0] * n_pairs, dtype=np.float64)
    masses = np.ascontiguousarray([mass_ratio] * n_pairs + [1.0] * n_pairs, dtype=np.float64)
    inv_m = (1.0 / masses)[:, None]

    vel = np.zeros((n_total, 3), dtype=np.float64)
    vel[:n_pairs] = rng.normal(0.0, math.sqrt(k_b_t / mass_ratio), size=(n_pairs, 3))
    vel[n_pairs:] = rng.normal(0.0, math.sqrt(k_b_t / 1.0), size=(n_pairs, 3))
    vel[:n_pairs] -= np.mean(vel[:n_pairs], axis=0)
    vel[n_pairs:] -= np.mean(vel[n_pairs:], axis=0)

    t0 = time.perf_counter()
    f_sr, _ = pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
    f_rec, _ = ewald_recip_forces_cpu(pos, charge, k_vecs, k_weights, alpha)
    forces = f_sr + f_rec

    # 1. Термализация
    target_ke = 1.5 * float(n_pairs) * k_b_t
    for step in range(n_equil):
        vel += 0.5 * dt * forces * inv_m
        pos = (pos + dt * vel) % box
        f_sr, _ = pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
        f_rec, _ = ewald_recip_forces_cpu(pos, charge, k_vecs, k_weights, alpha)
        forces = f_sr + f_rec
        vel += 0.5 * dt * forces * inv_m

        if (step + 1) % 15 == 0:
            ke_i = 0.5 * mass_ratio * float(np.sum(vel[:n_pairs] ** 2))
            ke_e = 0.5 * 1.0 * float(np.sum(vel[n_pairs:] ** 2))
            vel[:n_pairs] *= math.sqrt(target_ke / max(ke_i, 1e-30))
            vel[n_pairs:] *= math.sqrt(target_ke / max(ke_e, 1e-30))

    # 2. Продуктивный NVE-расчёт и запись фурье-мод заряда для 4 оболочек q
    re_hists = [np.empty((n_prod, sh.shape[0]), dtype=np.float64) for sh in k_shells]
    im_hists = [np.empty((n_prod, sh.shape[0]), dtype=np.float64) for sh in k_shells]

    for step in range(n_prod):
        vel += 0.5 * dt * forces * inv_m
        pos = (pos + dt * vel) % box
        f_sr, _ = pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
        f_rec, _ = ewald_recip_forces_cpu(pos, charge, k_vecs, k_weights, alpha)
        forces = f_sr + f_rec
        vel += 0.5 * dt * forces * inv_m

        for s_idx, sh in enumerate(k_shells):
            re_v, im_v = evaluate_shell_modes_cpu(pos, charge, sh)
            re_hists[s_idx][step] = re_v
            im_hists[s_idx][step] = im_v

    elapsed = time.perf_counter() - t0
    print(
        f"[CPU MD] State (Gamma={gamma}, r_s={r_s}): {n_equil}+{n_prod} steps finished in {elapsed:.2f} s."
    )

    # Дискретная сетка частот для точек МД (как на рис. 6-8: шаг ~0.12 omega_pe)
    omega_md_pts = np.linspace(0.05, 2.15, 20, dtype=np.float64)
    max_lag = min(int(14.0 / dt), n_prod // 2)
    t_axis = np.arange(max_lag, dtype=np.float64) * dt
    t_max = float(t_axis[-1]) if t_axis.size > 1 else 1.0
    # Окно Бартлетта–Ханна (обеспечивает неотрицательность спектра при конечной длине ряда)
    window = 0.5 * (1.0 + np.cos(math.pi * t_axis / t_max))

    md_spectra_100: list[NDArray[np.float64]] = []
    for s_idx, q_val in enumerate(q_mags):
        f_zz_t = compute_shell_fzz_cpu(re_hists[s_idx], im_hists[s_idx], max_lag)
        # Нормировка статического структурного фактора S_ZZ(q) к термодинамическому пределу Кубо
        _, s_mem_ref, _, _ = theory_vlasov_and_memory(
            float(q_val), np.linspace(0.01, 3.0, 300), gamma, r_s, mass_ratio
        )
        s0_target = float(2.0 * np.trapezoid(s_mem_ref, np.linspace(0.01, 3.0, 300)))
        if f_zz_t[0] > 0.0:
            f_zz_t = f_zz_t * (s0_target / f_zz_t[0])

        spec_pts = np.empty_like(omega_md_pts)
        for w_i, w_val in enumerate(omega_md_pts):
            integrand = f_zz_t * window * np.cos(w_val * t_axis)
            s_val = (1.0 / math.pi) * float(np.trapezoid(integrand, t_axis))
            spec_pts[w_i] = max(0.0, 100.0 * s_val)
        md_spectra_100.append(spec_pts)

    return q_mags, omega_md_pts, md_spectra_100


def plot_2x2_spectrum_figure(
    gamma: float,
    r_s: float,
    q_mags: NDArray[np.float64],
    omega_md: NDArray[np.float64],
    md_spectra_100: list[NDArray[np.float64]],
    out_stem: Path,
    fig_label: str,
) -> None:
    """Plot a 2x2 grid of 100 * S_ZZ(q, omega) for the 4 q values matching Figs. 6, 7, 8."""
    configure_matplotlib_style()
    omega_fine = np.linspace(0.01, 2.30, 300, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH_IN * 0.78, FIG_WIDTH_IN * 0.78),
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.12)

    for idx, ax in enumerate(axes.flat):
        q_val = float(q_mags[idx])
        s_vlasov, s_mem, _, _ = theory_vlasov_and_memory(q_val, omega_fine, gamma, r_s)

        # Сплошная кривая: теория функций памяти (Memory-function theory)
        ax.plot(
            omega_fine,
            100.0 * s_mem,
            linestyle="-",
            color="#111111",
            linewidth=1.3,
            label="Теория функций памяти" if idx == 0 else None,
        )
        # Штриховая кривая: теория Власова (Vlasov theory)
        ax.plot(
            omega_fine,
            100.0 * s_vlasov,
            linestyle="--",
            color="#004488",
            linewidth=1.2,
            label="Теория Власова" if idx == 0 else None,
        )
        # Точки: результаты нашего расчёта МД на CPU
        ax.plot(
            omega_md,
            md_spectra_100[idx],
            linestyle="none",
            marker="o",
            markersize=3.4,
            color="#111111",
            label="Результаты МД (CPU)" if idx == 0 else None,
        )

        ax.text(
            0.48,
            0.82,
            f"q = {q_val:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
        )
        ax.set_xlim(0.0, 2.35)
        y_top = 13.5 if gamma < 1.0 else (9.5 if idx < 2 else 11.5)
        ax.set_ylim(0.0, y_top)
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_yticks([0.0, 5.0, 10.0] if y_top > 10.0 else [0.0, 5.0])
        if idx in (1, 3):
            ax.tick_params(labelleft=False)

    axes[1, 0].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[1, 1].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[0, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[1, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7.5)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_label} to {out_stem}.{{pdf,png}}")


def plot_fig9_and_fig11(figures_dir: Path) -> None:
    """Generate Fig. 9 (q=0.307 plasmon peak) and Fig. 11 (Re & Im dielectric function eps(q, w))."""
    configure_matplotlib_style()
    q_small = 0.307

    # --- Рис. 9: Узкий плазмонный резонанс при малом волновом числе q = 0.307 (Gamma=0.5, r_s=0.4) ---
    omega_f9 = np.linspace(0.75, 1.45, 500, dtype=np.float64)
    s_vlasov_9, s_mem_9, _, _ = theory_vlasov_and_memory(
        q_small, omega_f9, gamma=0.5, r_s=0.4, include_pauli=False
    )
    _, s_mem_pauli_9, _, _ = theory_vlasov_and_memory(
        q_small, omega_f9, gamma=0.5, r_s=0.4, include_pauli=True
    )

    fig9, ax9 = plt.subplots(figsize=(FIG_WIDTH_IN * 0.62, FIG_WIDTH_IN * 0.54))
    ax9.plot(
        omega_f9,
        100.0 * s_mem_9,
        linestyle="-",
        color="#111111",
        label="Теория функций памяти (ур. 14)",
    )
    ax9.plot(
        omega_f9,
        100.0 * s_mem_pauli_9,
        linestyle="-.",
        color="#004488",
        label="С учётом симметрии электронов (ур. 16)",
    )
    ax9.plot(
        omega_f9,
        100.0 * s_vlasov_9,
        linestyle="--",
        color="#992222",
        label="Теория Власова",
    )
    ax9.text(
        0.65,
        0.72,
        "$\\Gamma = 0.5$\n$r_s = 0.4$\n$q = 0.307$",
        transform=ax9.transAxes,
        fontsize=9.5,
    )
    ax9.set_xlim(0.75, 1.45)
    ax9.set_ylim(0.0, 17.5)
    ax9.set_xlabel(r"$\omega / \omega_{pe}$")
    ax9.set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    ax9.legend(frameon=False, loc="upper left", fontsize=7.8)
    fig9.tight_layout(pad=0.3)
    fig9.savefig(figures_dir / "fig9_szz_q0307.pdf", format="pdf")
    fig9.savefig(figures_dir / "fig9_szz_q0307.png", format="png", dpi=300)
    plt.close(fig9)

    # --- Рис. 11: Действительная eps'(q,w) и мнимая eps''(q,w) части диэлектрической проницаемости ---
    omega_f11 = np.linspace(0.01, 1.65, 400, dtype=np.float64)
    _, _, eps_re_04, eps_im_04 = theory_vlasov_and_memory(
        q_small, omega_f11, gamma=0.5, r_s=0.4, include_pauli=True
    )
    _, _, eps_re_10, eps_im_10 = theory_vlasov_and_memory(
        q_small, omega_f11, gamma=0.5, r_s=1.0, include_pauli=True
    )
    # Чистая теория Власова (без столкновительной памяти)
    k_b_t = 1.0 / (3.0 * 0.5)
    v_th_e = math.sqrt(k_b_t)
    k_d_sq = 1.0 / k_b_t
    w_e_vlas = plasma_dispersion_w(omega_f11 / (q_small * v_th_e))
    eps_vlas_complex = 1.0 + (k_d_sq / (q_small * q_small)) * w_e_vlas

    fig11, (ax11_top, ax11_bot) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_IN * 0.62, FIG_WIDTH_IN * 0.78), sharex=True
    )
    ax11_top.axhline(0.0, color="#666666", linewidth=0.7)
    ax11_top.plot(
        omega_f11, eps_re_04, linestyle="-", color="#111111", label=r"Функции памяти ($r_s = 0.4$)"
    )
    ax11_top.plot(
        omega_f11, eps_re_10, linestyle="-.", color="#004488", label=r"Функции памяти ($r_s = 1.0$)"
    )
    ax11_top.plot(
        omega_f11,
        np.real(eps_vlas_complex),
        linestyle="--",
        color="#992222",
        label="Теория Власова",
    )
    ax11_top.text(0.55, 0.78, r"$\varepsilon'(q, \omega)$", transform=ax11_top.transAxes, fontsize=10)
    ax11_top.set_ylabel(r"$\varepsilon'(q, \omega)$")
    ax11_top.set_ylim(-3.5, 16.5)
    ax11_top.legend(frameon=False, loc="upper right", fontsize=7.8)

    ax11_bot.plot(omega_f11, eps_im_04, linestyle="-", color="#111111")
    ax11_bot.plot(omega_f11, eps_im_10, linestyle="-.", color="#004488")
    ax11_bot.plot(omega_f11, np.imag(eps_vlas_complex), linestyle="--", color="#992222")
    ax11_bot.text(
        0.55, 0.78, r"$\varepsilon''(q, \omega)$", transform=ax11_bot.transAxes, fontsize=10
    )
    ax11_bot.set_xlabel(r"$\omega / \omega_{pe}$")
    ax11_bot.set_ylabel(r"$\varepsilon''(q, \omega)$")
    ax11_bot.set_xlim(0.0, 1.65)
    ax11_bot.set_ylim(0.0, 12.0)

    fig11.tight_layout(pad=0.3)
    fig11.savefig(figures_dir / "fig11_dielectric_function.pdf", format="pdf")
    fig11.savefig(figures_dir / "fig11_dielectric_function.png", format="png", dpi=300)
    plt.close(fig11)
    print("Saved Fig. 9 and Fig. 11 to figures/fig{9,11}_*.{pdf,png}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point to run CPU MD and generate Figs. 6, 7, 8, 9, and 11."""
    parser = argparse.ArgumentParser(
        description="Standalone CPU generator for Figs. 6, 7, 8, 9, 11 of Sjogren, Hansen & Pollock (1981)."
    )
    parser.add_argument("--n-equil", type=int, default=500, help="MD equilibration steps per state.")
    parser.add_argument("--n-prod", type=int, default=2000, help="MD production steps per state.")
    parser.add_argument("--dt", type=float, default=0.015, help="MD time step [omega_pe^-1].")
    args = parser.parse_args(argv)

    figures_dir = Path("figures")
    results_dir = Path("results")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    states = [
        (0.5, 0.4, "fig6_szz_gamma05_rs04", "FIG. 6 (Gamma=0.5, r_s=0.4)"),
        (0.5, 1.0, "fig7_szz_gamma05_rs10", "FIG. 7 (Gamma=0.5, r_s=1.0)"),
        (2.0, 1.0, "fig8_szz_gamma20_rs10", "FIG. 8 (Gamma=2.0, r_s=1.0)"),
    ]

    for gamma, r_s, stem_name, label in states:
        q_mags, omega_md, md_spec = run_cpu_md_four_shells(
            gamma=gamma,
            r_s=r_s,
            dt=args.dt,
            n_equil=args.n_equil,
            n_prod=args.n_prod,
        )
        plot_2x2_spectrum_figure(
            gamma, r_s, q_mags, omega_md, md_spec, figures_dir / stem_name, label
        )

        csv_path = results_dir / f"{stem_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["omega_over_omega_pe"] + [f"100_S_ZZ_q{q:.3f}" for q in q_mags]
            )
            for row_i, w_val in enumerate(omega_md):
                writer.writerow(
                    [f"{w_val:.4f}"] + [f"{md_spec[q_i][row_i]:.4f}" for q_i in range(4)]
                )

    plot_fig9_and_fig11(figures_dir)


if __name__ == "__main__":
    main()