"""Standalone CPU-only molecular dynamics simulation and plotting for Figures 6, 7, and 8 of Hansen
& McDonald (1981).
"""

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

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Физические и математические константы в приведённых плазменных единицах (a = 1, m_e = 1,
# omega_pe^-1 = 1)
E_SQ: float = 1.0 / 3.0
TWO_OVER_SQRT_PI: float = 2.0 / math.sqrt(math.pi)

# Параметры вёрстки под полосу набора А4 (160 мм)
MM_TO_INCH: float = 1.0 / 25.4
FIG_WIDTH_IN: float = 160.0 * MM_TO_INCH
FIG_HEIGHT_IN: float = 100.0 * MM_TO_INCH


def configure_matplotlib_style() -> None:
    """Configure publication-quality matplotlib parameters (160 mm width, 10 pt font,
    B/W-friendly).
    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
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


def tune_ewald(
    box: float,
    tol: float = 1e-5,
    alpha: float | None = None,
) -> tuple[float, float, float]:
    """Select optimized Ewald parameters (alpha, r_cut, k_max) for target tolerance."""
    if tol <= 0.0 or tol >= 1.0:
        raise ValueError(f"Tolerance must be in (0, 1), got {tol}.")
    r_cut = 0.49 * box
    s = math.sqrt(-math.log(tol)) + 0.25
    if alpha is None:
        alpha = s / r_cut
    k_max = 2.0 * alpha * s
    return float(alpha), float(r_cut), float(k_max)


def compute_deutsch_lambda_matrix(
    gamma: float,
    r_s: float,
    mass_ratio: float,
) -> NDArray[np.float64]:
    """Compute the 2x2 matrix of thermal de Broglie wavelengths lambda_ab [a]."""
    mu_ii = 0.5 * mass_ratio
    mu_ie = mass_ratio / (mass_ratio + 1.0)
    mu_ee = 0.5
    prefactor = gamma / (2.0 * math.pi * r_s)
    return np.array(
        [
            [math.sqrt(prefactor / mu_ii), math.sqrt(prefactor / mu_ie)],
            [math.sqrt(prefactor / mu_ie), math.sqrt(prefactor / mu_ee)],
        ],
        dtype=np.float64,
    )


def build_reciprocal_lattice(
    box: float,
    alpha: float,
    k_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute hemisphere reciprocal wavevectors k_vecs and weights for Ewald sum."""
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
) -> tuple[NDArray[np.float64], float, float]:
    """Compute short-range Ewald + Deutsch diffraction pair forces, energy, and virial on CPU."""
    n_total = pos.shape[0]
    forces = np.zeros((n_total, 3), dtype=np.float64)
    u_total = 0.0
    w_total = 0.0

    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut

    for i in range(n_total):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        si = species[i]
        qi = charge[i]

        fx_i = 0.0
        fy_i = 0.0
        fz_i = 0.0

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
                lam_ij = lam[si, sj]
                inv_lam = 1.0 / lam_ij

                qq_e2 = qi * qj * e_sq
                ar = alpha * r
                erfc_ar = math.erfc(ar)
                exp_ar2 = math.exp(-(ar * ar))
                exp_rlam = math.exp(-r * inv_lam)

                # Потенциал: v_sr(r) = (q_i q_j e^2 / r) * [erfc(alpha * r) - exp(-r / lam_ij)]
                u_pair = qq_e2 * inv_r * (erfc_ar - exp_rlam)
                u_total += u_pair

                # Радиальная сила F_r = -dv_sr/dr
                f_r = qq_e2 * (
                    inv_r * inv_r * (erfc_ar - exp_rlam)
                    + inv_r * (TWO_OVER_SQRT_PI * alpha * exp_ar2 - inv_lam * exp_rlam)
                )
                w_total += f_r * r

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

    return forces, u_total, w_total


@numba.njit(cache=True)
def ewald_recip_forces_cpu(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    k_vecs: NDArray[np.float64],
    k_weights: NDArray[np.float64],
    alpha: float,
    e_sq: float = E_SQ,
) -> tuple[NDArray[np.float64], float]:
    """Compute Ewald reciprocal-space forces and energy (including self-term) purely on CPU."""
    n_total = pos.shape[0]
    n_k = k_vecs.shape[0]
    forces_k = np.zeros((n_total, 3), dtype=np.float64)
    u_recip = 0.0

    c_part = np.empty(n_total, dtype=np.float64)
    s_part = np.empty(n_total, dtype=np.float64)

    for m in range(n_k):
        kx = k_vecs[m, 0]
        ky = k_vecs[m, 1]
        kz = k_vecs[m, 2]
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
    u_self = -e_sq * (alpha / math.sqrt(math.pi)) * q_sq_sum

    return forces_k, u_recip + u_self


def compute_total_forces_cpu(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    k_vecs: NDArray[np.float64],
    k_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Evaluate total forces (short-range Deutsch/Ewald + reciprocal Ewald) on CPU."""
    f_sr, u_sr, _ = pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
    f_rec, u_rec = ewald_recip_forces_cpu(pos, charge, k_vecs, k_weights, alpha)
    return f_sr + f_rec, float(u_sr + u_rec)


@numba.njit(cache=True)
def compute_time_correlations_cpu(
    vel_e: NDArray[np.float64],
    vel_i: NDArray[np.float64],
    j_vec: NDArray[np.float64],
    rho_z_re: NDArray[np.float64],
    rho_z_im: NDArray[np.float64],
    max_lag_e: int,
    max_lag_i: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Compute normalized Z_e(t), J(t), Z_i(t), and F_ZZ(k, t) via multi-origin averaging on CPU."""
    n_frames = vel_e.shape[0]
    n_pairs = vel_e.shape[1]

    lag_e = min(max_lag_e, n_frames - 1)
    lag_i = min(max_lag_i, n_frames - 1)

    z_e = np.zeros(lag_e, dtype=np.float64)
    j_acf = np.zeros(lag_e, dtype=np.float64)
    f_zz = np.zeros(lag_e, dtype=np.float64)
    z_i = np.zeros(lag_i, dtype=np.float64)

    for lag in range(lag_e):
        n_orig = n_frames - lag
        sum_ve = 0.0
        sum_j = 0.0
        sum_fzz = 0.0
        for t0 in range(n_orig):
            t1 = t0 + lag
            sum_j += (
                j_vec[t0, 0] * j_vec[t1, 0]
                + j_vec[t0, 1] * j_vec[t1, 1]
                + j_vec[t0, 2] * j_vec[t1, 2]
            )
            sum_fzz += rho_z_re[t0] * rho_z_re[t1] + rho_z_im[t0] * rho_z_im[t1]
            for p in range(n_pairs):
                sum_ve += (
                    vel_e[t0, p, 0] * vel_e[t1, p, 0]
                    + vel_e[t0, p, 1] * vel_e[t1, p, 1]
                    + vel_e[t0, p, 2] * vel_e[t1, p, 2]
                )
        z_e[lag] = sum_ve / float(n_orig * n_pairs)
        j_acf[lag] = sum_j / float(n_orig)
        f_zz[lag] = sum_fzz / float(n_orig)

    for lag in range(lag_i):
        n_orig = n_frames - lag
        sum_vi = 0.0
        for t0 in range(n_orig):
            t1 = t0 + lag
            for p in range(n_pairs):
                sum_vi += (
                    vel_i[t0, p, 0] * vel_i[t1, p, 0]
                    + vel_i[t0, p, 1] * vel_i[t1, p, 1]
                    + vel_i[t0, p, 2] * vel_i[t1, p, 2]
                )
        z_i[lag] = sum_vi / float(n_orig * n_pairs)

    s_zz_0 = float(f_zz[0])
    if z_e[0] > 0.0:
        z_e /= z_e[0]
    if j_acf[0] > 0.0:
        j_acf /= j_acf[0]
    if z_i[0] > 0.0:
        z_i /= z_i[0]
    if f_zz[0] > 0.0:
        f_zz /= f_zz[0]

    return z_e, j_acf, z_i, f_zz, s_zz_0


def run_cpu_md_and_collect(
    n_pairs: int = 125,
    gamma: float = 0.5,
    r_s: float = 1.0,
    mass_ratio: float = 1836.0,
    dt: float = 0.015,
    n_equil: int = 800,
    n_prod: int = 3000,
    seed: int = 1981,
) -> dict[str, object]:
    """Run velocity-Verlet MD trajectory purely on CPU and compute correlation functions for Figs
    6-8.
    """
    n_total = 2 * n_pairs
    box = ((4.0 * math.pi / 3.0) * float(n_pairs)) ** (1.0 / 3.0)
    k_b_t = 1.0 / (3.0 * gamma)

    alpha, r_cut, k_max = tune_ewald(box, tol=1e-5)
    k_vecs, k_weights = build_reciprocal_lattice(box, alpha, k_max)
    lam = compute_deutsch_lambda_matrix(gamma, r_s, mass_ratio)

    rng = np.random.default_rng(seed)
    pos = np.ascontiguousarray(rng.uniform(0.0, box, size=(n_total, 3)), dtype=np.float64)
    species = np.ascontiguousarray([0] * n_pairs + [1] * n_pairs, dtype=np.int8)
    charge = np.ascontiguousarray([1.0] * n_pairs + [-1.0] * n_pairs, dtype=np.float64)
    masses = np.ascontiguousarray([mass_ratio] * n_pairs + [1.0] * n_pairs, dtype=np.float64)

    vel = np.zeros((n_total, 3), dtype=np.float64)
    vel[:n_pairs] = rng.normal(0.0, math.sqrt(k_b_t / mass_ratio), size=(n_pairs, 3))
    vel[n_pairs:] = rng.normal(0.0, math.sqrt(k_b_t / 1.0), size=(n_pairs, 3))
    vel[:n_pairs] -= np.mean(vel[:n_pairs], axis=0)
    vel[n_pairs:] -= np.mean(vel[n_pairs:], axis=0)

    print(
        f"[CPU MD] Compiling & warming up standalone Numba CPU kernels "
        f"(N_total={n_total}, Gamma={gamma}, r_s={r_s}, m_i/m_e={mass_ratio:.0f})..."
    )
    forces, _ = compute_total_forces_cpu(
        pos, species, charge, lam, box, alpha, r_cut, k_vecs, k_weights
    )

    # 1. Термализация с раздельным масштабированием скоростей ионов и электронов
    t_start_eq = time.perf_counter()
    inv_m = (1.0 / masses)[:, None]
    for step in range(n_equil):
        vel += 0.5 * dt * forces * inv_m
        pos = (pos + dt * vel) % box
        forces, _ = compute_total_forces_cpu(
            pos, species, charge, lam, box, alpha, r_cut, k_vecs, k_weights
        )
        vel += 0.5 * dt * forces * inv_m

        if (step + 1) % 20 == 0:
            ke_i = 0.5 * mass_ratio * float(np.sum(vel[:n_pairs] ** 2))
            ke_e = 0.5 * 1.0 * float(np.sum(vel[n_pairs:] ** 2))
            target_ke = 1.5 * float(n_pairs) * k_b_t
            vel[:n_pairs] *= math.sqrt(target_ke / max(ke_i, 1e-30))
            vel[n_pairs:] *= math.sqrt(target_ke / max(ke_e, 1e-30))

    t_eq = time.perf_counter() - t_start_eq
    print(f"[CPU MD] Equilibration ({n_equil} steps) completed in {t_eq:.2f} s.")

    # 2. Продуктивный NVE-прогон: запись скоростей, полного тока J(t) и моды заряда rho_Z(k_min, t)
    vel_e_hist = np.empty((n_prod, n_pairs, 3), dtype=np.float64)
    vel_i_hist = np.empty((n_prod, n_pairs, 3), dtype=np.float64)
    j_hist = np.empty((n_prod, 3), dtype=np.float64)
    rho_z_re = np.empty(n_prod, dtype=np.float64)
    rho_z_im = np.empty(n_prod, dtype=np.float64)

    k_min = 2.0 * math.pi / box
    inv_sqrt_n = 1.0 / math.sqrt(float(n_total))

    t_start_prod = time.perf_counter()
    for step in range(n_prod):
        vel += 0.5 * dt * forces * inv_m
        pos = (pos + dt * vel) % box
        forces, _ = compute_total_forces_cpu(
            pos, species, charge, lam, box, alpha, r_cut, k_vecs, k_weights
        )
        vel += 0.5 * dt * forces * inv_m

        vi = vel[:n_pairs]
        ve = vel[n_pairs:]
        vel_i_hist[step] = vi
        vel_e_hist[step] = ve
        j_hist[step] = np.sum(vi, axis=0) - np.sum(ve, axis=0)

        # Усреднение по трём ортогональным направлениям k_min вдоль осей x, y, z
        re_sum = 0.0
        im_sum = 0.0
        for axis in range(3):
            phases = k_min * pos[:, axis]
            re_sum += float(np.dot(charge, np.cos(phases))) * inv_sqrt_n
            im_sum += float(np.dot(charge, np.sin(phases))) * inv_sqrt_n
        rho_z_re[step] = re_sum / math.sqrt(3.0)
        rho_z_im[step] = im_sum / math.sqrt(3.0)

    t_prod = time.perf_counter() - t_start_prod
    ms_per_step = (t_prod / float(n_prod)) * 1e3
    print(
        f"[CPU MD] Production ({n_prod} steps) completed in {t_prod:.2f} s "
        f"({ms_per_step:.2f} ms/step on CPU)."
    )

    max_lag_e = min(int(18.0 / dt), n_prod // 2)
    max_lag_i = min(int(45.0 / dt), n_prod - 10)

    z_e, j_acf, z_i, f_zz, s_zz_0 = compute_time_correlations_cpu(
        vel_e_hist, vel_i_hist, j_hist, rho_z_re, rho_z_im, max_lag_e, max_lag_i
    )

    t_e = np.arange(z_e.size, dtype=np.float64) * dt
    omega_pi_over_pe = 1.0 / math.sqrt(mass_ratio)
    t_i_wpi = np.arange(z_i.size, dtype=np.float64) * dt * omega_pi_over_pe

    # Интегралы Грина–Кубо
    d_e_star = float(k_b_t * np.trapezoid(z_e, t_e))
    sigma_star = float((1.0 / (4.0 * math.pi)) * np.trapezoid(j_acf, t_e))

    # Динамический структурный фактор заряда S_ZZ(k, omega) / S_ZZ(k) через косинус-
    # преобразование с окном Ханна
    omega_grid = np.linspace(0.0, 2.2, 220, dtype=np.float64)
    t_max_f = float(t_e[-1]) if t_e.size > 1 else 1.0
    hann_win = 0.5 * (1.0 + np.cos(math.pi * t_e / t_max_f))
    s_zz_norm = np.empty_like(omega_grid)
    for idx, w_val in enumerate(omega_grid):
        integrand = f_zz * hann_win * np.cos(w_val * t_e)
        s_zz_norm[idx] = (1.0 / math.pi) * float(np.trapezoid(integrand, t_e))

    print(
        f"[CPU Results] D_e* (Green-Kubo) = {d_e_star:.3f}, "
        f"sigma* = {sigma_star:.3f} omega_pe, k_min*a = {k_min:.3f}"
    )

    return {
        "t_e": t_e,
        "z_e": z_e,
        "j_acf": j_acf,
        "t_i_wpi": t_i_wpi,
        "z_i": z_i,
        "f_zz": f_zz,
        "omega_grid": omega_grid,
        "s_zz_norm": s_zz_norm,
        "k_min": k_min,
        "s_zz_0": s_zz_0,
        "d_e_star": d_e_star,
        "sigma_star": sigma_star,
        "mass_ratio": mass_ratio,
    }


def save_results_and_plots(
    data: dict[str, object],
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
) -> None:
    """Save CSV tables and generate publication-ready Figures 6, 7, and 8 (PDF + 300 dpi PNG)."""
    configure_matplotlib_style()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    t_e = np.asarray(data["t_e"], dtype=np.float64)
    z_e = np.asarray(data["z_e"], dtype=np.float64)
    j_acf = np.asarray(data["j_acf"], dtype=np.float64)
    t_i_wpi = np.asarray(data["t_i_wpi"], dtype=np.float64)
    z_i = np.asarray(data["z_i"], dtype=np.float64)
    f_zz = np.asarray(data["f_zz"], dtype=np.float64)
    omega_grid = np.asarray(data["omega_grid"], dtype=np.float64)
    s_zz_norm = np.asarray(data["s_zz_norm"], dtype=np.float64)
    k_min = float(data["k_min"])

    with (results_dir / "fig6_ze_j_cpu.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_omega_pe", "Z_e", "J_acf", "F_ZZ"])
        for row in zip(t_e, z_e, j_acf, f_zz, strict=True):
            w.writerow([f"{v:.6f}" for v in row])

    with (results_dir / "fig7_zi_cpu.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_omega_pi", "Z_i"])
        for row in zip(t_i_wpi, z_i, strict=True):
            w.writerow([f"{v:.6f}" for v in row])

    with (results_dir / "fig8_szz_cpu.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["omega_over_omega_pe", "S_ZZ_over_S_ZZ_0"])
        for row in zip(omega_grid, s_zz_norm, strict=True):
            w.writerow([f"{v:.6f}" for v in row])

    # --- График 6: АКФ скорости электронов Z_e(t) и АКФ полного электрического тока J(t) ---
    fig6, ax6 = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax6.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    ax6.plot(
        t_e,
        z_e,
        linestyle="-",
        color="#111111",
        linewidth=1.5,
        label=r"$Z_e(t) = \langle \mathbf{v}_e(t)\cdot\mathbf{v}_e(0)\rangle / \langle v_e^2 \rangle$",
    )
    ax6.plot(
        t_e,
        j_acf,
        linestyle="--",
        color="#004488",
        linewidth=1.5,
        label=r"$J(t) = \langle \mathbf{j}(t)\cdot\mathbf{j}(0)\rangle / \langle j^2 \rangle$",
    )
    ax6.set_xlabel(r"Безразмерное время $\omega_{pe} t$")
    ax6.set_ylabel(r"Нормированные АКФ $Z_e(t),\; J(t)$")
    ax6.set_xlim(0.0, float(t_e[-1]))
    ax6.set_ylim(-0.25, 1.05)
    ax6.legend(frameon=False, loc="upper right")
    fig6.tight_layout(pad=0.3)
    fig6.savefig(figures_dir / "fig6_electron_current_acf.pdf", format="pdf")
    fig6.savefig(figures_dir / "fig6_electron_current_acf.png", format="png", dpi=300)
    plt.close(fig6)

    # --- График 7: Ионная АКФ скорости Z_i(t) ---
    fig7, ax7 = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax7.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    ax7.plot(
        t_i_wpi,
        z_i,
        linestyle="-",
        color="#111111",
        linewidth=1.5,
        label=r"$Z_i(t) = \langle \mathbf{v}_i(t)\cdot\mathbf{v}_i(0)\rangle / \langle v_i^2 \rangle$",
    )
    ax7.set_xlabel(r"Ионное безразмерное время $\omega_{pi} t$")
    ax7.set_ylabel(r"Ионная АКФ скорости $Z_i(t)$")
    ax7.set_xlim(0.0, float(t_i_wpi[-1]))
    ax7.set_ylim(-0.35, 1.05)
    ax7.legend(frameon=False, loc="upper right")
    fig7.tight_layout(pad=0.3)
    fig7.savefig(figures_dir / "fig7_ion_vacf.pdf", format="pdf")
    fig7.savefig(figures_dir / "fig7_ion_vacf.png", format="png", dpi=300)
    plt.close(fig7)

    # --- График 8: Коррелятор заряда F_ZZ(k, t) и динамический структурный фактор S_ZZ(k,
    # omega) ---
    fig8, (ax8_l, ax8_r) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.94))
    ax8_l.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    ax8_l.plot(
        t_e,
        f_zz,
        linestyle="-",
        color="#111111",
        linewidth=1.4,
        label=rf"$F_{{ZZ}}(k, t)/S_{{ZZ}}(k)$ ($ka={k_min:.2f}$)",
    )
    ax8_l.set_xlabel(r"Время $\omega_{pe} t$")
    ax8_l.set_ylabel(r"Коррелятор заряда $F_{ZZ}(k, t) / S_{ZZ}(k)$")
    ax8_l.set_xlim(0.0, float(t_e[-1]))
    ax8_l.legend(frameon=False, loc="upper right")

    ax8_r.plot(
        omega_grid,
        s_zz_norm,
        linestyle="-",
        color="#004488",
        linewidth=1.5,
        label=rf"$S_{{ZZ}}(k, \omega)/S_{{ZZ}}(k)$ ($ka={k_min:.2f}$)",
    )
    ax8_r.axvline(
        1.0, color="#992222", linestyle="--", linewidth=1.0, label=r"$\omega = \omega_{pe}$"
    )
    ax8_r.set_xlabel(r"Частота $\omega / \omega_{pe}$")
    ax8_r.set_ylabel(r"Спектр $\omega_{pe} S_{ZZ}(k, \omega) / S_{ZZ}(k)$")
    ax8_r.set_xlim(0.0, float(omega_grid[-1]))
    ax8_r.legend(frameon=False, loc="upper right")

    fig8.tight_layout(pad=0.3)
    fig8.savefig(figures_dir / "fig8_charge_dsf.pdf", format="pdf")
    fig8.savefig(figures_dir / "fig8_charge_dsf.png", format="png", dpi=300)
    plt.close(fig8)

    print("Saved Figures 6, 7, and 8 in figures/fig{6,7,8}_*.{pdf,png}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone CPU MD run."""
    parser = argparse.ArgumentParser(
        description="Run standalone CPU MD simulation for Figs 6-8 of Hansen & McDonald (1981)."
    )
    parser.add_argument("--n-pairs", type=int, default=125, help="Number of ion-electron pairs N.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Coupling parameter Gamma.")
    parser.add_argument("--rs", type=float, default=1.0, help="Density parameter r_s.")
    parser.add_argument(
        "--mass-ratio",
        type=float,
        default=1836.0,
        help="Ion-to-electron mass ratio m_i/m_e.",
    )
    parser.add_argument("--dt", type=float, default=0.015, help="Time step dt [omega_pe^-1].")
    parser.add_argument("--n-equil", type=int, default=800, help="Number of equilibration steps.")
    parser.add_argument("--n-prod", type=int, default=3000, help="Number of production steps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    res = run_cpu_md_and_collect(
        n_pairs=args.n_pairs,
        gamma=args.gamma,
        r_s=args.rs,
        mass_ratio=args.mass_ratio,
        dt=args.dt,
        n_equil=args.n_equil,
        n_prod=args.n_prod,
    )
    save_results_and_plots(res)


if __name__ == "__main__":
    main()
