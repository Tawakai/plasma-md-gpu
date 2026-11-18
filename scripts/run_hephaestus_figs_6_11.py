"""Hephaestus Ab-Initio Solver for Figs. 6-11 and FSS Diffusion (Sjogren et al. 1981).

Standalone first-principles script (keeps earlier phenomenological scripts intact):
- Uses ``solve_two_component_hnc`` and ``compute_table_v_row_ab_initio`` from
  ``scripts.run_hephaestus_tables`` to obtain S_ab(q), fourth moments Omega^(4)*_ab,
  and Resibois relaxation times tau_11*(q), tau_22*(q) directly from HNC.
- Evaluates the Mori-Zwanzig kinetic memory-function equations without adjustable constants.
- Uses Student-t quantiles and Birge ratio max(1.0, sqrt(chi2/dof)) for 95% confidence intervals.
- Saves all outputs exclusively to ``figures/hephaestus/`` and ``results/hephaestus/``.
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import dawsn
from scipy.stats import t as student_t

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plasmamd.gpu.kelbg_forces import E_SQ  # noqa: E402
from plasmamd.state import PlasmaState  # noqa: E402
from scripts.run_hephaestus_tables import (  # noqa: E402
    HNCSolution,
    compute_table_v_row_ab_initio,
    deutsch_wavelengths,
    select_olympian_dirs,
    solve_two_component_hnc,
)
from scripts.run_real_fss_gpu import (  # noqa: E402
    FIG_WIDTH_IN,
    FssPoint,
    _apply_style,
    _save,
    plot_fss,
    read_csv,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

SQRT_PI: float = math.sqrt(math.pi)
SQRT_TWO: float = math.sqrt(2.0)


@dataclass(frozen=True)
class HephaestusFssFit:
    """Weighted linear regression result with Student-t 95% CI and Birge error scaling."""

    model: str
    d_inf: float
    d_inf_err: float
    d_inf_ci95: tuple[float, float]
    c: float
    c_err: float
    chi2: float
    dof: int
    chi2_per_dof: float


def plasma_dispersion_w(z: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Ichimaru plasma dispersion function W(z) = 1 - 2*u*D(u) + i*sqrt(pi)*u*exp(-u^2)."""
    u = np.asarray(z, dtype=np.float64) / SQRT_TWO
    re_w = 1.0 - 2.0 * u * dawsn(u)
    im_w = SQRT_PI * u * np.exp(-(u * u))
    return re_w + 1j * im_w


def _gaussian_memory_laplace(
    omega: NDArray[np.float64], tau: float
) -> NDArray[np.complex128]:
    """Laplace-Fourier transform of Gaussian memory kernel exp(-t^2 / (2 * tau^2)) [Eq. (31)]."""
    u = omega * tau / SQRT_TWO
    re_m = math.sqrt(0.5 * math.pi) * tau * np.exp(-(u * u))
    im_m = SQRT_TWO * tau * dawsn(u)
    return re_m + 1j * im_m


def theory_vlasov_and_memory_ab_initio(
    q: float,
    omega_grid: NDArray[np.float64],
    gamma: float,
    r_s: float,
    mass_ratio: float = 1836.0,
    include_pauli: bool = False,
    hnc: HNCSolution | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute Vlasov and Mori-Zwanzig S_ZZ(q,w) and eps(q,w) strictly from Eqs. (9)-(27)."""
    if hnc is None:
        hnc = solve_two_component_hnc(
            gamma, r_s, include_pauli=include_pauli, mass_ratio=mass_ratio
        )

    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=mass_ratio)
    k_b_t = state.k_t
    v_th_e = math.sqrt(k_b_t)
    v_th_i = math.sqrt(k_b_t / mass_ratio)
    k_de_sq = 1.0 / k_b_t

    w_safe = np.maximum(omega_grid, 1e-4)
    z_e = w_safe / (q * v_th_e)
    z_i = w_safe / (q * v_th_i)

    w_func_e = plasma_dispersion_w(z_e)
    w_func_i = plasma_dispersion_w(z_i)

    lam_ii, lam_ie, lam_ee = deutsch_wavelengths(gamma, r_s, mass_ratio)
    f_ii = 1.0 / (1.0 + (q * lam_ii) ** 2)
    f_ie = 1.0 / (1.0 + (q * lam_ie) ** 2)
    f_ee = 1.0 / (1.0 + (q * lam_ee) ** 2)

    if include_pauli:
        pauli_amp = k_b_t * math.log(2.0)
        alpha_p = 1.0 / (math.pi * math.log(2.0) * lam_ee * lam_ee)
        v_pauli_q = (
            pauli_amp * ((math.pi / alpha_p) ** 1.5) * np.exp(-(q * q) / (4.0 * alpha_p))
        )
        v_coul_q = 4.0 * math.pi * E_SQ / (q * q)
        f_ee_eff = f_ee + v_pauli_q / v_coul_q
    else:
        f_ee_eff = f_ee

    # 1. Exact Vlasov response Eqs. (9) and (21)
    alpha0_e = (k_de_sq / (q * q)) * f_ee * w_func_e
    alpha0_i = (k_de_sq / (q * q)) * f_ii * w_func_i
    eps_vlasov = 1.0 + alpha0_e + alpha0_i

    fdt_factor = (q * q) / (2.0 * math.pi * k_de_sq * w_safe)
    s_zz_vlasov = fdt_factor * np.imag(-1.0 / eps_vlasov)

    # 2. First-principles Mori-Zwanzig response from HNC & Table V moments Eqs. (19)-(27)
    s_ii, s_ie, s_ee, _, _, w4_ee, _, tau_22 = compute_table_v_row_ab_initio(
        q, gamma, r_s, hnc, mass_ratio=mass_ratio
    )
    omega0_e_sq = (q * q) * k_b_t
    w1_ee_sq = w4_ee / omega0_e_sq
    w1_vlasov_e_sq = 3.0 * omega0_e_sq + f_ee_eff

    delta_coll_e = w1_ee_sq - w1_vlasov_e_sq
    s_zz_hnc = 0.5 * (s_ii + s_ee - 2.0 * s_ie)
    s_zz_vlas_stat = (q * q) / (q * q + k_de_sq * (f_ee_eff + f_ii))
    static_ratio = s_zz_vlas_stat / max(1e-4, s_zz_hnc)

    m_coll_e = delta_coll_e * _gaussian_memory_laplace(w_safe, tau_22)
    mem_shift = -1j * (w_safe / w1_vlasov_e_sq) * m_coll_e
    w_dressed_e = w_func_e / (1.0 - mem_shift * w_func_e)

    alpha_e_mem = (
        (k_de_sq / (q * q)) * (0.5 * (f_ee_eff + f_ie) * static_ratio) * w_dressed_e
    )
    eps_mem = 1.0 + alpha_e_mem + alpha0_i
    s_zz_mem = fdt_factor * np.imag(-1.0 / eps_mem)

    return s_zz_vlasov, s_zz_mem, np.real(eps_mem), np.imag(eps_mem)


def fit_fss_student_birge(points: list[FssPoint]) -> dict[str, HephaestusFssFit]:
    """Weighted linear regression using Student-t 95% CI and Birge factor max(1.0, sqrt(chi2/dof))."""
    n_arr = np.array([p.n_particles for p in points], dtype=np.float64)
    y = np.array([p.d_e for p in points], dtype=np.float64)
    sigma = np.maximum(np.array([p.d_e_err for p in points], dtype=np.float64), 1e-6)
    w = 1.0 / (sigma * sigma)

    results: dict[str, HephaestusFssFit] = {}
    for name, x in (("N^-1/3", n_arr ** (-1.0 / 3.0)), ("N^-1", 1.0 / n_arr)):
        design = np.column_stack([np.ones_like(x), x])
        xtw = design.T * w
        cov = np.linalg.inv(xtw @ design)
        beta = cov @ (xtw @ y)
        d_inf, c_slope = float(beta[0]), float(beta[1])
        residuals = y - (d_inf + c_slope * x)
        chi2 = float(np.sum(w * residuals * residuals))
        dof = max(1, len(points) - 2)
        chi2_dof = chi2 / dof

        birge = max(1.0, math.sqrt(max(chi2_dof, 0.0)))
        d_inf_err = math.sqrt(cov[0, 0]) * birge
        c_err = math.sqrt(cov[1, 1]) * birge
        t_crit = float(student_t.ppf(0.975, dof))
        half_ci = t_crit * d_inf_err

        results[name] = HephaestusFssFit(
            model=name,
            d_inf=d_inf,
            d_inf_err=d_inf_err,
            d_inf_ci95=(d_inf - half_ci, d_inf + half_ci),
            c=c_slope,
            c_err=c_err,
            chi2=chi2,
            dof=dof,
            chi2_per_dof=chi2_dof,
        )
    return results


def build_q_mags(box: float) -> NDArray[np.float64]:
    q0 = 2.0 * math.pi / box
    return np.array([q0 * math.sqrt(float(m)) for m in (1, 2, 3, 4)], dtype=np.float64)


def load_szz_from_csv(
    csv_path: Path, gamma: float, r_s: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[NDArray[np.float64]],
    HNCSolution,
]:
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=1836.0)
    q_mags = build_q_mags(state.box)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        rows = [[float(x) for x in r] for r in reader if r]
    arr = np.array(rows, dtype=np.float64)
    omega_md = arr[:, 0]
    md_spec = [arr[:, i + 1] for i in range(4)]
    hnc = solve_two_component_hnc(gamma, r_s, include_pauli=False)
    return q_mags, omega_md, md_spec, hnc


def plot_2x2_szz_figure(
    gamma: float,
    r_s: float,
    q_mags: NDArray[np.float64],
    omega_md: NDArray[np.float64],
    md_spectra_100: list[NDArray[np.float64]],
    hnc: HNCSolution,
    stem: Path,
) -> list[Path]:
    _apply_style()
    omega_fine = np.linspace(0.01, 2.30, 300, dtype=np.float64)
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_IN * 0.82, FIG_WIDTH_IN * 0.80), sharex=True
    )
    fig.subplots_adjust(wspace=0.10, hspace=0.14)

    for idx, ax in enumerate(axes.flat):
        q_val = float(q_mags[idx])
        s_vlasov, s_mem, _, _ = theory_vlasov_and_memory_ab_initio(
            q_val, omega_fine, gamma, r_s, hnc=hnc
        )

        ax.plot(
            omega_fine,
            100.0 * s_mem,
            "-",
            color="black",
            label="Функции памяти (HNC + правила сумм)" if idx == 0 else None,
        )
        ax.plot(
            omega_fine,
            100.0 * s_vlasov,
            "--",
            color="0.4",
            label="Теория Власова" if idx == 0 else None,
        )
        ax.plot(
            omega_md,
            md_spectra_100[idx],
            "o",
            color="black",
            markersize=3.5,
            label="МД (GPU Kelbg)" if idx == 0 else None,
        )
        ax.text(0.50, 0.82, f"q = {q_val:.3f}", transform=ax.transAxes, ha="center")
        ax.set_xlim(0.0, 2.35)
        y_top = 13.5 if gamma < 1.0 else (9.5 if idx < 2 else 11.5)
        ax.set_ylim(0.0, y_top)

    axes[1, 0].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[1, 1].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[0, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[1, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7.0)
    return _save(fig, stem)


def plot_fig9_and_fig11(figures_dir: Path) -> list[Path]:
    _apply_style()
    q_small = 0.307
    hnc1_04 = solve_two_component_hnc(0.5, 0.4, include_pauli=False)
    hnc2_04 = solve_two_component_hnc(0.5, 0.4, include_pauli=True)
    hnc2_10 = solve_two_component_hnc(0.5, 1.0, include_pauli=True)

    omega_f9 = np.linspace(0.75, 1.45, 500, dtype=np.float64)
    s_vlasov_9, s_mem_9, _, _ = theory_vlasov_and_memory_ab_initio(
        q_small, omega_f9, gamma=0.5, r_s=0.4, include_pauli=False, hnc=hnc1_04
    )
    _, s_mem_pauli_9, _, _ = theory_vlasov_and_memory_ab_initio(
        q_small, omega_f9, gamma=0.5, r_s=0.4, include_pauli=True, hnc=hnc2_04
    )

    fig9, ax9 = plt.subplots(figsize=(FIG_WIDTH_IN * 0.65, FIG_WIDTH_IN * 0.56))
    ax9.plot(omega_f9, 100.0 * s_mem_9, "-", color="black", label="Функции памяти (ур. 14)")
    ax9.plot(
        omega_f9, 100.0 * s_mem_pauli_9, "-.", color="0.25", label="С учётом Паули (ур. 16)"
    )
    ax9.plot(omega_f9, 100.0 * s_vlasov_9, "--", color="0.5", label="Теория Власова")
    ax9.text(
        0.65, 0.70, "$\\Gamma = 0.5$\n$r_s = 0.4$\n$q = 0.307$", transform=ax9.transAxes
    )
    ax9.set_xlim(0.75, 1.45)
    ax9.set_ylim(0.0, 17.5)
    ax9.set_xlabel(r"$\omega / \omega_{pe}$")
    ax9.set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    ax9.legend(frameon=False, loc="upper left", fontsize=8)
    fig9.tight_layout()
    paths = _save(fig9, figures_dir / "fig9_szz_q0307")

    omega_f11 = np.linspace(0.01, 1.65, 400, dtype=np.float64)
    _, _, eps_re_04, eps_im_04 = theory_vlasov_and_memory_ab_initio(
        q_small, omega_f11, gamma=0.5, r_s=0.4, include_pauli=True, hnc=hnc2_04
    )
    _, _, eps_re_10, eps_im_10 = theory_vlasov_and_memory_ab_initio(
        q_small, omega_f11, gamma=0.5, r_s=1.0, include_pauli=True, hnc=hnc2_10
    )
    k_b_t = 1.0 / (3.0 * 0.5)
    v_th_e = math.sqrt(k_b_t)
    k_d_sq = 1.0 / k_b_t
    w_e_vlas = plasma_dispersion_w(omega_f11 / (q_small * v_th_e))
    eps_vlas_complex = 1.0 + (k_d_sq / (q_small * q_small)) * w_e_vlas

    fig11, (ax11_top, ax11_bot) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_IN * 0.65, FIG_WIDTH_IN * 0.80), sharex=True
    )
    ax11_top.axhline(0.0, color="0.6", linewidth=0.7)
    ax11_top.plot(omega_f11, eps_re_04, "-", color="black", label=r"Память ($r_s = 0.4$)")
    ax11_top.plot(omega_f11, eps_re_10, "-.", color="0.25", label=r"Память ($r_s = 1.0$)")
    ax11_top.plot(
        omega_f11, np.real(eps_vlas_complex), "--", color="0.5", label="Теория Власова"
    )
    ax11_top.set_ylabel(r"$\varepsilon'(q, \omega)$")
    ax11_top.set_ylim(-3.5, 16.5)
    ax11_top.legend(frameon=False, loc="upper right", fontsize=8)

    ax11_bot.plot(omega_f11, eps_im_04, "-", color="black")
    ax11_bot.plot(omega_f11, eps_im_10, "-.", color="0.25")
    ax11_bot.plot(omega_f11, np.imag(eps_vlas_complex), "--", color="0.5")
    ax11_bot.set_xlabel(r"$\omega / \omega_{pe}$")
    ax11_bot.set_ylabel(r"$\varepsilon''(q, \omega)$")
    ax11_bot.set_xlim(0.0, 1.65)
    ax11_bot.set_ylim(0.0, 12.0)

    fig11.tight_layout()
    paths += _save(fig11, figures_dir / "fig11_dielectric_function")
    return paths


def main() -> None:
    god_name, figures_dir, results_dir = select_olympian_dirs(Path("."), "hephaestus")
    zeus_res_dir = Path("results/zeus")

    states_szz = [
        (0.5, 0.4, "fig6_szz_gamma05_rs04"),
        (0.5, 1.0, "fig7_szz_gamma05_rs10"),
        (2.0, 1.0, "fig8_szz_gamma20_rs10"),
    ]

    print(f"🔨 [HEPHAESTUS] Генерация Рис. 6–11 и FSS из первых принципов -> figures/{god_name}/")
    for gamma, r_s, stem_name in states_szz:
        csv_p = results_dir / f"{stem_name}.csv"
        if not csv_p.is_file() and (zeus_res_dir / f"{stem_name}.csv").is_file():
            shutil.copy2(zeus_res_dir / f"{stem_name}.csv", csv_p)
        if csv_p.is_file():
            q_mags, omega_md, md_spec, hnc = load_szz_from_csv(csv_p, gamma, r_s)
            plot_2x2_szz_figure(
                gamma, r_s, q_mags, omega_md, md_spec, hnc, figures_dir / stem_name
            )

    csv_fss = results_dir / "scaling_250_to_10000.csv"
    if not csv_fss.is_file() and (zeus_res_dir / "scaling_250_to_10000.csv").is_file():
        shutil.copy2(zeus_res_dir / "scaling_250_to_10000.csv", csv_fss)
    if csv_fss.is_file():
        pts = read_csv(csv_fss)
        fits = fit_fss_student_birge(pts)
        for fit in fits.values():
            low, high = fit.d_inf_ci95
            print(
                f"  [Student-t + Birge] {fit.model:>8s}: D_inf*={fit.d_inf:.3f} +- {fit.d_inf_err:.3f} "
                f"(95% CI [{low:.3f}, {high:.3f}]), chi2/dof={fit.chi2_per_dof:.2f}"
            )
        plot_fss(pts, fits, figures_dir / "fss_250_to_10000")

    plot_fig9_and_fig11(figures_dir)
    print(f"Готово! Все графики и таблицы без подгонок сохранены в figures/{god_name}/ и results/{god_name}/.")


if __name__ == "__main__":
    main()
