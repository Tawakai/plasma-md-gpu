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


def _free_streaming_m2(
    omega: NDArray[np.float64], q: float, v_th: float
) -> NDArray[np.complex128]:
    """Exact second-order memory function M_2^(0)(q, z=-i*omega) of free-streaming particles."""
    z = -1j * omega
    w0_sq = (q * v_th) ** 2
    f0 = _gaussian_memory_laplace(omega, 1.0 / (q * v_th))
    m1_0 = (1.0 / f0) - z
    return (w0_sq / m1_0) - z


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
    """Compute exact Vlasov and 2x2 Mori-Zwanzig S_ZZ(q,w) with free-streaming + HNC memory."""
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
    w_func_e = plasma_dispersion_w(w_safe / (q * v_th_e))
    w_func_i = plasma_dispersion_w(w_safe / (q * v_th_i))

    # 1. Exact Vlasov response (bare Coulomb screening, dashed curve in Figs. 6-8)
    eps_vlasov = 1.0 + (k_de_sq / (q * q)) * (w_func_e + w_func_i)
    s_zz_vlasov = (np.imag(w_func_e + w_func_i) / (2.0 * math.pi * w_safe)) / (
        np.abs(eps_vlasov) ** 2
    )

    # 2. Exact 2x2 Mori-Zwanzig continued fraction with free-streaming M_2^(0) + HNC S(q)
    s_ii, s_ie, s_ee, w4_ii, _, w4_ee, tau_11, tau_22 = compute_table_v_row_ab_initio(
        q, gamma, r_s, hnc, mass_ratio=mass_ratio
    )
    s_mat = np.array([[s_ii, s_ie], [s_ie, s_ee]], dtype=np.float64)
    s_inv = np.linalg.inv(s_mat)

    w0_i_sq = (q * q) * (k_b_t / mass_ratio)
    w0_e_sq = (q * q) * k_b_t
    restoring = np.diag([w0_i_sq, w0_e_sq]) @ s_inv

    # Exact free-streaming second-order memory functions (preserve exact Landau damping)
    m2_free_i = _free_streaming_m2(w_safe, q, v_th_i)
    m2_free_e = _free_streaming_m2(w_safe, q, v_th_e)

    # Collisional excess over free-streaming + static restoring: Delta_coll = w1^2 - 2*w0^2 - (w0^2*S^-1)_aa
    delta_coll_i = max(0.0, (w4_ii / w0_i_sq) - 2.0 * w0_i_sq - float(restoring[0, 0]))
    delta_coll_e = max(0.0, (w4_ee / w0_e_sq) - 2.0 * w0_e_sq - float(restoring[1, 1]))

    g11 = _gaussian_memory_laplace(w_safe, tau_11)
    g22 = _gaussian_memory_laplace(w_safe, tau_22)

    s_zz_mem = np.empty_like(w_safe)
    eps_re = np.empty_like(w_safe)
    eps_im = np.empty_like(w_safe)
    eye2 = np.eye(2, dtype=np.complex128)

    for idx, w_val in enumerate(w_safe):
        z = -1j * w_val
        m2_z = np.diag(
            [
                m2_free_i[idx] + delta_coll_i * g11[idx],
                m2_free_e[idx] + delta_coll_e * g22[idx],
            ]
        )
        m1_z = np.linalg.solve(z * eye2 + m2_z, restoring)
        f_z = np.linalg.solve(z * eye2 + m1_z, s_mat)
        f_zz = f_z[0, 0] + f_z[1, 1] - 2.0 * f_z[0, 1]
        s_zz_mem[idx] = max(0.0, float(np.real(f_zz)) / (2.0 * math.pi))

        s_zz_stat = 0.5 * (s_ii + s_ee - 2.0 * s_ie)
        inv_eps = 1.0 - (2.0 * k_de_sq / (q * q)) * (s_zz_stat + z * (0.5 * f_zz))
        eps_val = 1.0 / inv_eps
        eps_re[idx] = float(np.real(eps_val))
        eps_im[idx] = float(np.imag(eps_val))

    return s_zz_vlasov, s_zz_mem, eps_re, eps_im


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
