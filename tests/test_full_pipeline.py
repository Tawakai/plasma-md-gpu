"""End-to-end test: pair forces -> GPU-resident MD -> finite-size scaling and figures.

Run with ``pytest tests/test_full_pipeline.py -v -s``.

Environment switches
--------------------
``PLASMAMD_BACKEND=cpu``
    Run the MD-based tests on the CPU reference backend (machines without CUDA).
``NUMBA_ENABLE_CUDASIM=1``
    Numba's CUDA simulator.  It executes kernels in pure Python, so system sizes and step
    counts are cut down and only the logic (not the physics thresholds) is checked.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType

import numba
import numpy as np
import pytest
from numba import cuda
from PIL import Image

from plasmamd.gpu.kelbg_forces import (
    E_SQ,
    POTENTIAL_DEUTSCH,
    POTENTIAL_KELBG,
    pair_forces_kelbg_cpu,
    pair_forces_kelbg_gpu,
    pair_terms,
)
from plasmamd.gpu.md_engine_gpu import MDConfig, initial_positions, run_md
from plasmamd.state import PlasmaState

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = bool(numba.config.ENABLE_CUDASIM)
BACKEND = os.environ.get("PLASMAMD_BACKEND", "gpu")
CUDA_OK = cuda.is_available()
GAMMA, R_S = 0.5, 1.0


def _require_backend() -> None:
    if BACKEND == "gpu" and not CUDA_OK:
        pytest.skip("CUDA is not available; set PLASMAMD_BACKEND=cpu to run on the CPU backend")


def _random_system(n_pairs: int, seed: int = 7):
    """Random two-component configuration in the box of state (GAMMA, R_S)."""
    state = PlasmaState(GAMMA, R_S, n_pairs=n_pairs, mass_ratio=1836.0)
    rng = np.random.default_rng(seed)
    pos = initial_positions(n_pairs, state.box, rng)
    species = np.repeat(np.array([0, 1], dtype=np.int32), n_pairs)
    charge = np.where(species == 0, 1.0, -1.0)
    return state, pos, species, charge


def _load_fss_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_real_fss_gpu.py"
    spec = importlib.util.spec_from_file_location("run_real_fss_gpu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# 1. Pair forces: formulas, CPU, GPU
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("potential_type", [POTENTIAL_DEUTSCH, POTENTIAL_KELBG])
def test_pair_formulas_against_finite_differences(potential_type: int) -> None:
    """F_r = -dv/dr and T dv/dT of a single pair, both potentials, like and unlike charges."""
    alpha, h, eps = 0.63, 1e-6, 1e-5
    for qq_e2 in (E_SQ, -E_SQ):
        for lam in (0.282, 0.399):
            for r in np.geomspace(0.05, 5.0, 25):
                _, f_r, tdv = pair_terms(r, qq_e2, lam, alpha, potential_type)
                v_plus = pair_terms(r + h, qq_e2, lam, alpha, potential_type)[0]
                v_minus = pair_terms(r - h, qq_e2, lam, alpha, potential_type)[0]
                assert f_r == pytest.approx(-(v_plus - v_minus) / (2 * h), rel=1e-6, abs=1e-12)
                # lambda ~ T^(-1/2): T -> T (1 +- eps) means lambda -> lambda (1 +- eps)^(-1/2)
                v_hot = pair_terms(r, qq_e2, lam / math.sqrt(1 + eps), alpha, potential_type)[0]
                v_cold = pair_terms(r, qq_e2, lam / math.sqrt(1 - eps), alpha, potential_type)[0]
                assert tdv == pytest.approx((v_hot - v_cold) / (2 * eps), rel=1e-6, abs=1e-12)


def test_kelbg_cpu_gpu_and_finite_diff() -> None:
    """Kelbg forces and T dU/dT vs central differences, then CPU vs GPU for both potentials."""
    n_pairs = 20 if SIMULATOR else 250  # N_total = 500 on real hardware
    state, pos, species, charge = _random_system(n_pairs)
    box, lam, r_cut = state.box, state.lam, 0.5 * state.box

    # A large alpha makes erfc(alpha r_cut) ~ 1e-23, so a pair crossing the cutoff during
    # the finite-difference displacement cannot spoil the comparison.
    alpha_fd = 7.0 / r_cut

    def energy(positions: np.ndarray, lam_matrix: np.ndarray) -> float:
        return pair_forces_kelbg_cpu(
            positions, species, charge, lam_matrix, box, alpha_fd, r_cut, POTENTIAL_KELBG
        )[1]

    forces, u_sr, w_sr, tdv_dt = pair_forces_kelbg_cpu(
        pos, species, charge, lam, box, alpha_fd, r_cut, POTENTIAL_KELBG
    )
    assert np.all(np.isfinite(forces)) and math.isfinite(w_sr)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-10)

    h = 1e-5
    probes = [0, n_pairs // 2, n_pairs - 1, n_pairs, n_pairs + n_pairs // 2, 2 * n_pairs - 1]
    scale = float(np.abs(forces).max())
    for i in probes:
        for d in range(3):
            plus, minus = pos.copy(), pos.copy()
            plus[i, d] += h
            minus[i, d] -= h
            numeric = -(energy(plus, lam) - energy(minus, lam)) / (2 * h)
            assert forces[i, d] == pytest.approx(numeric, rel=1e-6, abs=1e-7 * scale)

    eps = 1e-4
    numeric_tdv = (
        energy(pos, lam / math.sqrt(1 + eps)) - energy(pos, lam / math.sqrt(1 - eps))
    ) / (2 * eps)
    assert tdv_dt == pytest.approx(numeric_tdv, rel=1e-6)
    print(f"\n[1] Kelbg CPU: U_sr={u_sr:.6f}  T dU/dT={tdv_dt:.6f} (FD {numeric_tdv:.6f})")

    if not CUDA_OK:
        pytest.skip("finite-difference part passed; CUDA is not available for the CPU/GPU part")

    alpha = 3.2 / r_cut
    for potential_type in (POTENTIAL_DEUTSCH, POTENTIAL_KELBG):
        cpu = pair_forces_kelbg_cpu(pos, species, charge, lam, box, alpha, r_cut, potential_type)
        gpu = pair_forces_kelbg_gpu(pos, species, charge, lam, box, alpha, r_cut, potential_type)
        np.testing.assert_allclose(gpu[0], cpu[0], rtol=0.0, atol=1e-11)
        for name, a, b in zip(("u_sr", "w_sr", "tdv_dt"), gpu[1:], cpu[1:], strict=True):
            assert a == pytest.approx(b, rel=0.0, abs=1e-11), name
        print(
            f"[1] potential_type={potential_type}: "
            f"max|F_gpu - F_cpu|={np.abs(gpu[0] - cpu[0]).max():.2e}  u_sr={gpu[1]:.6f}"
        )


# --------------------------------------------------------------------------------------
# 2. GPU-resident MD: NVE conservation laws
# --------------------------------------------------------------------------------------


def test_gpu_md_nve_energy_conservation() -> None:
    """E_tot = E_kin + U_sr + U_recip + U_self and total momentum over 300 NVE steps."""
    _require_backend()
    if SIMULATOR:
        sizes = dict(n_pairs=6, n_equil=10, n_prod=40, sample_stride=5)
    else:
        sizes = dict(n_pairs=125, n_equil=300, n_prod=300, sample_stride=10)
    config = MDConfig(gamma=GAMMA, r_s=R_S, n_blocks=1, seed=3, backend=BACKEND, **sizes)
    result = run_md(config)

    kinetic = 1.5 * config.n_pairs * float(np.mean(result.kt_ion + result.kt_electron))
    spread = float(result.e_total.max() - result.e_total.min()) / kinetic
    max_momentum = float(np.abs(result.momentum).max())
    print(
        f"\n[2] backend={BACKEND}  N={2 * config.n_pairs}  steps={config.n_prod}  "
        f"(E_max - E_min)/E_kin={spread:.2e}  max|P|={max_momentum:.2e}  "
        f"<kT_i>={result.kt_ion.mean():.3f}  <kT_e>={result.kt_electron.mean():.3f}  "
        f"{result.steps_per_second:.0f} steps/s"
    )
    assert np.all(np.isfinite(result.e_total))
    assert max_momentum < 1e-9
    assert np.all(np.diff(result.msd_e) != 0.0) and result.msd_e[0] == 0.0
    # U_int differs from E_tot by -T dU_sr/dT, which is non-zero for the Kelbg potential.
    assert np.all(np.abs(result.u_internal - result.e_total) > 0.0)
    if not SIMULATOR:
        # dt = 0.03: the spread is integration error (it falls ~5x when dt is halved).
        assert spread < 5e-3
        assert result.kt_electron.mean() == pytest.approx(config.state.k_t, rel=0.2)
        assert result.kt_ion.mean() == pytest.approx(config.state.k_t, rel=0.2)

    if BACKEND == "gpu":
        # Same initial state, no thermostat: GPU and CPU trajectories must agree.
        short = dict(n_pairs=sizes["n_pairs"], n_equil=0, n_prod=20, sample_stride=5, n_blocks=1)
        gpu = run_md(MDConfig(gamma=GAMMA, r_s=R_S, seed=3, backend="gpu", **short))
        cpu = run_md(MDConfig(gamma=GAMMA, r_s=R_S, seed=3, backend="cpu", **short))
        np.testing.assert_allclose(gpu.e_total, cpu.e_total, rtol=1e-9)
        np.testing.assert_allclose(gpu.u_internal, cpu.u_internal, rtol=1e-9)
        np.testing.assert_allclose(gpu.msd_e, cpu.msd_e, rtol=1e-7, atol=1e-12)


# --------------------------------------------------------------------------------------
# 3. Finite-size scaling pipeline and figures
# --------------------------------------------------------------------------------------


def test_end_to_end_real_fss_and_plots(tmp_path: Path) -> None:
    """Simulate three sizes, write the CSV, fit Yeh-Hummer, draw both figures."""
    _require_backend()
    fss = _load_fss_script()
    if SIMULATOR:
        n_pairs_list = (3, 4, 6)
        settings = fss.RunSettings(
            n_equil=4, n_prod=40, n_blocks=2, sample_stride=5, backend=BACKEND,
            rdf_n_pairs=4, rdf_n_equil=4, rdf_n_prod=20, rdf_stride=5, rdf_bins=8,
        )
    else:
        n_pairs_list = (64, 125, 250)
        settings = fss.RunSettings(**{**fss.QUICK_SETTINGS.__dict__, "backend": BACKEND})

    result = fss.run_pipeline(n_pairs_list, settings, output_root=tmp_path)

    csv_path = tmp_path / "results" / "diffusion_fss_real.csv"
    assert result.csv_path == csv_path and csv_path.is_file()
    points = fss.read_csv(csv_path)
    assert [p.n_pairs for p in points] == list(n_pairs_list)
    assert [p.n_particles for p in points] == [2 * n for n in n_pairs_list]
    assert all(math.isfinite(p.d_e) and math.isfinite(p.d_e_err) for p in points)
    assert all(p.d_e_err > 0.0 for p in points)

    yeh = result.fits["yeh_hummer"]
    inverse = result.fits["inverse_n"]
    low, high = yeh.d_inf_ci95
    for fit in (yeh, inverse):
        assert all(math.isfinite(v) for v in (fit.d_inf, fit.d_inf_err, fit.c, fit.c_err, fit.chi2))
        assert fit.d_inf_err > 0.0 and fit.dof == len(points) - 2
    assert low < yeh.d_inf < high
    if not SIMULATOR:
        assert all(p.d_e > 0.0 for p in points)
        assert yeh.d_inf > 0.0
        assert all(abs(p.energy_drift) < 0.1 for p in points)

    expected = [
        tmp_path / "figures" / f"{stem}.{ext}"
        for stem in ("real_fss_kelbg", "rdf_deutsch_vs_kelbg", "system_telemetry")
        for ext in ("pdf", "png")
    ]
    assert sorted(result.figure_paths) == sorted(expected)
    for path in expected:
        assert path.is_file() and path.stat().st_size > 1000, path
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")
        else:
            with Image.open(path) as image:
                dpi = image.info["dpi"]
                width_mm = image.size[0] / dpi[0] * 25.4
            assert dpi[0] == pytest.approx(300, abs=0.5) and dpi[1] == pytest.approx(300, abs=0.5)
            assert width_mm == pytest.approx(160.0, abs=1.0)
    print(f"\n[3] D_inf* (Yeh-Hummer) = {yeh.d_inf:.3f}, 95% CI [{low:.3f}, {high:.3f}]")
    print(f"[3] files under {tmp_path}")
