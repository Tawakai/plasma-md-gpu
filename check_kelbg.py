"""Verify analytical Kelbg force -dv/dr and T dv/dT against finite differences."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc


def kelbg_potential(
    r: NDArray[np.float64],
    q_a: float,
    q_b: float,
    e_sq: float,
    lam_ref: float,
    t_val: float,
    t_ref: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate Kelbg potential v(r, T) with lambda(T) = lam_ref * (T / T_ref)^(-1/2)."""
    lam = lam_ref * math.sqrt(t_ref / t_val)
    x = r / lam
    return (q_a * q_b * e_sq / r) * (1.0 - np.exp(-(x * x)) + math.sqrt(math.pi) * x * erfc(x))


def kelbg_force_and_t_deriv(
    r: NDArray[np.float64],
    q_a: float,
    q_b: float,
    e_sq: float,
    lam_ref: float,
    t_val: float,
    t_ref: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate analytical radial force F(r) = -dv/dr and T * dv/dT."""
    lam = lam_ref * math.sqrt(t_ref / t_val)
    x = r / lam
    pref = q_a * q_b * e_sq
    force = (pref / (r * r)) * (1.0 - np.exp(-(x * x)))
    t_dv_dt = (0.5 * math.sqrt(math.pi) * pref / lam) * erfc(x)
    return force, t_dv_dt


def main() -> None:
    """Run grid comparison on r in [0.05, 5.0] and assert FD agreement."""
    r_grid = np.linspace(0.05, 5.0, 200, dtype=np.float64)
    q_a, q_b = 1.0, -1.0
    e_sq = 1.0 / 3.0
    lam_ref = 0.45
    t0 = 1.0 / (3.0 * 0.5)

    f_ana, tdv_ana = kelbg_force_and_t_deriv(r_grid, q_a, q_b, e_sq, lam_ref, t0, t_ref=t0)

    dr = 1e-6
    v_r_plus = kelbg_potential(r_grid + dr, q_a, q_b, e_sq, lam_ref, t0, t_ref=t0)
    v_r_minus = kelbg_potential(r_grid - dr, q_a, q_b, e_sq, lam_ref, t0, t_ref=t0)
    f_fd = -(v_r_plus - v_r_minus) / (2.0 * dr)

    # Optimal central-difference step ~(eps_mach)^(1/3) ~ 1e-5 to balance roundoff in 1/r tail
    dt = 1e-5 * t0
    v_t_plus = kelbg_potential(r_grid, q_a, q_b, e_sq, lam_ref, t0 + dt, t_ref=t0)
    v_t_minus = kelbg_potential(r_grid, q_a, q_b, e_sq, lam_ref, t0 - dt, t_ref=t0)
    tdv_fd = t0 * (v_t_plus - v_t_minus) / (2.0 * dt)

    max_rel_f = float(np.max(np.abs(f_ana - f_fd) / np.abs(f_ana)))
    max_abs_tdv = float(np.max(np.abs(tdv_ana - tdv_fd)))

    print(f"Max relative error in -dv/dr  on r in [0.05, 5.0]: {max_rel_f:.3e}")
    print(f"Max absolute error in T dv/dT on r in [0.05, 5.0]: {max_abs_tdv:.3e}")

    np.testing.assert_allclose(f_ana, f_fd, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(tdv_ana, tdv_fd, rtol=1e-7, atol=1e-9)
    print("OK: Analytical expressions match central finite differences.")


if __name__ == "__main__":
    main()
