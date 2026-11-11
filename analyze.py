"""CLI script for computing RDF, VACF, MSD, and S_ZZ(q, omega) from run.h5 (T3)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from plasmamd.observables import (
    charge_dsf,
    diffusion_block_errors,
    msd,
    rdf,
    unwrap_positions,
    vacf,
)


def analyze_h5(run_h5_path: str | Path, out_json_path: str | Path) -> dict[str, object]:
    """Analyze MD trajectory HDF5 file and write observables to JSON.

    Parameters
    ----------
    run_h5_path : str or Path
        Input HDF5 trajectory path (format from T2).
    out_json_path : str or Path
        Output JSON path.

    Returns
    -------
    results : dict
        Computed observables, diffusion coefficients with 5-block errors, and plasmon peak.
    """
    with h5py.File(run_h5_path, "r") as f:
        gamma = float(f.attrs["gamma"])
        r_s = float(f.attrs["r_s"])
        n_pairs = int(f.attrs["n_pairs"])
        box = float(f.attrs["box"])
        k_t = float(f.attrs["k_t"])
        dt_md = float(f.attrs["dt"])
        steps_frames = f["steps_frames"][:]
        positions = f["positions"][:]
        velocities = f["velocities"][:]

    frame_stride = int(steps_frames[1] - steps_frames[0]) if len(steps_frames) > 1 else 1
    dt_frame = dt_md * float(frame_stride)

    species = np.array([0] * n_pairs + [1] * n_pairs, dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)

    r, g_ii, g_ie, g_ee = rdf(positions, species, box, dr=0.02, r_max=0.5 * box)
    dr = float(r[1] - r[0]) if len(r) > 1 else 0.02
    n_density = float(n_pairs) / (box**3)
    neighbors_ie = float(np.sum(4.0 * math.pi * (r**2) * n_density * g_ie * dr))

    unwrapped = unwrap_positions(positions, box)
    t_vacf, z_e, d_e_vacf = vacf(velocities, species, dt=dt_frame, target_species=1, k_t=k_t)
    t_msd, msd_e, d_e_msd = msd(unwrapped, species, dt=dt_frame, target_species=1)
    d_v_mean, d_v_err, d_m_mean, d_m_err = diffusion_block_errors(
        unwrapped, velocities, species, dt=dt_frame, n_blocks=5, target_species=1, k_t=k_t
    )

    q_min = 2.0 * math.pi / box
    omega, s_zz, omega_peak = charge_dsf(positions, charge, box, dt=dt_frame, q_target=q_min)

    results: dict[str, object] = {
        "gamma": gamma,
        "r_s": r_s,
        "n_pairs": n_pairs,
        "box": box,
        "dt_frame": dt_frame,
        "q_min": q_min,
        "neighbors_ie_sphere": neighbors_ie,
        "d_e_star_vacf": d_e_vacf,
        "d_e_star_msd": d_e_msd,
        "d_e_star_vacf_block_mean": d_v_mean,
        "d_e_star_vacf_block_err": d_v_err,
        "d_e_star_msd_block_mean": d_m_mean,
        "d_e_star_msd_block_err": d_m_err,
        "plasmon_peak_omega": omega_peak,
        "r": r.tolist(),
        "g_ii": g_ii.tolist(),
        "g_ie": g_ie.tolist(),
        "g_ee": g_ee.tolist(),
        "t_vacf": t_vacf.tolist(),
        "z_e": z_e.tolist(),
        "t_msd": t_msd.tolist(),
        "msd_e": msd_e.tolist(),
        "omega": omega.tolist(),
        "s_zz": s_zz.tolist(),
    }

    Path(out_json_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for analyze.py."""
    parser = argparse.ArgumentParser(description="Analyze run.h5 plasma MD trajectory.")
    parser.add_argument("run_h5", type=Path, help="Input HDF5 file.")
    parser.add_argument("--out", type=Path, default=Path("results.json"), help="Output JSON file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    analyze_h5(args.run_h5, args.out)


if __name__ == "__main__":
    main()
