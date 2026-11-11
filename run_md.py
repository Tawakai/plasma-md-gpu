"""CLI script for running two-component hydrogen plasma MD and saving HDF5 output."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from plasmamd.md import MDTrajectoryData, run_simulation
from plasmamd.state import PlasmaState


def save_trajectory_h5(
    out_path: str | Path,
    state: PlasmaState,
    traj: MDTrajectoryData,
    dt: float,
    n_equil_steps: int,
    n_prod_steps: int,
    seed: int,
) -> None:
    """Write MD observables and trajectory snapshots to an HDF5 file.

    Parameters
    ----------
    out_path : str or Path
        Output HDF5 file path.
    state : PlasmaState
        Simulated plasma thermodynamic state.
    traj : MDTrajectoryData
        Recorded trajectory data.
    dt : float
        Integration time step [1/omega_pe].
    n_equil_steps : int
        Number of equilibration steps [dimensionless].
    n_prod_steps : int
        Number of NVE production steps [dimensionless].
    seed : int
        Random number seed [dimensionless].
    """
    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["gamma"] = state.gamma
        h5f.attrs["r_s"] = state.r_s
        h5f.attrs["n_pairs"] = state.n_pairs
        h5f.attrs["mass_ratio"] = state.mass_ratio
        h5f.attrs["box"] = state.box
        h5f.attrs["k_t"] = state.k_t
        h5f.attrs["dt"] = dt
        h5f.attrs["n_equil_steps"] = n_equil_steps
        h5f.attrs["n_prod_steps"] = n_prod_steps
        h5f.attrs["seed"] = seed

        h5f.create_dataset("steps_energy", data=traj.steps_energy)
        h5f.create_dataset("potential_energy", data=traj.u_therm)
        h5f.create_dataset("configurational_energy", data=traj.u_cfg)
        h5f.create_dataset("kinetic_energy", data=traj.kinetic_energy)
        h5f.create_dataset("total_energy", data=traj.total_energy)
        h5f.create_dataset("virial", data=traj.virial)
        h5f.create_dataset("temperature_ion", data=traj.temp_i)
        h5f.create_dataset("temperature_electron", data=traj.temp_e)
        h5f.create_dataset("momentum", data=traj.momentum)

        h5f.create_dataset("steps_frames", data=traj.steps_frames)
        h5f.create_dataset("positions", data=traj.positions)
        h5f.create_dataset("velocities", data=traj.velocities)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run_md.py."""
    parser = argparse.ArgumentParser(
        description="Run two-component hydrogen plasma MD simulation on GPU."
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="Coupling parameter Gamma.")
    parser.add_argument("--rs", type=float, default=1.0, help="Density parameter r_s.")
    parser.add_argument("--n-pairs", type=int, default=125, help="Number of i-e pairs N.")
    parser.add_argument("--steps", type=int, default=50000, help="Number of NVE steps.")
    parser.add_argument("--equil-steps", type=int, default=10000, help="Number of Berendsen steps.")
    parser.add_argument("--dt", type=float, default=0.03, help="Time step [1/omega_pe].")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed.")
    parser.add_argument("--out", type=str, default="run.h5", help="Output HDF5 file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for CLI MD execution."""
    args = parse_args(argv)
    state = PlasmaState(
        gamma=args.gamma,
        r_s=args.rs,
        n_pairs=args.n_pairs,
        mass_ratio=1836.15,
    )
    tau_berendsen = 100.0 * args.dt
    traj = run_simulation(
        state=state,
        n_equil_steps=args.equil_steps,
        n_prod_steps=args.steps,
        dt=args.dt,
        tau_berendsen=tau_berendsen,
        min_dist=0.3,
        seed=args.seed,
        energy_stride=10,
        frame_stride=100,
    )
    save_trajectory_h5(
        out_path=args.out,
        state=state,
        traj=traj,
        dt=args.dt,
        n_equil_steps=args.equil_steps,
        n_prod_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
