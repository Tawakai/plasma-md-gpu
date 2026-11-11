"""Pure functions for RDF, VACF, MSD, and charge dynamical structure factor S_ZZ(q, omega)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def unwrap_positions(
    positions: NDArray[np.float64],
    box: float,
) -> NDArray[np.float64]:
    """Unwrap periodic trajectory coordinates using minimum-image frame-to-frame displacements.

    Parameters
    ----------
    positions : np.ndarray, shape (M, N_tot, 3), dtype float64
        Wrapped trajectory coordinates in [0, box)^3 [a].
    box : float
        Cubic simulation box side length L [a].

    Returns
    -------
    unwrapped : np.ndarray, shape (M, N_tot, 3), dtype float64
        Continuous unwrapped particle coordinates [a].
    """
    pos_c = np.ascontiguousarray(positions, dtype=np.float64)
    if pos_c.shape[0] <= 1:
        return pos_c.copy()

    deltas = np.diff(pos_c, axis=0)
    deltas -= box * np.round(deltas / box)
    unwrapped = np.empty_like(pos_c)
    unwrapped[0] = pos_c[0]
    unwrapped[1:] = pos_c[0] + np.cumsum(deltas, axis=0)
    return unwrapped


def rdf(
    positions: NDArray[np.float64],
    species: NDArray[np.int8],
    box: float,
    dr: float = 0.02,
    r_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute partial radial distribution functions g_ii(r), g_ie(r), g_ee(r) [Sec. III].

    Parameters
    ----------
    positions : np.ndarray, shape (M, 2N, 3), dtype float64
        Trajectory coordinate frames in reduced units [a].
    species : np.ndarray, shape (2N,), dtype int8
        Species index for each particle (0 for ions "i", 1 for electrons "e") [dimensionless].
    box : float
        Cubic box side length L in reduced units [a].
    dr : float, optional
        Histogram bin width in reduced units [a], default 0.02.
    r_max : float, optional
        Maximum radius in reduced units [a], default box / 2.

    Returns
    -------
    r : np.ndarray, shape (K,), dtype float64
        Bin center radii in reduced units [a].
    g_ii : np.ndarray, shape (K,), dtype float64
        Ion-ion pair distribution function g_ii(r) [dimensionless].
    g_ie : np.ndarray, shape (K,), dtype float64
        Ion-electron pair distribution function g_ie(r) [dimensionless].
    g_ee : np.ndarray, shape (K,), dtype float64
        Electron-electron pair distribution function g_ee(r) [dimensionless].
    """
    if r_max is None:
        r_max = 0.5 * box

    n_bins = int(math.floor(r_max / dr))
    r_edges = np.linspace(0.0, n_bins * dr, n_bins + 1, dtype=np.float64)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])

    mask_i = species == 0
    mask_e = species == 1
    n_i = int(np.sum(mask_i))
    n_e = int(np.sum(mask_e))
    volume = box * box * box
    rho_i = n_i / volume
    rho_e = n_e / volume

    hist_ii = np.zeros(n_bins, dtype=np.float64)
    hist_ie = np.zeros(n_bins, dtype=np.float64)
    hist_ee = np.zeros(n_bins, dtype=np.float64)

    n_frames = positions.shape[0]
    inv_box = 1.0 / box
    upper_ii = np.triu_indices(n_i, k=1)
    upper_ee = np.triu_indices(n_e, k=1)

    for frame_idx in range(n_frames):
        pos = positions[frame_idx]
        pos_i = pos[mask_i]
        pos_e = pos[mask_e]

        diff_ii = pos_i[:, None, :] - pos_i[None, :, :]
        diff_ii -= box * np.round(diff_ii * inv_box)
        dist_ii = np.linalg.norm(diff_ii[upper_ii], axis=-1)
        h_ii, _ = np.histogram(dist_ii, bins=r_edges)
        hist_ii += 2.0 * h_ii

        diff_ee = pos_e[:, None, :] - pos_e[None, :, :]
        diff_ee -= box * np.round(diff_ee * inv_box)
        dist_ee = np.linalg.norm(diff_ee[upper_ee], axis=-1)
        h_ee, _ = np.histogram(dist_ee, bins=r_edges)
        hist_ee += 2.0 * h_ee

        diff_ie = pos_i[:, None, :] - pos_e[None, :, :]
        diff_ie -= box * np.round(diff_ie * inv_box)
        dist_ie = np.linalg.norm(diff_ie, axis=-1).ravel()
        h_ie, _ = np.histogram(dist_ie, bins=r_edges)
        hist_ie += h_ie

    shell_vol = 4.0 * math.pi * (r * r) * dr

    norm_ii = n_frames * n_i * rho_i * shell_vol
    norm_ie = n_frames * n_i * rho_e * shell_vol
    norm_ee = n_frames * n_e * rho_e * shell_vol

    g_ii = hist_ii / norm_ii
    g_ie = hist_ie / norm_ie
    g_ee = hist_ee / norm_ee
    return r, g_ii, g_ie, g_ee


def vacf(
    velocities: NDArray[np.float64],
    species: NDArray[np.int8],
    dt: float,
    target_species: int = 1,
    k_t: float | None = None,
    mass: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute normalized VACF Z_a(t) and diffusion D_a [Eq. (32)-(33)].

    Parameters
    ----------
    velocities : np.ndarray, shape (M, 2N, 3), dtype float64
        Particle velocity frames in reduced units [a omega_pe].
    species : np.ndarray, shape (2N,), dtype int8
        Species index for each particle (0 for ions, 1 for electrons) [dimensionless].
    dt : float
        Time interval between consecutive velocity frames [1/omega_pe].
    target_species : int, optional
        Species index to analyze (default 1 for electrons) [dimensionless].
    k_t : float, optional
        Thermal energy k_B T [m_e a^2 omega_pe^2]. If None, estimated from <m v^2 / 3>.
    mass : float, optional
        Particle mass m_a in reduced units [m_e], default 1.0.

    Returns
    -------
    t_lags : np.ndarray, shape (M,), dtype float64
        Lag times t in reduced units [1/omega_pe].
    z_t : np.ndarray, shape (M,), dtype float64
        Normalized velocity autocorrelation function Z_a(t) [dimensionless].
    d_star : float
        Reduced diffusion coefficient D_a^* = (k_B T / m_a) int_0^t_cutoff Z_a(t) dt [a^2 omega_pe].
    """
    mask = species == target_species
    v_s = np.ascontiguousarray(velocities[:, mask, :], dtype=np.float64)
    n_frames, _, _ = v_s.shape

    n_fft = 1 << (2 * n_frames - 1).bit_length()
    f_v = np.fft.rfft(v_s, n=n_fft, axis=0)
    power = np.abs(f_v) ** 2
    acf_full = np.fft.irfft(power, n=n_fft, axis=0)[:n_frames]
    counts = np.arange(n_frames, 0, -1, dtype=np.float64)[:, None, None]
    unnorm_acf = np.mean(np.sum(acf_full / counts, axis=-1), axis=-1)

    v_sq_mean = float(unnorm_acf[0])
    if v_sq_mean <= 0.0:
        zeros = np.zeros(n_frames, dtype=np.float64)
        return np.arange(n_frames, dtype=np.float64) * dt, zeros, 0.0

    z_t = unnorm_acf / v_sq_mean
    t_lags = np.arange(n_frames, dtype=np.float64) * dt

    zero_crossings = np.where(z_t <= 0.0)[0]
    if len(zero_crossings) > 0:
        cutoff_idx = int(zero_crossings[0])
    else:
        cutoff_idx = min(n_frames, max(2, n_frames // 2))
    cutoff_idx = max(2, cutoff_idx)

    kt_over_m = (k_t / mass) if k_t is not None else (v_sq_mean / 3.0)
    integral_z = float(np.trapezoid(z_t[:cutoff_idx], t_lags[:cutoff_idx]))
    d_star = kt_over_m * integral_z
    return t_lags, z_t, d_star


def msd(
    positions_unwrapped: NDArray[np.float64],
    species: NDArray[np.int8],
    dt: float,
    target_species: int = 1,
    fit_start_frac: float = 0.1,
    fit_end_frac: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute mean squared displacement MSD(t) and diffusion coefficient from slope.

    Parameters
    ----------
    positions_unwrapped : np.ndarray, shape (M, 2N, 3), dtype float64
        Unwrapped particle coordinates in reduced units [a].
    species : np.ndarray, shape (2N,), dtype int8
        Species index for each particle (0 for ions, 1 for electrons) [dimensionless].
    dt : float
        Time interval between consecutive coordinate frames [1/omega_pe].
    target_species : int, optional
        Species index to analyze (default 1 for electrons) [dimensionless].
    fit_start_frac : float, optional
        Start fraction of lag range for linear Einstein slope fit [dimensionless].
    fit_end_frac : float, optional
        End fraction of lag range for linear Einstein slope fit [dimensionless].

    Returns
    -------
    t_lags : np.ndarray, shape (M,), dtype float64
        Lag times t in reduced units [1/omega_pe].
    msd_vals : np.ndarray, shape (M,), dtype float64
        Mean squared displacement <|r(t) - r(0)|^2> in reduced units [a^2].
    d_star : float
        Diffusion coefficient D_a^* = slope / 6 in reduced units [a^2 omega_pe].
    """
    mask = species == target_species
    pos_s = np.ascontiguousarray(positions_unwrapped[:, mask, :], dtype=np.float64)
    n_frames = pos_s.shape[0]

    msd_vals = np.zeros(n_frames, dtype=np.float64)
    for lag in range(1, n_frames):
        disp = pos_s[lag:] - pos_s[: n_frames - lag]
        msd_vals[lag] = float(np.mean(np.sum(disp * disp, axis=-1)))

    t_lags = np.arange(n_frames, dtype=np.float64) * dt
    i_start = max(1, int(fit_start_frac * n_frames))
    i_end = max(i_start + 2, int(fit_end_frac * n_frames))
    slope, _ = np.polyfit(t_lags[i_start:i_end], msd_vals[i_start:i_end], 1)
    d_star = float(slope / 6.0)
    return t_lags, msd_vals, d_star


def diffusion_block_errors(
    positions_unwrapped: NDArray[np.float64],
    velocities: NDArray[np.float64],
    species: NDArray[np.int8],
    dt: float,
    n_blocks: int = 5,
    target_species: int = 1,
    k_t: float | None = None,
    mass: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute block-averaged diffusion coefficients and standard errors over n_blocks=5 blocks.

    Parameters
    ----------
    positions_unwrapped : np.ndarray, shape (M, 2N, 3), dtype float64
        Unwrapped particle coordinates [a].
    velocities : np.ndarray, shape (M, 2N, 3), dtype float64
        Particle velocities [a omega_pe].
    species : np.ndarray, shape (2N,), dtype int8
        Species indices [dimensionless].
    dt : float
        Time step between trajectory frames [1/omega_pe].
    n_blocks : int, optional
        Number of trajectory blocks for error estimation (default 5) [dimensionless].
    target_species : int, optional
        Species index (default 1 for electrons) [dimensionless].
    k_t : float, optional
        Thermal energy k_B T [m_e a^2 omega_pe^2].
    mass : float, optional
        Particle mass [m_e], default 1.0.

    Returns
    -------
    d_vacf_mean : float
        Block mean of D_e^* from VACF [a^2 omega_pe].
    d_vacf_err : float
        Standard error of the mean for D_e^* from VACF [a^2 omega_pe].
    d_msd_mean : float
        Block mean of D_e^* from MSD [a^2 omega_pe].
    d_msd_err : float
        Standard error of the mean for D_e^* from MSD [a^2 omega_pe].
    """
    n_frames = positions_unwrapped.shape[0]
    block_len = n_frames // n_blocks
    vacf_blocks: list[float] = []
    msd_blocks: list[float] = []

    for b in range(n_blocks):
        s_idx = b * block_len
        e_idx = (b + 1) * block_len
        pos_b = positions_unwrapped[s_idx:e_idx]
        vel_b = velocities[s_idx:e_idx]
        _, _, d_v = vacf(vel_b, species, dt=dt, target_species=target_species, k_t=k_t, mass=mass)
        _, _, d_m = msd(pos_b, species, dt=dt, target_species=target_species)
        vacf_blocks.append(d_v)
        msd_blocks.append(d_m)

    v_arr = np.array(vacf_blocks, dtype=np.float64)
    m_arr = np.array(msd_blocks, dtype=np.float64)
    denom = math.sqrt(max(1, n_blocks - 1))
    return (
        float(np.mean(v_arr)),
        float(np.std(v_arr, ddof=1) / denom),
        float(np.mean(m_arr)),
        float(np.std(m_arr, ddof=1) / denom),
    )


def charge_dsf(
    positions: NDArray[np.float64],
    charge: NDArray[np.float64],
    box: float,
    dt: float,
    q_target: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute charge dynamical structure factor S_ZZ(q, omega) [Sec. V, Eq. (24), (59)-(64)].

    Averages over all wave vectors k of equal magnitude |k| = q_target (default 2*pi/L)
    and identifies the plasmon peak frequency omega_peak > 0.

    Parameters
    ----------
    positions : np.ndarray, shape (M, 2N, 3), dtype float64
        Trajectory coordinate frames in reduced units [a].
    charge : np.ndarray, shape (2N,), dtype float64
        Particle charge numbers q_j in units of [e] (±1.0).
    box : float
        Cubic simulation box side length L in reduced units [a].
    dt : float
        Time interval between consecutive coordinate frames [1/omega_pe].
    q_target : float, optional
        Target wave-number magnitude q in reduced units [1/a], default 2*pi/box [Eq. (76)].

    Returns
    -------
    omega : np.ndarray, shape (W,), dtype float64
        Non-negative angular frequencies omega in units of [omega_pe].
    s_zz : np.ndarray, shape (W,), dtype float64
        Charge dynamical structure factor S_ZZ(q, omega) in units of [1/omega_pe].
    omega_peak : float
        Plasmon peak frequency in reduced units [omega_pe].
    """
    dk = 2.0 * math.pi / box
    if q_target is None:
        q_target = dk

    m_max = max(1, int(round(q_target / dk)) + 1)
    k_list: list[NDArray[np.float64]] = []
    for mx in range(-m_max, m_max + 1):
        for my in range(-m_max, m_max + 1):
            for mz in range(-m_max, m_max + 1):
                if mx == 0 and my == 0 and mz == 0:
                    continue
                k_vec = np.array([mx, my, mz], dtype=np.float64) * dk
                if abs(float(np.linalg.norm(k_vec)) - q_target) <= 1e-4 * dk:
                    k_list.append(k_vec)

    if not k_list:
        k_list = [
            np.array([q_target, 0.0, 0.0], dtype=np.float64),
            np.array([-q_target, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, q_target, 0.0], dtype=np.float64),
            np.array([0.0, -q_target, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, q_target], dtype=np.float64),
            np.array([0.0, 0.0, -q_target], dtype=np.float64),
        ]

    n_frames, n_total, _ = positions.shape
    window = np.hanning(n_frames)
    win_norm = float(np.sum(window * window))

    n_pos = n_frames // 2 + 1
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    omega = 2.0 * math.pi * freqs
    s_zz_accum = np.zeros_like(omega)

    for k_vec in k_list:
        kr = np.tensordot(positions, k_vec, axes=([2], [0]))
        rho_z = np.sum(charge[None, :] * np.exp(1j * kr), axis=1)
        rho_z = rho_z - np.mean(rho_z)
        fft_vals = np.fft.fft(rho_z * window)[:n_pos]
        s_zz_k = (dt / (2.0 * math.pi * float(n_total) * win_norm)) * (np.abs(fft_vals) ** 2)
        s_zz_accum += s_zz_k

    s_zz = s_zz_accum / float(len(k_list))

    pos_mask = omega > (omega[1] if len(omega) > 1 else 0.0)
    if np.any(pos_mask):
        omega_pos = omega[pos_mask]
        s_pos = s_zz[pos_mask]
        omega_peak = float(omega_pos[int(np.argmax(s_pos))])
    else:
        omega_peak = 0.0

    return omega, s_zz, omega_peak


def block_average(
    x: NDArray[np.float64],
) -> tuple[float, float, int]:
    """Compute sample mean, standard error, and optimal block count via Flyvbjerg-Petersen method.

    Implements iterative pairwise blocking from H. Flyvbjerg and H. G. Petersen,
    J. Chem. Phys. 91, 461 (1989) [Eq. (17), (20), (28)] with automatic plateau detection.

    Parameters
    ----------
    x : np.ndarray, shape (N,), dtype float64
        1D autocorrelated time series (e.g. potential energy from MD) [arbitrary units].

    Returns
    -------
    mean : float
        Sample mean of the full time series [same units as x].
    stderr : float
        Estimated standard error of the mean at the blocking plateau [same units as x].
    n_blocks_used : int
        Number of blocks n_k at the selected plateau transformation step [dimensionless].
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n_total = int(arr.size)
    if n_total < 2:
        raise ValueError(f"Time series must contain at least 2 points, got {n_total}.")

    mean_val = float(np.mean(arr))

    s_vals: list[float] = []
    ds_vals: list[float] = []
    n_blocks_list: list[int] = []

    curr = arr.copy()
    while curr.size >= 4:
        n_k = int(curr.size)
        curr_mean = float(np.mean(curr))
        c0_k = float(np.mean((curr - curr_mean) ** 2))
        s_k = math.sqrt(c0_k / float(n_k - 1))
        ds_k = s_k / math.sqrt(2.0 * float(n_k - 1))

        s_vals.append(s_k)
        ds_vals.append(ds_k)
        n_blocks_list.append(n_k)

        n_pairs = n_k // 2
        if n_pairs < 4:
            break
        curr = 0.5 * (curr[: 2 * n_pairs : 2] + curr[1 : 2 * n_pairs : 2])

    # Automatic plateau selection:
    # Restrict search to levels with at least 16 blocks so variance of s_k is small (< 18%).
    # Identify the first step k (after initial correlation rise) where two consecutive
    # blocking steps show growth smaller than the 1-sigma Flyvbjerg-Petersen error bar ds_k.
    valid_indices = [idx for idx, nb in enumerate(n_blocks_list) if nb >= 16]
    if not valid_indices:
        valid_indices = list(range(len(s_vals)))

    chosen_idx = valid_indices[-1]
    for idx in valid_indices[:-1]:
        # Require at least 2 blocking steps if series is long, then check plateau condition:
        # s_{k+1} - s_k <= 0.5 * ds_k (growth has flattened well within statistical noise)
        if idx >= 2 and (s_vals[idx + 1] - s_vals[idx]) <= 0.5 * ds_vals[idx]:
            chosen_idx = idx
            break

    return mean_val, float(s_vals[chosen_idx]), int(n_blocks_list[chosen_idx])
