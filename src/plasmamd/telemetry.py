"""Real-time CPU and GPU hardware telemetry collector for the MD pipeline."""

from __future__ import annotations

import contextlib
import csv
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numba
import numpy as np
import psutil

try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

FIG_WIDTH_IN = 160.0 / 25.4
FIG_DPI = 300


@dataclass(frozen=True)
class TelemetrySample:
    """One snapshot of system load during the MD simulation."""

    elapsed_s: float
    stage: str
    cpu_total_pct: float
    cpu_active_cores: int
    cpu_max_core_pct: float
    ram_rss_mb: float
    gpu_util_pct: float
    gpu_mem_util_pct: float
    gpu_vram_mb: float
    gpu_power_w: float
    gpu_temp_c: float


class SystemMonitor:
    """Background daemon thread polling CPU (psutil) and GPU (NVML) metrics in real time."""

    def __init__(self, interval: float = 0.25, active_core_threshold: float = 20.0) -> None:
        self.interval = interval
        self.threshold = active_core_threshold
        self.samples: list[TelemetrySample] = []
        self._stage = "init"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())
        self.logical_cores = psutil.cpu_count(logical=True) or 1
        self.physical_cores = psutil.cpu_count(logical=False) or 1
        self.numba_threads = int(numba.get_num_threads())
        self.gpu_name = "N/A"
        self._gpu_handle: Any = None
        self._t0 = 0.0

    def set_stage(self, stage_name: str) -> None:
        """Tag subsequent telemetry samples with a human-readable pipeline stage."""
        self._stage = stage_name

    def __enter__(self) -> SystemMonitor:
        psutil.cpu_percent(percpu=True)
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                self.gpu_name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            except Exception:
                self._gpu_handle = None

        self._stop_event.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if _HAS_NVML and self._gpu_handle is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    def _record_once(self) -> None:
        elapsed = time.perf_counter() - self._t0
        per_core = psutil.cpu_percent(percpu=True)
        cpu_tot = float(np.mean(per_core)) if per_core else 0.0
        active_cores = sum(1 for c in per_core if c >= self.threshold)
        max_core = float(max(per_core)) if per_core else 0.0
        ram_mb = self._proc.memory_info().rss / (1024.0 * 1024.0)

        gpu_util, gpu_mem_util, vram_mb, power_w, temp_c = 0.0, 0.0, 0.0, 0.0, 0.0
        if self._gpu_handle is not None:
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = float(rates.gpu)
                gpu_mem_util = float(rates.memory)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_mb = float(mem_info.used) / (1024.0 * 1024.0)
                power_w = float(pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)) / 1000.0
                temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass

        self.samples.append(
            TelemetrySample(
                elapsed_s=elapsed,
                stage=self._stage,
                cpu_total_pct=cpu_tot,
                cpu_active_cores=active_cores,
                cpu_max_core_pct=max_core,
                ram_rss_mb=ram_mb,
                gpu_util_pct=gpu_util,
                gpu_mem_util_pct=gpu_mem_util,
                gpu_vram_mb=vram_mb,
                gpu_power_w=power_w,
                gpu_temp_c=temp_c,
            )
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._record_once()
            self._stop_event.wait(self.interval)
        if not self.samples:
            self._record_once()

    def print_summary(self) -> None:
        """Print hardware summary table to stdout."""
        if not self.samples:
            return
        cpu_arr = np.array([s.cpu_total_pct for s in self.samples])
        cores_arr = np.array([s.cpu_active_cores for s in self.samples])
        ram_arr = np.array([s.ram_rss_mb for s in self.samples])
        gpu_arr = np.array([s.gpu_util_pct for s in self.samples])
        vram_arr = np.array([s.gpu_vram_mb for s in self.samples])
        pwr_arr = np.array([s.gpu_power_w for s in self.samples])

        print("=" * 76)
        print(
            f"TELEMETRY SUMMARY | CPU: {self.physical_cores}C/{self.logical_cores}T "
            f"(Numba threads={self.numba_threads}) | GPU: {self.gpu_name}"
        )
        print("-" * 76)
        print(
            f"  CPU Load (%):       mean = {cpu_arr.mean():5.1f}% | peak = {cpu_arr.max():5.1f}%"
        )
        print(
            f"  Active CPU Cores:   mean = {cores_arr.mean():5.1f}  | max  = {cores_arr.max():5d} "
            f"(>{self.threshold:.0f}% load)"
        )
        print(f"  Process RAM (MB):   peak = {ram_arr.max():6.1f} MB")
        print(
            f"  GPU Core Load (%):  mean = {gpu_arr.mean():5.1f}% | peak = {gpu_arr.max():5.1f}%"
        )
        print(
            f"  GPU VRAM & Power:   peak = {vram_arr.max():6.1f} MB | peak = {pwr_arr.max():5.1f} W"
        )
        print("=" * 76)

    def save_csv(self, path: Path) -> Path:
        """Write full time series of hardware metrics to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(TelemetrySample.__dataclass_fields__)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for s in self.samples:
                writer.writerow([getattr(s, name) for name in fields])
        return path

    def plot(self, stem: Path) -> list[Path]:
        """Plot 2x2 diagnostic figure of CPU cores, GPU utilization, memory, and power."""
        if not self.samples:
            return []
        t = np.array([s.elapsed_s for s in self.samples])
        cores = np.array([s.cpu_active_cores for s in self.samples])
        cpu_pct = np.array([s.cpu_total_pct for s in self.samples])
        gpu_pct = np.array([s.gpu_util_pct for s in self.samples])
        gpu_mem_pct = np.array([s.gpu_mem_util_pct for s in self.samples])
        ram_mb = np.array([s.ram_rss_mb for s in self.samples])
        vram_mb = np.array([s.gpu_vram_mb for s in self.samples])
        pwr_w = np.array([s.gpu_power_w for s in self.samples])

        fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 0.72 * FIG_WIDTH_IN), sharex=True)
        ax1, ax2, ax3, ax4 = axes.flat

        ax1.plot(t, cores, "-", color="black", label="Активные ядра CPU")
        ax1.set_ylabel("Активные ядра")
        ax1.set_ylim(0, max(self.logical_cores + 1, int(cores.max()) + 2))
        ax1.legend(frameon=False, loc="upper right")

        ax2.plot(t, gpu_pct, "-", color="black", label="Ядро GPU (%)")
        ax2.plot(t, gpu_mem_pct, "--", color="0.4", label="Память GPU (%)")
        ax2.plot(t, cpu_pct, ":", color="0.2", label="CPU (%)")
        ax2.set_ylabel("Загрузка (%)")
        ax2.set_ylim(0, 105)
        ax2.legend(frameon=False, loc="upper right")

        ax3.plot(t, ram_mb, "-", color="black", label="RAM (МБ)")
        ax3.plot(t, vram_mb, "--", color="0.4", label="VRAM (МБ)")
        ax3.set_xlabel("Время (с)")
        ax3.set_ylabel("Память (МБ)")
        ax3.legend(frameon=False, loc="upper right")

        ax4.plot(t, pwr_w, "-", color="black", label="Мощность GPU (Вт)")
        ax4.set_xlabel("Время (с)")
        ax4.set_ylabel("Мощность (Вт)")
        ax4.legend(frameon=False, loc="upper right")

        fig.tight_layout()
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
        fig.savefig(paths[0])
        fig.savefig(paths[1], dpi=FIG_DPI)
        plt.close(fig)
        return paths
