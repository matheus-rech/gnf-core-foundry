"""
src/feature_engine/kinematic_features.py
==========================================
Kinematic feature extraction for movement analysis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal, integrate

logger = logging.getLogger(__name__)


@dataclass
class KinematicFeatures:
    """Container for kinematic features."""

    peak_velocity: float
    mean_velocity: float
    velocity_rms: float
    peak_acceleration: float
    mean_acceleration: float
    acceleration_rms: float
    peak_jerk: float
    mean_jerk: float
    jerk_rms: float
    sparc: float
    ldlj: float
    range_of_motion: float
    path_length: float
    normalised_path_length: float
    movement_duration_s: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


def extract_kinematic_features(
    position_or_accel: np.ndarray,
    fs: float,
    signal_type: str = "acceleration",
    sparc_cutoff: float = 20.0,
    sparc_padlevel: int = 4,
) -> KinematicFeatures:
    """Extract kinematic features from a 1-D position or acceleration signal."""
    arr = np.asarray(position_or_accel, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    if n < 4:
        raise ValueError(f"Signal too short ({n} samples) for kinematic analysis.")

    dt = 1.0 / fs
    duration_s = n * dt

    if signal_type == "position":
        pos = arr
        vel = np.gradient(pos, dt)
        accel = np.gradient(vel, dt)
    elif signal_type == "velocity":
        vel = arr
        accel = np.gradient(vel, dt)
        pos = integrate.cumulative_trapezoid(vel, dx=dt, initial=0.0)
    elif signal_type == "acceleration":
        accel = arr
        vel = integrate.cumulative_trapezoid(accel, dx=dt, initial=0.0)
        pos = integrate.cumulative_trapezoid(vel, dx=dt, initial=0.0)
    else:
        raise ValueError(f"Unknown signal_type: '{signal_type}'.")

    jerk = np.gradient(accel, dt)

    peak_vel = float(np.max(np.abs(vel)))
    mean_vel = float(np.mean(np.abs(vel)))
    vel_rms = float(np.sqrt(np.mean(vel ** 2)))
    peak_accel = float(np.max(np.abs(accel)))
    mean_accel = float(np.mean(np.abs(accel)))
    accel_rms = float(np.sqrt(np.mean(accel ** 2)))
    peak_jerk = float(np.max(np.abs(jerk)))
    mean_jerk = float(np.mean(np.abs(jerk)))
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    rom = float(np.ptp(pos))
    path_len = float(np.sum(np.abs(np.diff(pos))))
    displacement = abs(float(pos[-1] - pos[0]))
    norm_path = path_len / displacement if displacement > 0 else float("nan")

    sparc_val = _compute_sparc(vel, fs, fc=sparc_cutoff, padlevel=sparc_padlevel)
    ldlj_val = _compute_ldlj(vel, fs)

    return KinematicFeatures(
        peak_velocity=peak_vel, mean_velocity=mean_vel, velocity_rms=vel_rms,
        peak_acceleration=peak_accel, mean_acceleration=mean_accel, acceleration_rms=accel_rms,
        peak_jerk=peak_jerk, mean_jerk=mean_jerk, jerk_rms=jerk_rms,
        sparc=sparc_val, ldlj=ldlj_val, range_of_motion=rom,
        path_length=path_len, normalised_path_length=norm_path,
        movement_duration_s=duration_s, n_samples=n,
    )


def _compute_sparc(velocity, fs, fc=20.0, padlevel=4, amp_threshold=0.05):
    vel = np.asarray(velocity, dtype=float)
    n = len(vel)
    nfft = int(2 ** (math.ceil(math.log2(n)) + padlevel))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(vel, n=nfft)) / n
    mag_norm = mag / (mag[0] + 1e-12) if mag[0] > 0 else mag
    fc_idx = np.searchsorted(f, fc) + 1
    amp_idx = np.where(mag_norm < amp_threshold)[0]
    amp_cutoff = amp_idx[0] if len(amp_idx) > 0 else len(f)
    stop_idx = min(fc_idx, amp_cutoff, len(f))
    f_sub = f[:stop_idx]
    m_sub = mag_norm[:stop_idx]
    if len(f_sub) < 2:
        return 0.0
    df = f_sub[1] - f_sub[0]
    dm = np.diff(m_sub)
    arc = -float(np.sum(np.sqrt((df / fc) ** 2 + (dm / df * df / 1.0) ** 2)))
    return arc


def _compute_ldlj(velocity, fs):
    vel = np.asarray(velocity, dtype=float)
    dt = 1.0 / fs
    T = len(vel) * dt
    D = float(np.max(np.abs(vel)))
    if D == 0 or T == 0:
        return 0.0
    jerk = np.gradient(vel, dt)
    jerk_sq_integral = float(np.trapz(jerk ** 2, dx=dt))
    if jerk_sq_integral <= 0:
        return 0.0
    ldlj = -math.log((T ** 3 / D ** 2) * jerk_sq_integral)
    return ldlj
