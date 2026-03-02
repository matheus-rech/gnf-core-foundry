"""
src/feature_engine/spectral_features.py
==========================================
Spectral domain feature extraction using FFT and power spectral density.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
    "tremor": (3.0, 8.0),
    "gait": (0.5, 3.0),
}


@dataclass
class SpectralFeatures:
    """Container for spectral domain features."""

    dominant_frequency: float
    mean_frequency: float
    median_frequency: float
    spectral_entropy: float
    spectral_centroid: float
    total_power: float
    band_powers: dict[str, float]
    relative_band_powers: dict[str, float]
    psd_freqs: np.ndarray
    psd_values: np.ndarray
    fs: float

    def to_dict(self) -> dict:
        d: dict = {
            "dominant_frequency": self.dominant_frequency,
            "mean_frequency": self.mean_frequency,
            "median_frequency": self.median_frequency,
            "spectral_entropy": self.spectral_entropy,
            "spectral_centroid": self.spectral_centroid,
            "total_power": self.total_power,
        }
        d.update({f"power_{k}": v for k, v in self.band_powers.items()})
        d.update({f"rel_power_{k}": v for k, v in self.relative_band_powers.items()})
        return d


def extract_spectral_features(
    signal_data: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    freq_bands: Optional[dict[str, tuple[float, float]]] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> SpectralFeatures:
    """Extract spectral features using Welch's PSD method."""
    arr = np.asarray(signal_data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    if n < 4:
        raise ValueError(f"Signal too short ({n} samples) for spectral analysis.")

    nyq = fs / 2.0
    fmax_use = fmax if fmax is not None else nyq
    bands = freq_bands if freq_bands is not None else FREQ_BANDS

    seg = min(nperseg or 256, n)
    ov = noverlap if noverlap is not None else seg // 2

    freqs, psd = signal.welch(arr, fs=fs, nperseg=seg, noverlap=ov)

    freq_mask = (freqs >= fmin) & (freqs <= fmax_use)
    freqs_r = freqs[freq_mask]
    psd_r = psd[freq_mask]

    if len(psd_r) == 0:
        raise ValueError(f"No PSD components in [{fmin}, {fmax_use}] Hz range.")

    total_power = float(np.trapz(psd_r, freqs_r))
    if total_power <= 0:
        total_power = float(np.sum(psd_r))

    dom_freq = float(freqs_r[np.argmax(psd_r)])

    if total_power > 0:
        mean_freq = float(np.sum(freqs_r * psd_r) / np.sum(psd_r))
        spectral_centroid = mean_freq
    else:
        mean_freq = dom_freq
        spectral_centroid = dom_freq

    cumpower = np.cumsum(psd_r)
    half_power = cumpower[-1] / 2.0
    median_idx = np.searchsorted(cumpower, half_power)
    median_idx = min(median_idx, len(freqs_r) - 1)
    median_freq = float(freqs_r[median_idx])

    psd_norm = psd_r / (np.sum(psd_r) + 1e-12)
    entropy = -float(np.sum(psd_norm * np.log(psd_norm + 1e-12)))
    max_entropy = float(np.log(len(psd_norm)))
    spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    band_powers: dict[str, float] = {}
    relative_band_powers: dict[str, float] = {}
    for band_name, (bf_lo, bf_hi) in bands.items():
        band_mask = (freqs >= bf_lo) & (freqs <= bf_hi)
        band_psd = psd[band_mask]
        band_f = freqs[band_mask]
        if len(band_psd) > 1:
            bp = float(np.trapz(band_psd, band_f))
        else:
            bp = float(np.sum(band_psd))
        band_powers[band_name] = max(bp, 0.0)
        relative_band_powers[band_name] = bp / total_power if total_power > 0 else 0.0

    return SpectralFeatures(
        dominant_frequency=dom_freq, mean_frequency=mean_freq,
        median_frequency=median_freq, spectral_entropy=spectral_entropy,
        spectral_centroid=spectral_centroid, total_power=total_power,
        band_powers=band_powers, relative_band_powers=relative_band_powers,
        psd_freqs=freqs_r, psd_values=psd_r, fs=fs,
    )


def compute_band_power(
    signal_data: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    relative: bool = False,
    nperseg: Optional[int] = None,
) -> float:
    """Compute power in a single frequency band."""
    arr = np.asarray(signal_data, dtype=float)
    seg = min(nperseg or 256, len(arr))
    freqs, psd = signal.welch(arr, fs=fs, nperseg=seg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = float(np.trapz(psd[mask], freqs[mask])) if mask.sum() > 1 else float(np.sum(psd[mask]))
    if relative:
        total = float(np.trapz(psd, freqs))
        return band_power / total if total > 0 else 0.0
    return max(band_power, 0.0)
