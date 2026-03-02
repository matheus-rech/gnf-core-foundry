"""
src/feature_engine/temporal_features.py
==========================================
Temporal domain feature extraction from wearable sensor time-series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal, stats

logger = logging.getLogger(__name__)


@dataclass
class TemporalFeatures:
    """Container for temporal domain features."""

    mean: float
    median: float
    variance: float
    std: float
    rms: float
    skewness: float
    kurtosis: float
    zero_crossing_rate: float
    peak_count: int
    peak_rate: float
    peak_mean_height: float
    signal_range: float
    iqr: float
    cv: float
    autocorr_lag1: float
    signal_length: int
    duration_s: float

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


def extract_temporal_features(
    signal_data: np.ndarray,
    fs: float,
    peak_prominence: float = 0.1,
    peak_distance: Optional[int] = None,
) -> TemporalFeatures:
    """Extract temporal domain features from a 1-D time-series signal."""
    arr = np.asarray(signal_data, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 3:
        raise ValueError(f"Signal has fewer than 3 finite samples; got {len(arr)}.")

    n = len(arr)
    duration_s = n / fs

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    var_val = float(np.var(arr, ddof=1))
    std_val = float(np.std(arr, ddof=1))
    rms_val = float(np.sqrt(np.mean(arr ** 2)))
    skew_val = float(stats.skew(arr))
    kurt_val = float(stats.kurtosis(arr))

    zero_crossings = np.where(np.diff(np.sign(arr - mean_val)))[0]
    zcr = float(len(zero_crossings) / duration_s)

    pk_dist = peak_distance if peak_distance is not None else max(1, int(fs * 0.1))
    peaks, peak_props = signal.find_peaks(arr, prominence=peak_prominence, distance=pk_dist)
    peak_count = int(len(peaks))
    peak_rate = float(peak_count / duration_s) if duration_s > 0 else 0.0
    peak_heights = arr[peaks] if peak_count > 0 else np.array([float("nan")])
    peak_mean_h = float(np.mean(peak_heights))

    sig_range = float(np.ptp(arr))
    iqr_val = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    cv = (std_val / abs(mean_val)) if mean_val != 0 else float("nan")

    if n > 2:
        ac = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        autocorr_lag1 = float(ac)
    else:
        autocorr_lag1 = float("nan")

    return TemporalFeatures(
        mean=mean_val, median=median_val, variance=var_val, std=std_val,
        rms=rms_val, skewness=skew_val, kurtosis=kurt_val,
        zero_crossing_rate=zcr, peak_count=peak_count, peak_rate=peak_rate,
        peak_mean_height=peak_mean_h, signal_range=sig_range, iqr=iqr_val,
        cv=cv, autocorr_lag1=autocorr_lag1, signal_length=n, duration_s=duration_s,
    )


def extract_temporal_features_multiaxis(
    data: np.ndarray,
    fs: float,
    axis_names: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, TemporalFeatures]:
    """Extract temporal features from a multi-axis signal."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_axes = arr.shape[1]
    names = axis_names if axis_names is not None else [f"axis_{i}" for i in range(n_axes)]
    if len(names) != n_axes:
        raise ValueError(f"axis_names length ({len(names)}) != n_axes ({n_axes}).")
    results = {}
    for i, name in enumerate(names):
        results[name] = extract_temporal_features(arr[:, i], fs=fs, **kwargs)
    return results
