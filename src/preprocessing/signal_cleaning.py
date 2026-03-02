"""
src/preprocessing/signal_cleaning.py

Signal cleaning utilities for wearable sensor data.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)


def butterworth_filter(
    data: np.ndarray,
    fs: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    order: int = 4,
    filter_type: Literal["bandpass", "lowpass", "highpass", "bandstop"] = "bandpass",
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply a Butterworth filter to a 1-D signal."""
    arr = np.asarray(data, dtype=float)
    nyq = 0.5 * fs

    if filter_type in ("bandpass", "bandstop"):
        if lowcut is None or highcut is None:
            raise ValueError(f"Both lowcut and highcut are required for '{filter_type}' filter.")
        if not (0 < lowcut < highcut < nyq):
            raise ValueError(
                f"Invalid cutoff frequencies: lowcut={lowcut}, highcut={highcut}, "
                f"Nyquist={nyq}. Require 0 < lowcut < highcut < Nyquist."
            )
        Wn = [lowcut / nyq, highcut / nyq]
    elif filter_type == "lowpass":
        if highcut is None:
            raise ValueError("highcut is required for lowpass filter.")
        if not (0 < highcut < nyq):
            raise ValueError(f"Invalid highcut={highcut}; must be 0 < highcut < Nyquist={nyq}.")
        Wn = highcut / nyq
    elif filter_type == "highpass":
        if lowcut is None:
            raise ValueError("lowcut is required for highpass filter.")
        if not (0 < lowcut < nyq):
            raise ValueError(f"Invalid lowcut={lowcut}; must be 0 < lowcut < Nyquist={nyq}.")
        Wn = lowcut / nyq
    else:
        raise ValueError(f"Unknown filter_type: '{filter_type}'.")

    sos = signal.butter(order, Wn, btype=filter_type, output="sos")
    filtered = signal.sosfiltfilt(sos, arr) if zero_phase else signal.sosfilt(sos, arr)
    return filtered.astype(float)


def moving_average(data: np.ndarray, window: int = 5, mode: Literal["valid", "same", "full"] = "same") -> np.ndarray:
    """Apply a simple symmetric moving average."""
    arr = np.asarray(data, dtype=float)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}.")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode=mode)


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using a z-score threshold."""
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr, ddof=1)
    if std == 0:
        return np.zeros(len(arr), dtype=bool)
    z = np.abs((arr - mean) / std)
    return z > threshold


def detect_outliers_iqr(data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """Detect outliers using the IQR (Tukey fence) method."""
    arr = np.asarray(data, dtype=float)
    q1, q3 = np.nanpercentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return (arr < lower) | (arr > upper)


def remove_outliers(
    data: np.ndarray,
    method: Literal["zscore", "iqr"] = "zscore",
    threshold: float = 3.0,
    replace_with: Union[float, Literal["nan", "clip"]] = "nan",
) -> np.ndarray:
    """Remove or replace outliers in a 1-D array."""
    arr = np.asarray(data, dtype=float).copy()
    if method == "zscore":
        mask = detect_outliers_zscore(arr, threshold=threshold)
    elif method == "iqr":
        mask = detect_outliers_iqr(arr, iqr_factor=threshold)
    else:
        raise ValueError(f"Unknown method: '{method}'.")

    n_outliers = int(np.sum(mask))
    if n_outliers > 0:
        logger.debug("Detected %d outlier(s) using %s method.", n_outliers, method)

    if replace_with == "nan":
        arr[mask] = np.nan
    elif replace_with == "clip":
        q1, q3 = np.nanpercentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        arr = np.clip(arr, lower, upper)
    else:
        arr[mask] = float(replace_with)

    return arr


def interpolate_missing(
    data: np.ndarray,
    method: Literal["linear", "cubic", "ffill", "bfill"] = "linear",
    max_gap: Optional[int] = None,
) -> np.ndarray:
    """Interpolate NaN values in a 1-D array."""
    arr = np.asarray(data, dtype=float)
    n_nan = int(np.sum(np.isnan(arr)))
    if n_nan == 0:
        return arr
    logger.debug("Interpolating %d NaN value(s) using '%s' method.", n_nan, method)
    s = pd.Series(arr)
    if method in ("ffill", "bfill"):
        filled = s.fillna(method=method, limit=max_gap)  # type: ignore[arg-type]
    else:
        pd_method = "linear" if method == "linear" else "cubic"
        filled = s.interpolate(method=pd_method, limit=max_gap, limit_direction="both")
    return filled.values.astype(float)


def clean_signal(
    data: np.ndarray,
    fs: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    filter_order: int = 4,
    filter_type: str = "lowpass",
    outlier_method: Optional[str] = "zscore",
    outlier_threshold: float = 3.0,
    smooth_window: Optional[int] = None,
    interpolate_method: str = "linear",
    interpolate_max_gap: Optional[int] = None,
) -> np.ndarray:
    """Full signal cleaning pipeline."""
    arr = np.asarray(data, dtype=float)
    if outlier_method is not None:
        arr = remove_outliers(arr, method=outlier_method, threshold=outlier_threshold)
    if np.any(np.isnan(arr)):
        arr = interpolate_missing(arr, method=interpolate_method, max_gap=interpolate_max_gap)
    if lowcut is not None or highcut is not None:
        arr = butterworth_filter(arr, fs=fs, lowcut=lowcut, highcut=highcut, order=filter_order, filter_type=filter_type)
    if smooth_window is not None and smooth_window > 1:
        arr = moving_average(arr, window=smooth_window)
    return arr
