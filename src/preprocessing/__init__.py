"""Preprocessing module — signal cleaning and normalization."""

from .signal_cleaning import (
    butterworth_filter,
    clean_signal,
    detect_outliers_iqr,
    detect_outliers_zscore,
    interpolate_missing,
    moving_average,
    remove_outliers,
)
from .normalization import (
    baseline_normalize,
    group_zscore_normalize,
    minmax_scale,
    zscore_normalize,
)

__all__ = [
    "butterworth_filter",
    "clean_signal",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "interpolate_missing",
    "moving_average",
    "remove_outliers",
    "baseline_normalize",
    "group_zscore_normalize",
    "minmax_scale",
    "zscore_normalize",
]
