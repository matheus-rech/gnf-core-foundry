"""
src/preprocessing/normalization.py
=====================================
Normalization utilities for biomarker feature data.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def zscore_normalize(
    data: Union[np.ndarray, pd.DataFrame, pd.Series],
    columns: Optional[list[str]] = None,
    ddof: int = 1,
) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """Standardise data to zero mean and unit variance."""
    if isinstance(data, pd.Series):
        return (data - data.mean()) / data.std(ddof=ddof)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        cols = columns or df.select_dtypes(include="number").columns.tolist()
        for col in cols:
            mu = df[col].mean()
            sd = df[col].std(ddof=ddof)
            if sd == 0:
                logger.warning("Column '%s' has zero variance; z-score will be 0.", col)
                df[col] = 0.0
            else:
                df[col] = (df[col] - mu) / sd
        return df

    arr = np.asarray(data, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=ddof)
    if sd == 0:
        logger.warning("Array has zero variance; z-score will be 0.")
        return np.zeros_like(arr)
    return (arr - mu) / sd


def minmax_scale(
    data: Union[np.ndarray, pd.DataFrame, pd.Series],
    feature_range: tuple[float, float] = (0.0, 1.0),
    columns: Optional[list[str]] = None,
) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """Scale data to a specified range."""
    a, b = feature_range

    if isinstance(data, pd.Series):
        mn, mx = data.min(), data.max()
        if mn == mx:
            return data * 0.0 + a
        return a + (data - mn) / (mx - mn) * (b - a)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        cols = columns or df.select_dtypes(include="number").columns.tolist()
        for col in cols:
            mn, mx = df[col].min(), df[col].max()
            if mn == mx:
                df[col] = a
            else:
                df[col] = a + (df[col] - mn) / (mx - mn) * (b - a)
        return df

    arr = np.asarray(data, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mn == mx:
        return np.full_like(arr, a)
    return a + (arr - mn) / (mx - mn) * (b - a)


def baseline_normalize(
    data: pd.DataFrame,
    value_col: str,
    session_col: str,
    subject_col: str,
    baseline_session: str,
    method: str = "ratio",
) -> pd.DataFrame:
    """Normalise all sessions relative to a baseline session."""
    if method not in ("ratio", "difference"):
        raise ValueError(f"method must be 'ratio' or 'difference'; got '{method}'.")

    df = data.copy()
    baseline = (
        df[df[session_col] == baseline_session]
        .groupby(subject_col)[value_col]
        .mean()
        .rename("baseline_value")
    )
    df = df.join(baseline, on=subject_col)

    if method == "ratio":
        df[f"{value_col}_normalised"] = df[value_col] / df["baseline_value"]
    else:
        df[f"{value_col}_normalised"] = df[value_col] - df["baseline_value"]

    df.drop(columns=["baseline_value"], inplace=True)
    logger.info("Baseline normalisation ('%s') applied relative to session '%s'.", method, baseline_session)
    return df


def group_zscore_normalize(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Apply z-score normalisation within each group."""
    df = data.copy()
    df[f"{value_col}_zscore"] = (
        df.groupby(group_col)[value_col]
        .transform(lambda x: (x - x.mean()) / (x.std(ddof=1) if x.std(ddof=1) != 0 else 1))
    )
    return df
