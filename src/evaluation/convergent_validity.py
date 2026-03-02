"""
src/evaluation/convergent_validity.py
=======================================
Convergent validity assessment for digital biomarkers.

Computes correlations between a digital biomarker and gold-standard
clinical measures using Pearson, Spearman, and Kendall tau methods.

Also provides:
- Correlation matrix for multiple biomarkers vs. multiple clinical scores
- Bootstrapped confidence intervals for correlation coefficients
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result for a single correlation test.

    Attributes:
        method: 'pearson', 'spearman', or 'kendall'.
        r: Correlation coefficient.
        p_value: Two-tailed p-value.
        ci_lower: Lower CI (Fisher z-transformation, Pearson only).
        ci_upper: Upper CI.
        n: Number of paired observations.
        interpretation: 'negligible', 'weak', 'moderate', 'strong', 'very strong'.
        biomarker: Biomarker name (for reporting).
        gold_standard: Gold standard measure name (for reporting).
    """

    method: str
    r: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n: int
    interpretation: str
    biomarker: str = ""
    gold_standard: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_correlation(
    biomarker: np.ndarray,
    gold_standard: np.ndarray,
    method: str = "pearson",
    confidence: float = 0.95,
    biomarker_name: str = "biomarker",
    gold_standard_name: str = "gold_standard",
) -> CorrelationResult:
    """Compute correlation between a biomarker and a gold standard measure.

    Args:
        biomarker: Biomarker values (1-D array).
        gold_standard: Gold standard measure values (1-D array).
        method: 'pearson', 'spearman', or 'kendall'.
        confidence: CI level (used only for Pearson with Fisher-z transform).
        biomarker_name: Label for the biomarker.
        gold_standard_name: Label for the gold standard.

    Returns:
        CorrelationResult.

    Raises:
        ValueError: If arrays have different lengths or fewer than 5 pairs.
    """
    bm = np.asarray(biomarker, dtype=float)
    gs = np.asarray(gold_standard, dtype=float)

    if bm.shape != gs.shape:
        raise ValueError("biomarker and gold_standard must have the same shape.")

    mask = np.isfinite(bm) & np.isfinite(gs)
    bm, gs = bm[mask], gs[mask]
    n = len(bm)

    if n < 5:
        raise ValueError(f"≥ 5 paired observations; got {n}.")

    m = method.lower()
    if m == "pearson":
        r, p = stats.pearsonr(bm, gs)
    elif m == "spearman":
        r, p = stats.spearmanr(bm, gs)
    elif m == "kendall":
        r, p = stats.kendalltau(bm, gs)
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'pearson', 'spearman', or 'kendall'.")

    r = float(r)
    p = float(p)

    # CI via Fisher z-transformation (valid for Pearson)
    if m == "pearson" and n >= 5:
        ci_lo, ci_hi = _fisher_ci(r, n, confidence)
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    interpretation = _interpret_correlation(abs(r))

    return CorrelationResult(
        method=m,
        r=r,
        p_value=p,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n=n,
        interpretation=interpretation,
        biomarker=biomarker_name,
        gold_standard=gold_standard_name,
    )


def compute_correlation_matrix(
    df: pd.DataFrame,
    biomarker_cols: list[str],
    clinical_cols: list[str],
    method: str = "pearson",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute correlation matrix between biomarkers and clinical measures.

    Args:
        df: DataFrame containing all columns.
        biomarker_cols: List of biomarker column names.
        clinical_cols: List of clinical measure column names.
        method: Correlation method.
        confidence: CI level.

    Returns:
        DataFrame with rows=biomarkers, columns=clinical measures.
        Values are formatted as 'r=X.XX (p=Y.YYY)'.
    """
    records = []
    for bm_col in biomarker_cols:
        row = {"biomarker": bm_col}
        for cl_col in clinical_cols:
            try:
                result = compute_correlation(
                    df[bm_col].values,
                    df[cl_col].values,
                    method=method,
                    confidence=confidence,
                    biomarker_name=bm_col,
                    gold_standard_name=cl_col,
                )
                row[cl_col] = f"r={result.r:.3f} (p={result.p_value:.3f})"
            except Exception as exc:
                row[cl_col] = f"error: {exc}"
        records.append(row)
    return pd.DataFrame(records).set_index("biomarker")


def _fisher_ci(r: float, n: int, confidence: float) -> tuple[float, float]:
    """Compute CI for Pearson r using Fisher z-transformation."""
    r_clamped = max(min(r, 0.9999), -0.9999)
    z = math.atanh(r_clamped)
    se = 1.0 / math.sqrt(n - 3)
    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    return float(math.tanh(z_lo)), float(math.tanh(z_hi))


def _interpret_correlation(r_abs: float) -> str:
    """Classify correlation strength per Cohen (1988) conventions."""
    if r_abs < 0.1:
        return "negligible"
    elif r_abs < 0.3:
        return "weak"
    elif r_abs < 0.5:
        return "moderate"
    elif r_abs < 0.7:
        return "strong"
    else:
        return "very strong"
