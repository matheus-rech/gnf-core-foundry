"""
src/validation/data_quality.py
================================
Data quality reporting for biomarker datasets.

Generates comprehensive quality reports covering:
- Completeness (missing data rates per column and per subject)
- Consistency (duplicate records, out-of-range values)
- Plausibility (statistical anomalies, flag suspicious values)
- Signal quality (if a quality_flag column is present)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report.

    Attributes:
        n_rows: Total number of rows.
        n_columns: Total number of columns.
        completeness_pct: Overall completeness percentage.
        column_completeness: Dict of column → completeness pct.
        n_duplicates: Number of duplicate rows.
        suspicious_rows: Indices of suspicious rows.
        quality_flag_summary: Counts of quality flags if present.
        plausibility_flags: Dict of column → number of plausibility violations.
        summary_stats: Dict of summary statistics per numeric column.
        issues: List of data quality issue strings.
        warnings: List of warning strings.
    """

    n_rows: int
    n_columns: int
    completeness_pct: float
    column_completeness: dict[str, float] = field(default_factory=dict)
    n_duplicates: int = 0
    suspicious_rows: list[int] = field(default_factory=list)
    quality_flag_summary: dict[str, int] = field(default_factory=dict)
    plausibility_flags: dict[str, int] = field(default_factory=dict)
    summary_stats: dict[str, dict] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"DataQualityReport(rows={self.n_rows}, completeness={self.completeness_pct:.1f}%, "
            f"duplicates={self.n_duplicates}, issues={len(self.issues)})"
        )

    def to_dict(self) -> dict:
        """Return JSON-serialisable dict."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "completeness_pct": round(self.completeness_pct, 2),
            "n_duplicates": self.n_duplicates,
            "n_suspicious_rows": len(self.suspicious_rows),
            "quality_flag_summary": self.quality_flag_summary,
            "plausibility_flags": self.plausibility_flags,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def assess_data_quality(
    df: pd.DataFrame,
    value_columns: Optional[list[str]] = None,
    quality_flag_col: Optional[str] = "quality_flag",
    z_score_threshold: float = 4.0,
    expected_ranges: Optional[dict[str, tuple[float, float]]] = None,
) -> DataQualityReport:
    """Generate a comprehensive data quality report.

    Args:
        df: Input DataFrame.
        value_columns: Numeric columns to assess.  Defaults to all numeric columns.
        quality_flag_col: Column containing quality flags ('valid'/'suspect'/'missing').
            Set to None to skip.
        z_score_threshold: Z-score threshold for identifying statistical anomalies.
        expected_ranges: Dict of column → (min, max) expected range.

    Returns:
        DataQualityReport.
    """
    n_rows, n_cols = df.shape
    issues: list[str] = []
    warnings_list: list[str] = []

    # --- Completeness ---
    col_completeness = {}
    for col in df.columns:
        pct = float((1 - df[col].isna().mean()) * 100)
        col_completeness[col] = round(pct, 2)

    overall_completeness = float((1 - df.isna().mean().mean()) * 100)

    # --- Duplicates ---
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        issues.append(f"{n_dups} duplicate row(s) found.")

    # --- Quality flags ---
    qf_summary: dict[str, int] = {}
    if quality_flag_col and quality_flag_col in df.columns:
        qf_summary = df[quality_flag_col].value_counts().to_dict()
        n_suspect = qf_summary.get("suspect", 0) + qf_summary.get("missing", 0)
        if n_suspect > 0:
            warnings_list.append(
                f"{n_suspect} records flagged as 'suspect' or 'missing'."
            )

    # --- Value columns ---
    num_cols = value_columns or df.select_dtypes(include="number").columns.tolist()
    plausibility_flags: dict[str, int] = {}
    summary_stats: dict[str, dict] = {}
    suspicious_indices: set[int] = set()

    for col in num_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            warnings_list.append(f"Column '{col}' is entirely null.")
            continue

        # Summary stats
        summary_stats[col] = {
            "mean": round(float(series.mean()), 6),
            "std": round(float(series.std(ddof=1)), 6),
            "min": round(float(series.min()), 6),
            "p25": round(float(series.quantile(0.25)), 6),
            "median": round(float(series.median()), 6),
            "p75": round(float(series.quantile(0.75)), 6),
            "max": round(float(series.max()), 6),
            "n_null": int(df[col].isna().sum()),
            "n_zero": int((series == 0).sum()),
        }

        # Plausibility: z-score outliers
        mu = float(series.mean())
        sd = float(series.std(ddof=1))
        if sd > 0:
            z_scores = np.abs((series.values - mu) / sd)
            n_extreme = int(np.sum(z_scores > z_score_threshold))
            if n_extreme > 0:
                plausibility_flags[col] = n_extreme
                extreme_idx = series.index[z_scores > z_score_threshold].tolist()
                suspicious_indices.update(extreme_idx)
                warnings_list.append(
                    f"Column '{col}': {n_extreme} value(s) exceed z={z_score_threshold}."
                )

        # Expected range check
        if expected_ranges and col in expected_ranges:
            rmin, rmax = expected_ranges[col]
            out_of_range = int(((series < rmin) | (series > rmax)).sum())
            if out_of_range > 0:
                issues.append(
                    f"Column '{col}': {out_of_range} value(s) outside expected range "
                    f"[{rmin}, {rmax}]."
                )

    return DataQualityReport(
        n_rows=n_rows,
        n_columns=n_cols,
        completeness_pct=overall_completeness,
        column_completeness=col_completeness,
        n_duplicates=n_dups,
        suspicious_rows=list(suspicious_indices),
        quality_flag_summary=qf_summary,
        plausibility_flags=plausibility_flags,
        summary_stats=summary_stats,
        issues=issues,
        warnings=warnings_list,
    )
