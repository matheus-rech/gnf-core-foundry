"""
src/stability/test_retest.py
==============================
Test-retest reliability assessment combining ICC and Bland-Altman.

Provides a single function that runs both analyses and produces a
comprehensive stability report with a reliability classification.

Reliability classification follows Koo & Mae (2016):
    Excellent : ICC > 0.90
    Good      : 0.75 ≤ ICC ≤ 0.90
    Moderate  : 0.50 ≤ ICC < 0.75
    Poor      : ICC < 0.50
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .bland_altman import BlandAltmanResult, bland_altman_analysis
from .icc import ICCResult, compute_icc
from .mdc import MDCResult, compute_mdc

logger = logging.getLogger(__name__)


@dataclass
class TestRetestResult:
    """Combined test-retest reliability result.

    Attributes:
        biomarker: Name of the biomarker assessed.
        icc_result: ICCResult object.
        ba_result: BlandAltmanResult object.
        mdc_result: MDCResult object.
        reliability_class: 'excellent', 'good', 'moderate', or 'poor'.
        summary: Human-readable summary string.
        passed_threshold: Whether the ICC exceeds the configured threshold.
        icc_threshold: The threshold used for pass/fail.
    """

    biomarker: str
    icc_result: ICCResult
    ba_result: BlandAltmanResult
    mdc_result: MDCResult
    reliability_class: str
    summary: str
    passed_threshold: bool
    icc_threshold: float = 0.75

    def to_dict(self) -> dict:
        """Return a flat summary dict for reporting."""
        return {
            "biomarker": self.biomarker,
            "icc_value": round(self.icc_result.icc_value, 4),
            "icc_type": self.icc_result.icc_type,
            "icc_ci_lower": round(self.icc_result.ci_lower, 4),
            "icc_ci_upper": round(self.icc_result.ci_upper, 4),
            "icc_p": round(self.icc_result.p_value, 4),
            "sem": round(self.mdc_result.sem, 4),
            "mdc95": round(self.mdc_result.mdc95, 4),
            "mdc_pct": round(self.mdc_result.mdc_pct, 2),
            "bias": round(self.ba_result.bias, 4),
            "bias_pct": round(self.ba_result.bias_pct, 2),
            "loa_lower": round(self.ba_result.loa_lower, 4),
            "loa_upper": round(self.ba_result.loa_upper, 4),
            "proportional_bias_p": round(self.ba_result.proportional_bias_p, 4),
            "proportional_bias_present": self.ba_result.proportional_bias_present,
            "reliability_class": self.reliability_class,
            "passed_threshold": self.passed_threshold,
            "icc_threshold": self.icc_threshold,
        }


def assess_test_retest(
    session1: np.ndarray,
    session2: np.ndarray,
    biomarker: str = "biomarker",
    subjects: Optional[np.ndarray] = None,
    icc_type: str = "ICC2k",
    icc_threshold: float = 0.75,
    confidence: float = 0.95,
    use_r: bool = True,
) -> TestRetestResult:
    """Assess test-retest reliability of a biomarker across two sessions.

    Runs ICC computation, Bland-Altman analysis, and MDC calculation
    and returns an integrated reliability report.

    Args:
        session1: Measurements from session 1 (one per subject).
        session2: Measurements from session 2 (one per subject).
        biomarker: Name of the biomarker (for reporting).
        subjects: Optional subject ID array.  If None, sequential IDs are assigned.
        icc_type: ICC form to compute (default ICC2k).
        icc_threshold: Pass/fail threshold for ICC (default 0.75).
        confidence: CI level (default 0.95).
        use_r: Whether to prefer R backend for ICC.

    Returns:
        TestRetestResult with full reliability assessment.

    Raises:
        ValueError: If session1 and session2 have different lengths or < 3 subjects.
    """
    s1 = np.asarray(session1, dtype=float)
    s2 = np.asarray(session2, dtype=float)

    if len(s1) != len(s2):
        raise ValueError("session1 and session2 must have the same length.")
    if len(s1) < 3:
        raise ValueError("≥ 3 subjects for test-retest assessment.")

    n = len(s1)
    subj_ids = subjects if subjects is not None else np.array([f"S{i+1:03d}" for i in range(n)])

    # Build long-format DataFrame for ICC
    df = pd.DataFrame(
        {
            "subject": np.concatenate([subj_ids, subj_ids]),
            "session": ["session1"] * n + ["session2"] * n,
            "value": np.concatenate([s1, s2]),
        }
    )

    # ICC
    icc_result = compute_icc(
        data=df,
        subjects="subject",
        raters="session",
        measurements="value",
        icc_type=icc_type,
        confidence=confidence,
        use_r=use_r,
    )

    # Bland-Altman
    ba_result = bland_altman_analysis(s1, s2, confidence=confidence)

    # MDC
    all_values = np.concatenate([s1, s2])
    mdc_result = compute_mdc(all_values, icc=icc_result.icc_value)

    # Classification
    reliability_class = _classify_reliability(icc_result.icc_value)
    passed = icc_result.icc_value >= icc_threshold

    summary = (
        f"Test-retest reliability for '{biomarker}': "
        f"{reliability_class.upper()} (ICC={icc_result.icc_value:.3f}, "
        f"95% CI [{icc_result.ci_lower:.3f}, {icc_result.ci_upper:.3f}]). "
        f"Bias = {ba_result.bias:.3f} ({ba_result.bias_pct:.1f}% of mean). "
        f"MDC95 = {mdc_result.mdc95:.3f} ({mdc_result.mdc_pct:.1f}% of mean). "
        f"{'PASS' if passed else 'FAIL'} (threshold ICC ≥ {icc_threshold})."
    )

    logger.info(summary)

    return TestRetestResult(
        biomarker=biomarker,
        icc_result=icc_result,
        ba_result=ba_result,
        mdc_result=mdc_result,
        reliability_class=reliability_class,
        summary=summary,
        passed_threshold=passed,
        icc_threshold=icc_threshold,
    )


def _classify_reliability(icc: float) -> str:
    """Classify reliability per Koo & Mae (2016).

    Args:
        icc: ICC value.

    Returns:
        'excellent', 'good', 'moderate', or 'poor'.
    """
    if icc >= 0.90:
        return "excellent"
    elif icc >= 0.75:
        return "good"
    elif icc >= 0.50:
        return "moderate"
    else:
        return "poor"
