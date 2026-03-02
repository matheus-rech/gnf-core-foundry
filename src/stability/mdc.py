"""
src/stability/mdc.py
=====================
Minimal Detectable Change (MDC) and related clinimetric statistics.

Provides:
- MDC at 90%, 95%, and 99% confidence levels
- MDC as percentage of mean (MDC%)
- Smallest Real Difference (SRD)
- Standard Error of Measurement (SEM)

Reference:
    Weir JP (2005). Quantifying test-retest reliability using the intraclass
    correlation coefficient and the SEM. J Strength Cond Res, 19(1), 231–240.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MDCResult:
    """Container for MDC statistics.

    Attributes:
        sem: Standard Error of Measurement.
        mdc90: MDC at 90% confidence (z=1.645).
        mdc95: MDC at 95% confidence (z=1.96).
        mdc99: MDC at 99% confidence (z=2.576).
        mdc_pct: MDC95 as percentage of grand mean.
        srd: Smallest Real Difference (= MDC95).
        grand_mean: Grand mean of all measurements.
        icc_used: ICC value used for SEM computation.
        sd_total: Total standard deviation of measurements.
    """

    sem: float
    mdc90: float
    mdc95: float
    mdc99: float
    mdc_pct: float
    srd: float
    grand_mean: float
    icc_used: float
    sd_total: float

    def __repr__(self) -> str:
        return (
            f"MDCResult(SEM={self.sem:.4f}, "
            f"MDC90={self.mdc90:.4f}, "
            f"MDC95={self.mdc95:.4f}, "
            f"MDC99={self.mdc99:.4f}, "
            f"MDC%={self.mdc_pct:.2f}%)"
        )


def compute_mdc(
    measurements: np.ndarray,
    icc: float,
    grand_mean: Optional[float] = None,
) -> MDCResult:
    """Compute MDC statistics from measurement data and an ICC value.

    MDC formula:
        SEM  = SD_total * sqrt(1 - ICC)
        MDC_z = z * sqrt(2) * SEM

    The SRD (Smallest Real Difference) equals MDC95 by convention.

    Args:
        measurements: 1-D array of all measurement values (both sessions).
        icc: ICC point estimate (0–1).  If icc > 1, clipped to 1; if < 0, clipped to 0.
        grand_mean: Optional override for the grand mean (useful when computing
            MDC% relative to a reference session mean rather than grand mean).

    Returns:
        MDCResult with SEM, MDC at three confidence levels, MDC%, and SRD.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    arr = np.asarray(measurements, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        raise ValueError("≥ 2 finite measurements to compute MDC.")

    icc_clipped = float(np.clip(icc, 0.0, 1.0))
    sd_total = float(np.std(arr, ddof=1))
    sem = sd_total * math.sqrt(1.0 - icc_clipped)

    mdc90 = 1.645 * math.sqrt(2.0) * sem
    mdc95 = 1.960 * math.sqrt(2.0) * sem
    mdc99 = 2.576 * math.sqrt(2.0) * sem

    gm = float(np.mean(arr)) if grand_mean is None else float(grand_mean)
    mdc_pct = (mdc95 / gm * 100.0) if gm != 0.0 else float("nan")

    return MDCResult(
        sem=sem,
        mdc90=mdc90,
        mdc95=mdc95,
        mdc99=mdc99,
        mdc_pct=mdc_pct,
        srd=mdc95,
        grand_mean=gm,
        icc_used=icc_clipped,
        sd_total=sd_total,
    )


def mdc_from_sem(
    sem: float,
    grand_mean: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute MDC and MDC% directly from a precomputed SEM.

    Args:
        sem: Standard Error of Measurement.
        grand_mean: Mean value for MDC% normalisation.
        confidence: Confidence level (0.90, 0.95, or 0.99).

    Returns:
        Dict with keys 'mdc' and 'mdc_pct'.
    """
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(round(confidence, 2), 1.960)
    mdc = z * math.sqrt(2.0) * sem
    mdc_pct = (mdc / grand_mean * 100.0) if grand_mean != 0.0 else float("nan")
    return {"mdc": mdc, "mdc_pct": mdc_pct}
