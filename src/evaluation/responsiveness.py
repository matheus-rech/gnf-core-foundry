"""
src/evaluation/responsiveness.py
===================================
Responsiveness and effect size measures for digital biomarkers.

Provides:
- Standardized Response Mean (SRM)
- Cohen's d
- Hedges' g (small-sample corrected)
- Responsiveness Index (RI = SRM / CI_width)
- Glass's delta (when control group SD is the reference)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EffectSizeResult:
    """Container for effect size and responsiveness statistics."""

    cohens_d: float
    hedges_g: float
    glass_delta: float
    srm: float
    mean_change: float
    sd_change: float
    ci_lower: float
    ci_upper: float
    n: int
    p_value: float
    interpretation_d: str

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_effect_size(
    pre: np.ndarray,
    post: np.ndarray,
    confidence: float = 0.95,
    control_sd: Optional[float] = None,
) -> EffectSizeResult:
    """Compute effect sizes and responsiveness from pre/post paired measurements."""
    a = np.asarray(pre, dtype=float)
    b = np.asarray(post, dtype=float)

    if a.shape != b.shape:
        raise ValueError("pre and post must have the same shape.")
    if len(a) < 3:
        raise ValueError("≥ 3 paired observations.")

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)

    change = b - a
    mean_change = float(np.mean(change))
    sd_change = float(np.std(change, ddof=1))

    t_stat, p_val = stats.ttest_rel(a, b)

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    se = sd_change / math.sqrt(n)
    ci_lo = mean_change - t_crit * se
    ci_hi = mean_change + t_crit * se

    cohens_d = mean_change / sd_change if sd_change > 0 else float("nan")
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    hedges_g = cohens_d * j if not math.isnan(cohens_d) else float("nan")

    glass_delta = float("nan")
    if control_sd is not None and control_sd > 0:
        glass_delta = mean_change / control_sd

    srm = cohens_d
    interpretation = _interpret_cohens_d(abs(cohens_d))

    return EffectSizeResult(
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        glass_delta=glass_delta,
        srm=srm,
        mean_change=mean_change,
        sd_change=sd_change,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n,
        p_value=float(p_val),
        interpretation_d=interpretation,
    )


def compute_group_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Compute independent-samples effect size."""
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group needs ≥ 2 observations.")

    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))

    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / sp if sp > 0 else float("nan")
    j = 1.0 - 3.0 / (4.0 * (n1 + n2 - 2) - 1.0)
    hedges_g = cohens_d * j if not math.isnan(cohens_d) else float("nan")

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

    alpha = 1.0 - confidence
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    df = (s1**2 / n1 + s2**2 / n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    t_crit = float(stats.t.ppf(1 - alpha/2, df=df))
    ci_lo = (m1 - m2) - t_crit * se
    ci_hi = (m1 - m2) + t_crit * se

    interpretation = _interpret_cohens_d(abs(cohens_d))

    return EffectSizeResult(
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        glass_delta=float("nan"),
        srm=cohens_d,
        mean_change=m1 - m2,
        sd_change=sp,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n1 + n2,
        p_value=float(p_val),
        interpretation_d=interpretation,
    )


def _interpret_cohens_d(d: float) -> str:
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "very large"
