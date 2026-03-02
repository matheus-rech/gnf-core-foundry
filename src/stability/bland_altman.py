"""
src/stability/bland_altman.py
==============================
Full Bland-Altman analysis module for method comparison and test-retest studies.

References:
    Bland JM, Altman DG (1986). Statistical methods for assessing agreement
    between two methods of clinical measurement. Lancet, 1(8476), 307-310.

    Bland JM, Altman DG (1999). Measuring agreement in method comparison
    studies. Stat Methods Med Res, 8(2), 135-160.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BlandAltmanResult:
    """Container for Bland-Altman analysis results."""
    n: int
    means: np.ndarray
    diffs: np.ndarray
    bias: float
    bias_ci_lower: float
    bias_ci_upper: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    loa_lower_ci_lower: float
    loa_lower_ci_upper: float
    loa_upper_ci_lower: float
    loa_upper_ci_upper: float
    confidence: float = 0.95
    proportional_bias_slope: float = float("nan")
    proportional_bias_intercept: float = float("nan")
    proportional_bias_r: float = float("nan")
    proportional_bias_p: float = float("nan")
    proportional_bias_present: bool = False
    percent_within_loa: float = float("nan")
    bias_pct: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BlandAltmanResult("
            f"n={self.n}, "
            f"bias={self.bias:.4f} [{self.bias_ci_lower:.4f}, {self.bias_ci_upper:.4f}], "
            f"LoA=[{self.loa_lower:.4f}, {self.loa_upper:.4f}], "
            f"prop_bias_p={self.proportional_bias_p:.4f}, "
            f"bias_pct={self.bias_pct:.2f}%)"
        )

    def summary_dict(self) -> dict:
        return {
            "n": self.n,
            "bias": round(self.bias, 6),
            "bias_ci_lower": round(self.bias_ci_lower, 6),
            "bias_ci_upper": round(self.bias_ci_upper, 6),
            "sd_diff": round(self.sd_diff, 6),
            "loa_lower": round(self.loa_lower, 6),
            "loa_upper": round(self.loa_upper, 6),
            "loa_lower_ci": [round(self.loa_lower_ci_lower, 6), round(self.loa_lower_ci_upper, 6)],
            "loa_upper_ci": [round(self.loa_upper_ci_lower, 6), round(self.loa_upper_ci_upper, 6)],
            "confidence": self.confidence,
            "proportional_bias_slope": round(self.proportional_bias_slope, 6),
            "proportional_bias_p": round(self.proportional_bias_p, 6),
            "proportional_bias_present": self.proportional_bias_present,
            "percent_within_loa": round(self.percent_within_loa, 2),
            "bias_pct": round(self.bias_pct, 4),
        }


def bland_altman_analysis(
    method1: np.ndarray,
    method2: np.ndarray,
    confidence: float = 0.95,
    subjects: Optional[np.ndarray] = None,
) -> BlandAltmanResult:
    """Run a complete Bland-Altman analysis."""
    m1 = np.asarray(method1, dtype=float)
    m2 = np.asarray(method2, dtype=float)

    if m1.shape != m2.shape:
        raise ValueError(f"method1 and method2 must have the same shape; got {m1.shape} vs {m2.shape}.")
    if m1.ndim != 1:
        raise ValueError("method1 and method2 must be 1-D arrays.")

    warn_list: list[str] = []

    valid_mask = np.isfinite(m1) & np.isfinite(m2)
    n_nan = int(np.sum(~valid_mask))
    if n_nan > 0:
        msg = f"Removed {n_nan} pairs with NaN/Inf values."
        logger.warning(msg)
        warn_list.append(msg)
        m1, m2 = m1[valid_mask], m2[valid_mask]
        if subjects is not None:
            subjects = np.asarray(subjects)[valid_mask]

    n = len(m1)
    if n < 3:
        raise ValueError(f"Need >= 3 valid pairs; got {n}.")

    if subjects is not None:
        m1, m2 = _repeated_measures_collapse(m1, m2, np.asarray(subjects))
        n = len(m1)

    means = (m1 + m2) / 2.0
    diffs = m1 - m2

    bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    se_bias = sd_diff / math.sqrt(n)
    bias_ci_lo = bias - t_crit * se_bias
    bias_ci_hi = bias + t_crit * se_bias

    loa_lo = bias - 1.96 * sd_diff
    loa_hi = bias + 1.96 * sd_diff

    se_loa = math.sqrt(3.0 * sd_diff**2 / n)
    loa_lo_ci_lo = loa_lo - t_crit * se_loa
    loa_lo_ci_hi = loa_lo + t_crit * se_loa
    loa_hi_ci_lo = loa_hi - t_crit * se_loa
    loa_hi_ci_hi = loa_hi + t_crit * se_loa

    slope, intercept, r_val, p_val, se_slope = stats.linregress(means, diffs)
    prop_bias_present = bool(p_val < 0.05)

    if prop_bias_present:
        msg = f"Proportional bias detected: slope={slope:.4f}, p={p_val:.4f}."
        logger.info(msg)
        warn_list.append(msg)

    within = np.sum((diffs >= loa_lo) & (diffs <= loa_hi))
    pct_within = float(within / n * 100.0)

    grand_mean = float(np.mean(means))
    bias_pct = (abs(bias) / grand_mean * 100.0) if grand_mean != 0 else float("nan")

    return BlandAltmanResult(
        n=n, means=means, diffs=diffs,
        bias=bias, bias_ci_lower=float(bias_ci_lo), bias_ci_upper=float(bias_ci_hi),
        sd_diff=sd_diff, loa_lower=float(loa_lo), loa_upper=float(loa_hi),
        loa_lower_ci_lower=float(loa_lo_ci_lo), loa_lower_ci_upper=float(loa_lo_ci_hi),
        loa_upper_ci_lower=float(loa_hi_ci_lo), loa_upper_ci_upper=float(loa_hi_ci_hi),
        confidence=confidence,
        proportional_bias_slope=float(slope), proportional_bias_intercept=float(intercept),
        proportional_bias_r=float(r_val), proportional_bias_p=float(p_val),
        proportional_bias_present=prop_bias_present,
        percent_within_loa=pct_within, bias_pct=bias_pct, warnings=warn_list,
    )


def plot_bland_altman(
    result: BlandAltmanResult,
    title: str = "Bland-Altman Plot",
    xlabel: str = "Mean of two methods",
    ylabel: str = "Difference (Method 1 - Method 2)",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9.0, 6.0),
    show_ci_bands: bool = True,
    show_regression: bool = True,
    color_points: str = "#2166AC",
    color_bias: str = "#D6604D",
    color_loa: str = "#4DAC26",
    dpi: int = 150,
) -> "matplotlib.figure.Figure":
    """Generate a publication-quality Bland-Altman plot."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(result.means, result.diffs, color=color_points, alpha=0.7, s=50, zorder=5, label="Observations")

    x_min, x_max = float(np.min(result.means)), float(np.max(result.means))
    x_pad = (x_max - x_min) * 0.05
    x_range = np.linspace(x_min - x_pad, x_max + x_pad, 300)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.axhline(result.bias, color=color_bias, linewidth=2.0, linestyle="-", zorder=6, label=f"Bias = {result.bias:.4f}")
    if show_ci_bands:
        ax.fill_between([x_min - x_pad, x_max + x_pad], result.bias_ci_lower, result.bias_ci_upper,
                        color=color_bias, alpha=0.15, zorder=3)

    for loa_val, loa_lo, loa_hi, lbl in [
        (result.loa_upper, result.loa_upper_ci_lower, result.loa_upper_ci_upper, f"Upper LoA = {result.loa_upper:.4f}"),
        (result.loa_lower, result.loa_lower_ci_lower, result.loa_lower_ci_upper, f"Lower LoA = {result.loa_lower:.4f}"),
    ]:
        ax.axhline(loa_val, color=color_loa, linewidth=2.0, linestyle="--", zorder=6, label=lbl)
        if show_ci_bands:
            ax.fill_between([x_min - x_pad, x_max + x_pad], loa_lo, loa_hi, color=color_loa, alpha=0.10, zorder=3)

    if show_regression and result.proportional_bias_present:
        y_reg = result.proportional_bias_intercept + result.proportional_bias_slope * x_range
        ax.plot(x_range, y_reg, color="darkorange", linewidth=1.5, linestyle=":", zorder=7,
                label=f"Prop. bias (slope={result.proportional_bias_slope:.3f}, p={result.proportional_bias_p:.3f})")

    ax.annotate(f"Bias = {result.bias:.3f}\n[{result.bias_ci_lower:.3f}, {result.bias_ci_upper:.3f}]",
                xy=(x_max, result.bias), xytext=(x_max + x_pad * 0.5, result.bias),
                fontsize=8, color=color_bias, va="center", ha="left")
    ax.annotate(f"+1.96 SD = {result.loa_upper:.3f}", xy=(x_max, result.loa_upper),
                xytext=(x_max + x_pad * 0.5, result.loa_upper), fontsize=8, color=color_loa, va="center", ha="left")
    ax.annotate(f"-1.96 SD = {result.loa_lower:.3f}", xy=(x_max, result.loa_lower),
                xytext=(x_max + x_pad * 0.5, result.loa_lower), fontsize=8, color=color_loa, va="center", ha="left")

    ax.set_xlim(x_min - x_pad, x_max + x_pad * 4)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    stats_text = (
        f"n = {result.n}\n"
        f"Bias = {result.bias:.3f}\n"
        f"SD_diff = {result.sd_diff:.3f}\n"
        f"% within LoA = {result.percent_within_loa:.1f}%\n"
        f"Prop. bias p = {result.proportional_bias_p:.3f}"
    )
    ax.text(0.98, 0.04, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.8})

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Bland-Altman plot saved to: %s", save_path)

    return fig


def _repeated_measures_collapse(m1: np.ndarray, m2: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated measures to subject-level means."""
    import pandas as pd
    df = pd.DataFrame({"subject": subjects, "m1": m1, "m2": m2})
    grp = df.groupby("subject").mean().reset_index()
    return grp["m1"].values, grp["m2"].values


def batch_bland_altman(
    data: "pd.DataFrame",
    biomarker_col: str,
    method_col: str,
    subject_col: str,
    reference_method: str,
    test_method: str,
    confidence: float = 0.95,
) -> dict[str, BlandAltmanResult]:
    """Run Bland-Altman for each unique biomarker in a long-format DataFrame."""
    import pandas as pd
    results: dict[str, BlandAltmanResult] = {}
    for bm_name, grp in data.groupby(biomarker_col):
        ref = grp[grp[method_col] == reference_method].set_index(subject_col)["value"]
        tst = grp[grp[method_col] == test_method].set_index(subject_col)["value"]
        common_subjects = ref.index.intersection(tst.index)
        if len(common_subjects) < 3:
            logger.warning("Skipping %s: fewer than 3 paired observations.", bm_name)
            continue
        try:
            results[str(bm_name)] = bland_altman_analysis(
                ref.loc[common_subjects].values, tst.loc[common_subjects].values, confidence=confidence)
        except Exception as exc:
            logger.warning("Bland-Altman failed for %s: %s", bm_name, exc)
    return results
