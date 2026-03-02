"""
src/stability/icc.py
====================
Full Intraclass Correlation Coefficient (ICC) computation module.

Supports all six ICC forms as defined by Shrout & Fleiss (1979) and
McGraw & Wong (1996):

    ICC(1,1)  -- One-way random, single measures
    ICC(2,1)  -- Two-way random, single measures
    ICC(3,1)  -- Two-way mixed, single measures
    ICC(1,k)  -- One-way random, average measures (k raters)
    ICC(2,k)  -- Two-way random, average measures
    ICC(3,k)  -- Two-way mixed, average measures

Primary implementation uses rpy2 to call R's ``irr::icc()`` for maximum
numerical accuracy and full reporting. A pure-Python fallback via ``pingouin``
is provided for environments without R.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ICC_TYPE_ALIASES: dict[str, tuple[str, str]] = {
    "ICC11": ("oneway", "single"),
    "ICC21": ("twoway", "single"),
    "ICC31": ("twoway", "single"),
    "ICC1k": ("oneway", "average"),
    "ICC2k": ("twoway", "average"),
    "ICC3k": ("twoway", "average"),
    "ICC(1,1)": ("oneway", "single"),
    "ICC(2,1)": ("twoway", "single"),
    "ICC(3,1)": ("twoway", "single"),
    "ICC(1,k)": ("oneway", "average"),
    "ICC(2,k)": ("twoway", "average"),
    "ICC(3,k)": ("twoway", "average"),
}

_PINGOUIN_ICC_MAP: dict[str, str] = {
    "ICC11": "ICC1", "ICC21": "ICC2", "ICC31": "ICC3",
    "ICC1k": "ICC1k", "ICC2k": "ICC2k", "ICC3k": "ICC3k",
    "ICC(1,1)": "ICC1", "ICC(2,1)": "ICC2", "ICC(3,1)": "ICC3",
    "ICC(1,k)": "ICC1k", "ICC(2,k)": "ICC2k", "ICC(3,k)": "ICC3k",
}


@dataclass
class ICCResult:
    """Container for ICC computation results."""
    icc_value: float
    icc_type: str
    ci_lower: float
    ci_upper: float
    confidence: float = 0.95
    f_value: float = float("nan")
    df1: float = float("nan")
    df2: float = float("nan")
    p_value: float = float("nan")
    n_subjects: int = 0
    n_raters: int = 0
    sem: float = float("nan")
    mdc95: float = float("nan")
    mdc_pct: float = float("nan")
    method: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ICCResult(icc_type={self.icc_type!r}, "
            f"icc={self.icc_value:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f}, "
            f"SEM={self.sem:.4f}, "
            f"MDC95={self.mdc95:.4f}, "
            f"method={self.method!r})"
        )

    def reliability_label(self) -> str:
        """Return a textual reliability label per Koo & Mae (2016)."""
        v = self.icc_value
        if v < 0.50:
            return "poor"
        elif v < 0.75:
            return "moderate"
        elif v < 0.90:
            return "good"
        else:
            return "excellent"


def compute_icc(
    data: pd.DataFrame,
    subjects: str,
    raters: str,
    measurements: str,
    icc_type: str = "ICC2k",
    confidence: float = 0.95,
    use_r: bool = True,
) -> ICCResult:
    """Compute an Intraclass Correlation Coefficient."""
    icc_type_norm = _normalise_icc_type(icc_type)

    for col in (subjects, raters, measurements):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(data[measurements]):
        raise TypeError(f"Measurements column '{measurements}' must be numeric.")

    n_subj = data[subjects].nunique()
    n_raters_count = data[raters].nunique()

    if n_subj < 2:
        raise ValueError(f"Need >= 2 subjects; got {n_subj}.")
    if n_raters_count < 2:
        raise ValueError(f"Need >= 2 raters/occasions; got {n_raters_count}.")

    clean = data[[subjects, raters, measurements]].dropna(subset=[measurements]).copy()
    n_dropped = len(data) - len(clean)
    warn_list: list[str] = []
    if n_dropped > 0:
        msg = f"Dropped {n_dropped} rows with missing measurements before ICC computation."
        logger.warning(msg)
        warn_list.append(msg)

    if use_r:
        try:
            result = _compute_icc_r(clean, subjects, raters, measurements, icc_type_norm, confidence)
            result.warnings.extend(warn_list)
            return result
        except Exception as exc:
            msg = f"R/irr backend failed ({exc}); falling back to pingouin."
            logger.warning(msg)
            warn_list.append(msg)

    result = _compute_icc_pingouin(clean, subjects, raters, measurements, icc_type_norm, confidence)
    result.warnings.extend(warn_list)
    return result


def _compute_icc_r(
    data: pd.DataFrame,
    subjects: str,
    raters: str,
    measurements: str,
    icc_type: str,
    confidence: float,
) -> ICCResult:
    """Compute ICC using R's ``irr::icc()`` via rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError as exc:
        raise ImportError("rpy2 is required for the R backend. Install it with: pip install rpy2") from exc

    irr = importr("irr")
    wide = data.pivot_table(index=subjects, columns=raters, values=measurements, aggfunc="mean")

    with ro.default_converter + pandas2ri.converter:
        r_matrix = ro.conversion.get_conversion().py2rpy(wide)

    model, unit = _ICC_TYPE_ALIASES[icc_type]
    r_type = "consistency" if icc_type in ("ICC31", "ICC3k", "ICC(3,1)", "ICC(3,k)") else "agreement"

    r_result = irr.icc(r_matrix, model=model, type=r_type, unit=unit, conf_level=confidence)

    icc_val = float(r_result.rx2("value")[0])
    ci_lo = float(r_result.rx2("lbound")[0])
    ci_hi = float(r_result.rx2("ubound")[0])
    f_val = float(r_result.rx2("Fvalue")[0])
    df1 = float(r_result.rx2("df1")[0])
    df2 = float(r_result.rx2("df2")[0])
    p_val = float(r_result.rx2("p.value")[0])
    n_subj = int(r_result.rx2("subjects")[0])
    n_rat = int(r_result.rx2("raters")[0])

    sem, mdc95, mdc_pct = _compute_sem_mdc(data[measurements].values, icc_val)

    return ICCResult(
        icc_value=icc_val, icc_type=icc_type, ci_lower=ci_lo, ci_upper=ci_hi,
        confidence=confidence, f_value=f_val, df1=df1, df2=df2, p_value=p_val,
        n_subjects=n_subj, n_raters=n_rat, sem=sem, mdc95=mdc95, mdc_pct=mdc_pct,
        method="r_irr",
    )


def _compute_icc_pingouin(
    data: pd.DataFrame,
    subjects: str,
    raters: str,
    measurements: str,
    icc_type: str,
    confidence: float,
) -> ICCResult:
    """Compute ICC using pingouin (pure Python)."""
    try:
        import pingouin as pg
    except ImportError as exc:
        raise ImportError("pingouin is required for the Python backend: pip install pingouin") from exc

    pg_type = _PINGOUIN_ICC_MAP[icc_type]
    icc_df = pg.intraclass_corr(data=data, targets=subjects, raters=raters, ratings=measurements, nan_policy="omit")

    row = icc_df[icc_df["Type"] == pg_type]
    if row.empty:
        row = icc_df[icc_df["Type"].str.upper() == pg_type.upper()]
    if row.empty:
        raise RuntimeError(f"pingouin did not return ICC type '{pg_type}'. Available: {icc_df['Type'].tolist()}")
    row = row.iloc[0]

    icc_val = float(row["ICC"])
    ci_col = "CI95" if "CI95" in row.index else "CI95%"
    ci_arr = row[ci_col]
    ci_lo = float(ci_arr[0])
    ci_hi = float(ci_arr[1])
    f_val = float(row.get("F", float("nan")))
    df1_val = float(row.get("df1", float("nan")))
    df2_val = float(row.get("df2", float("nan")))
    p_val = float(row.get("pval", float("nan")))

    n_subj = data[subjects].nunique()
    n_rat = data[raters].nunique()
    sem, mdc95, mdc_pct = _compute_sem_mdc(data[measurements].values, icc_val)

    return ICCResult(
        icc_value=icc_val, icc_type=icc_type, ci_lower=ci_lo, ci_upper=ci_hi,
        confidence=confidence, f_value=f_val, df1=df1_val, df2=df2_val, p_value=p_val,
        n_subjects=n_subj, n_raters=n_rat, sem=sem, mdc95=mdc95, mdc_pct=mdc_pct,
        method="pingouin_python",
    )


def _compute_icc_numpy(
    wide: np.ndarray,
    icc_type: str,
    confidence: float,
) -> tuple[float, float, float, float, float, float]:
    """Compute ICC and CI from a wide (subjects x raters) matrix using ANOVA.

    Follows Shrout & Fleiss (1979) Table 1 formulas exactly.
    """
    n, k = wide.shape
    grand_mean = np.nanmean(wide)
    row_means = np.nanmean(wide, axis=1)
    col_means = np.nanmean(wide, axis=0)

    ss_total = np.nansum((wide - grand_mean) ** 2)
    ss_r = k * np.nansum((row_means - grand_mean) ** 2)
    ss_c = n * np.nansum((col_means - grand_mean) ** 2)
    ss_e = ss_total - ss_r - ss_c

    df_r = n - 1
    df_c = k - 1
    df_e = (n - 1) * (k - 1)

    ms_r = ss_r / df_r
    ms_c = ss_c / df_c
    ms_e = ss_e / df_e

    if icc_type in ("ICC11", "ICC(1,1)"):
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
        F = ms_r / ms_e
        df1, df2 = df_r, df_e
    elif icc_type in ("ICC21", "ICC(2,1)"):
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
        F = ms_r / ms_e
        df1, df2 = df_r, df_e
    elif icc_type in ("ICC31", "ICC(3,1)"):
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
        F = ms_r / ms_e
        df1, df2 = df_r, df_e
    elif icc_type in ("ICC1k", "ICC(1,k)"):
        icc = (ms_r - ms_e) / ms_r
        F = ms_r / ms_e
        df1, df2 = df_r, df_e
    elif icc_type in ("ICC2k", "ICC(2,k)"):
        icc = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
        F = ms_r / ms_e
        df1, df2 = df_r, df_e
    elif icc_type in ("ICC3k", "ICC(3,k)"):
        icc = (ms_r - ms_e) / ms_r
        F = ms_r / ms_e
        df1, df2 = df_r, df_e
    else:
        raise ValueError(f"Unknown icc_type: {icc_type}")

    from scipy.stats import f as f_dist
    alpha = 1.0 - confidence
    F_lo = F / f_dist.ppf(1 - alpha / 2, df1, df2)
    F_hi = F / f_dist.ppf(alpha / 2, df1, df2)

    if icc_type in ("ICC11", "ICC(1,1)", "ICC31", "ICC(3,1)"):
        ci_lo = (F_lo - 1) / (F_lo + k - 1)
        ci_hi = (F_hi - 1) / (F_hi + k - 1)
    elif icc_type in ("ICC1k", "ICC(1,k)", "ICC3k", "ICC(3,k)"):
        ci_lo = 1 - 1 / F_lo
        ci_hi = 1 - 1 / F_hi
    else:
        ci_lo = (F_lo - 1) / (F_lo + k - 1)
        ci_hi = (F_hi - 1) / (F_hi + k - 1)

    return float(np.clip(icc, -1, 1)), float(ci_lo), float(ci_hi), float(F), float(df1), float(df2)


def _compute_sem_mdc(values: np.ndarray, icc: float) -> tuple[float, float, float]:
    """Compute Standard Error of Measurement and Minimal Detectable Change.

    SEM = SD_total * sqrt(1 - ICC)
    MDC95 = 1.96 * sqrt(2) * SEM
    MDC% = (MDC95 / grand_mean) * 100
    """
    sd_total = float(np.nanstd(values, ddof=1))
    icc_clipped = max(min(icc, 1.0), 0.0)
    sem = sd_total * math.sqrt(1.0 - icc_clipped)
    mdc95 = 1.96 * math.sqrt(2.0) * sem
    grand_mean = float(np.nanmean(values))
    mdc_pct = (mdc95 / grand_mean * 100.0) if grand_mean != 0 else float("nan")
    return sem, mdc95, mdc_pct


def _normalise_icc_type(icc_type: str) -> str:
    """Normalise ICC type string to a canonical form."""
    cleaned = icc_type.replace(" ", "").upper()
    lookup = {k.upper(): k for k in _ICC_TYPE_ALIASES}
    if cleaned not in lookup:
        raise ValueError(f"Unknown ICC type: '{icc_type}'. Valid options: {list(_ICC_TYPE_ALIASES.keys())}")
    return lookup[cleaned]


def compute_all_icc_forms(
    data: pd.DataFrame,
    subjects: str,
    raters: str,
    measurements: str,
    confidence: float = 0.95,
    use_r: bool = True,
) -> dict[str, ICCResult]:
    """Compute all 6 ICC forms and return as a dictionary."""
    forms = ["ICC11", "ICC21", "ICC31", "ICC1k", "ICC2k", "ICC3k"]
    results: dict[str, ICCResult] = {}
    for form in forms:
        try:
            results[form] = compute_icc(
                data=data, subjects=subjects, raters=raters,
                measurements=measurements, icc_type=form,
                confidence=confidence, use_r=use_r,
            )
        except Exception as exc:
            logger.warning("Failed to compute %s: %s", form, exc)
    return results


def icc_summary_table(results: dict[str, ICCResult]) -> pd.DataFrame:
    """Convert a dict of ICCResult to a summary DataFrame."""
    rows = []
    for icc_type, r in results.items():
        rows.append({
            "Type": icc_type,
            "ICC": round(r.icc_value, 4),
            "CI_Lower": round(r.ci_lower, 4),
            "CI_Upper": round(r.ci_upper, 4),
            "F": round(r.f_value, 4),
            "p_value": round(r.p_value, 4),
            "n_subjects": r.n_subjects,
            "n_raters": r.n_raters,
            "SEM": round(r.sem, 4),
            "MDC95": round(r.mdc95, 4),
            "MDC_pct": round(r.mdc_pct, 2),
            "Reliability": r.reliability_label(),
            "Method": r.method,
        })
    return pd.DataFrame(rows)
