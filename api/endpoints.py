"""
api/endpoints.py
FastAPI application for the gnf-core-foundry pipeline engine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GNF Core Foundry API",
    description="Digital biomarker pipeline API for Global NeuroFoundry.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class ICCRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of records with subject_id, rater_id, and value columns.")
    subject_col: str = Field("subject_id")
    rater_col: str = Field("rater_id")
    value_col: str = Field("value")
    icc_type: str = Field("ICC2k")
    confidence: float = Field(0.95, ge=0.80, le=0.99)


class ICCResponse(BaseModel):
    icc_type: str
    icc_value: float
    ci_lower: float
    ci_upper: float
    confidence: float
    f_value: float
    p_value: float
    n_subjects: int
    n_raters: int
    sem: float
    mdc95: float
    mdc_pct: float
    reliability_label: str
    method: str
    warnings: list[str]


class BlandAltmanRequest(BaseModel):
    method1: list[float] = Field(...)
    method2: list[float] = Field(...)
    confidence: float = Field(0.95, ge=0.80, le=0.99)
    title: str = Field("Bland-Altman Analysis")


class BlandAltmanResponse(BaseModel):
    n: int
    bias: float
    bias_ci_lower: float
    bias_ci_upper: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    loa_lower_ci: list[float]
    loa_upper_ci: list[float]
    proportional_bias_slope: float
    proportional_bias_p: float
    proportional_bias_present: bool
    percent_within_loa: float
    bias_pct: float
    confidence: float
    warnings: list[str]


class MilestoneRequest(BaseModel):
    config_name: str = Field("r21_template")
    metrics: dict[str, Any] = Field(...)
    milestone_id: Optional[str] = Field(None)


class MilestoneResponse(BaseModel):
    phase: str
    study_title: str
    overall_status: str
    milestones: list[dict]
    summary: str


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    return {"status": "ok", "service": "gnf-core-foundry"}


@app.post("/compute-icc", response_model=ICCResponse, tags=["Reliability"])
def compute_icc_endpoint(request: ICCRequest) -> ICCResponse:
    """Compute an Intraclass Correlation Coefficient."""
    try:
        from stability.icc import compute_icc
    except ImportError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"gnf-core-foundry not installed: {exc}")

    if len(request.records) < 4:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Need at least 4 records.")

    df = pd.DataFrame(request.records)
    try:
        result = compute_icc(
            data=df, subjects=request.subject_col, raters=request.rater_col,
            measurements=request.value_col, icc_type=request.icc_type,
            confidence=request.confidence, use_r=False,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    return ICCResponse(
        icc_type=result.icc_type, icc_value=result.icc_value,
        ci_lower=result.ci_lower, ci_upper=result.ci_upper,
        confidence=result.confidence,
        f_value=result.f_value if not _is_nan(result.f_value) else -1.0,
        p_value=result.p_value if not _is_nan(result.p_value) else -1.0,
        n_subjects=result.n_subjects, n_raters=result.n_raters,
        sem=result.sem if not _is_nan(result.sem) else -1.0,
        mdc95=result.mdc95 if not _is_nan(result.mdc95) else -1.0,
        mdc_pct=result.mdc_pct if not _is_nan(result.mdc_pct) else -1.0,
        reliability_label=result.reliability_label(),
        method=result.method, warnings=result.warnings,
    )


@app.post("/bland-altman", response_model=BlandAltmanResponse, tags=["Reliability"])
def bland_altman_endpoint(request: BlandAltmanRequest) -> BlandAltmanResponse:
    """Run Bland-Altman method comparison analysis."""
    try:
        from stability.bland_altman import bland_altman_analysis
    except ImportError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"gnf-core-foundry not installed: {exc}")

    if len(request.method1) != len(request.method2):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="method1 and method2 must have the same length.")
    if len(request.method1) < 3:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Need >= 3 paired observations.")

    try:
        result = bland_altman_analysis(np.array(request.method1, dtype=float), np.array(request.method2, dtype=float), confidence=request.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    return BlandAltmanResponse(
        n=result.n, bias=result.bias, bias_ci_lower=result.bias_ci_lower, bias_ci_upper=result.bias_ci_upper,
        sd_diff=result.sd_diff, loa_lower=result.loa_lower, loa_upper=result.loa_upper,
        loa_lower_ci=[result.loa_lower_ci_lower, result.loa_lower_ci_upper],
        loa_upper_ci=[result.loa_upper_ci_lower, result.loa_upper_ci_upper],
        proportional_bias_slope=result.proportional_bias_slope,
        proportional_bias_p=result.proportional_bias_p,
        proportional_bias_present=result.proportional_bias_present,
        percent_within_loa=result.percent_within_loa,
        bias_pct=result.bias_pct if not _is_nan(result.bias_pct) else -1.0,
        confidence=result.confidence, warnings=result.warnings,
    )


@app.post("/run-milestone", response_model=MilestoneResponse, tags=["Milestone"])
def run_milestone_endpoint(request: MilestoneRequest) -> MilestoneResponse:
    """Evaluate milestone gating criteria."""
    try:
        from milestone_runner.runner import MilestoneRunner
    except ImportError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"gnf-core-foundry not installed: {exc}")

    import pathlib
    config_path = pathlib.Path(__file__).parent.parent / "configs" / f"{request.config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config '{request.config_name}' not found.")

    try:
        runner = MilestoneRunner(config_path)
        if request.milestone_id:
            gate = runner.run_single_milestone(request.milestone_id, request.metrics)
            from milestone_runner.reporting import MilestoneReport
            report = MilestoneReport(runner.phase, runner.study_title, [gate])
        else:
            report = runner.run(request.metrics)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    return MilestoneResponse(
        phase=report.config_phase, study_title=report.study_title,
        overall_status=report.overall_status().value,
        milestones=[r.to_dict() for r in report.results],
        summary=report.summary(),
    )


def _is_nan(v: float) -> bool:
    import math
    return not math.isfinite(v)
