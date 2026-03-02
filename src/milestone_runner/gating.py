"""
src/milestone_runner/gating.py
Gate criteria evaluation with full evidence trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARNING = "WARNING"


class GateLogic(str, Enum):
    AND = "AND"
    OR = "OR"


@dataclass
class GatingCriterion:
    criterion_id: str
    metric: str
    operator: str
    threshold: Any = None
    threshold_upper: Any = None
    label: str = ""
    rationale: str = ""
    critical: bool = False
    weight: float = 1.0


@dataclass
class CriterionResult:
    criterion_id: str
    metric: str
    label: str
    status: GateStatus
    observed_value: Any
    threshold: Any
    operator: str
    critical: bool
    evidence: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class GateResult:
    milestone_id: str
    status: GateStatus
    logic: GateLogic
    criterion_results: list[CriterionResult] = field(default_factory=list)
    evidence_trail: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    n_pass: int = 0
    n_fail: int = 0
    n_skip: int = 0
    score_pct: float = 0.0

    def summary(self) -> str:
        return (
            f"Milestone {self.milestone_id}: {self.status.value} "
            f"({self.n_pass} pass / {self.n_fail} fail / {self.n_skip} skip, "
            f"score={self.score_pct:.1f}%)"
        )

    def to_dict(self) -> dict:
        return {
            "milestone_id": self.milestone_id,
            "status": self.status.value,
            "logic": self.logic.value,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_skip": self.n_skip,
            "score_pct": round(self.score_pct, 2),
            "recommendations": self.recommendations,
            "criteria": [
                {
                    "criterion_id": cr.criterion_id,
                    "metric": cr.metric,
                    "label": cr.label,
                    "status": cr.status.value,
                    "observed_value": cr.observed_value,
                    "threshold": cr.threshold,
                    "operator": cr.operator,
                    "critical": cr.critical,
                    "message": cr.message,
                }
                for cr in self.criterion_results
            ],
        }


def evaluate_gate(
    criteria: list[GatingCriterion],
    metrics: dict[str, Any],
    milestone_id: str = "unknown",
    logic: GateLogic = GateLogic.AND,
    no_go_actions: Optional[list[str]] = None,
) -> GateResult:
    """Evaluate a list of gating criteria against a metrics dictionary."""
    criterion_results: list[CriterionResult] = []
    evidence: dict = {}

    for crit in criteria:
        cr = _evaluate_criterion(crit, metrics)
        criterion_results.append(cr)
        evidence[crit.criterion_id] = cr.evidence
        logger.debug("Criterion %s [%s]: %s", crit.criterion_id, crit.metric, cr.status.value)

    n_pass = sum(1 for cr in criterion_results if cr.status == GateStatus.PASS)
    n_fail = sum(1 for cr in criterion_results if cr.status == GateStatus.FAIL)
    n_skip = sum(1 for cr in criterion_results if cr.status == GateStatus.SKIP)

    total_weight = sum(c.weight for c in criteria if c.criterion_id in
                       {cr.criterion_id for cr in criterion_results if cr.status != GateStatus.SKIP})
    pass_weight = sum(c.weight for c, cr in zip(criteria, criterion_results) if cr.status == GateStatus.PASS)
    score_pct = (pass_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    overall_status = _determine_overall_status(criterion_results, logic)

    recommendations: list[str] = []
    if overall_status == GateStatus.FAIL:
        recommendations.extend(no_go_actions or [])
        for cr in criterion_results:
            if cr.status == GateStatus.FAIL:
                recommendations.append(f"[{cr.criterion_id}] {cr.label}: {cr.message}")

    result = GateResult(
        milestone_id=milestone_id,
        status=overall_status,
        logic=logic,
        criterion_results=criterion_results,
        evidence_trail=evidence,
        recommendations=recommendations,
        n_pass=n_pass,
        n_fail=n_fail,
        n_skip=n_skip,
        score_pct=score_pct,
    )
    logger.info(result.summary())
    return result


def _evaluate_criterion(crit: GatingCriterion, metrics: dict[str, Any]) -> CriterionResult:
    """Evaluate a single criterion against the metrics dict."""
    if crit.metric not in metrics:
        return CriterionResult(
            criterion_id=crit.criterion_id, metric=crit.metric, label=crit.label,
            status=GateStatus.SKIP, observed_value=None, threshold=crit.threshold,
            operator=crit.operator, critical=crit.critical,
            evidence={"status": "metric_not_found"},
            message=f"Metric '{crit.metric}' not present in metrics dict.",
        )

    observed = metrics[crit.metric]

    if crit.operator == "not_null":
        passed = observed is not None
        return CriterionResult(
            criterion_id=crit.criterion_id, metric=crit.metric, label=crit.label,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            observed_value=observed, threshold=None, operator=crit.operator, critical=crit.critical,
            evidence={"observed": observed, "operator": "not_null"},
            message="not null" if passed else "value is null",
        )

    if isinstance(crit.threshold, bool):
        passed = bool(observed) == crit.threshold
        return CriterionResult(
            criterion_id=crit.criterion_id, metric=crit.metric, label=crit.label,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            observed_value=observed, threshold=crit.threshold, operator="eq", critical=crit.critical,
            evidence={"observed": observed, "expected": crit.threshold},
            message=f"{observed} == {crit.threshold}" if passed else f"{observed} != {crit.threshold}",
        )

    try:
        obs_float = float(observed)
        thr = crit.threshold
    except (TypeError, ValueError) as exc:
        return CriterionResult(
            criterion_id=crit.criterion_id, metric=crit.metric, label=crit.label,
            status=GateStatus.SKIP, observed_value=observed, threshold=crit.threshold,
            operator=crit.operator, critical=crit.critical,
            evidence={"error": str(exc)}, message=f"Cannot cast observed value to float: {exc}",
        )

    op = crit.operator.lower()
    if op == "gt":
        passed = obs_float > float(thr)
        msg = f"{obs_float:.4g} > {float(thr):.4g}"
    elif op == "gte":
        passed = obs_float >= float(thr)
        msg = f"{obs_float:.4g} >= {float(thr):.4g}"
    elif op == "lt":
        passed = obs_float < float(thr)
        msg = f"{obs_float:.4g} < {float(thr):.4g}"
    elif op == "lte":
        passed = obs_float <= float(thr)
        msg = f"{obs_float:.4g} <= {float(thr):.4g}"
    elif op == "eq":
        passed = obs_float == float(thr)
        msg = f"{obs_float:.4g} == {float(thr):.4g}"
    elif op == "neq":
        passed = obs_float != float(thr)
        msg = f"{obs_float:.4g} != {float(thr):.4g}"
    elif op == "between":
        lo = float(thr) if not isinstance(thr, (list, tuple)) else float(thr[0])
        hi = float(crit.threshold_upper) if crit.threshold_upper is not None else float(thr[1])
        passed = lo <= obs_float <= hi
        msg = f"{lo:.4g} <= {obs_float:.4g} <= {hi:.4g}"
        thr = [lo, hi]
    else:
        return CriterionResult(
            criterion_id=crit.criterion_id, metric=crit.metric, label=crit.label,
            status=GateStatus.SKIP, observed_value=observed, threshold=crit.threshold,
            operator=crit.operator, critical=crit.critical,
            evidence={"error": f"Unknown operator: {op}"}, message=f"Unknown operator '{op}'.",
        )

    status = GateStatus.PASS if passed else GateStatus.FAIL
    full_msg = f"{crit.label}: {msg} -> {'PASS' if passed else 'FAIL'}"

    return CriterionResult(
        criterion_id=crit.criterion_id, metric=crit.metric, label=crit.label,
        status=status, observed_value=observed, threshold=thr, operator=op, critical=crit.critical,
        evidence={"observed": observed, "threshold": thr, "operator": op, "passed": passed},
        message=full_msg,
    )


def _determine_overall_status(criterion_results: list[CriterionResult], logic: GateLogic) -> GateStatus:
    """Determine the overall gate status."""
    for cr in criterion_results:
        if cr.critical and cr.status == GateStatus.FAIL:
            return GateStatus.FAIL

    non_skip = [cr for cr in criterion_results if cr.status != GateStatus.SKIP]
    if not non_skip:
        return GateStatus.SKIP

    if logic == GateLogic.AND:
        all_pass = all(cr.status == GateStatus.PASS for cr in non_skip)
        return GateStatus.PASS if all_pass else GateStatus.FAIL
    else:
        any_pass = any(cr.status == GateStatus.PASS for cr in non_skip)
        return GateStatus.PASS if any_pass else GateStatus.FAIL
