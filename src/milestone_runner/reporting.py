"""
src/milestone_runner/reporting.py
===================================
Milestone report generation - JSON and Markdown outputs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .gating import GateResult, GateStatus

logger = logging.getLogger(__name__)


class MilestoneReport:
    """Wraps a list of GateResults and generates report artefacts."""

    def __init__(self, config_phase, study_title, results, generated_by="gnf-core-foundry"):
        self.config_phase = config_phase
        self.study_title = study_title
        self.results = results
        self.generated_by = generated_by
        self.timestamp = datetime.now(tz=timezone.utc).isoformat()

    def overall_status(self) -> GateStatus:
        statuses = [r.status for r in self.results]
        if all(s == GateStatus.PASS for s in statuses):
            return GateStatus.PASS
        if any(s == GateStatus.FAIL for s in statuses):
            return GateStatus.FAIL
        return GateStatus.WARNING

    def summary(self) -> str:
        n_pass = sum(1 for r in self.results if r.status == GateStatus.PASS)
        n_fail = sum(1 for r in self.results if r.status == GateStatus.FAIL)
        return (
            f"[{self.config_phase}] {self.study_title} — "
            f"Overall: {self.overall_status().value} "
            f"({n_pass} pass / {n_fail} fail out of {len(self.results)} milestones)"
        )

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "report_metadata": {
                "phase": self.config_phase,
                "study_title": self.study_title,
                "generated_by": self.generated_by,
                "generated_at": self.timestamp,
                "overall_status": self.overall_status().value,
            },
            "milestones": [r.to_dict() for r in self.results],
        }
        return json.dumps(payload, indent=indent, default=str)

    def save_json(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        logger.info("Milestone JSON report saved to: %s", p)
        return p

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Milestone Report — {self.config_phase}: {self.study_title}")
        lines.append("")
        lines.append(f"**Generated:** {self.timestamp}  ")
        lines.append(f"**Overall Status:** {self.overall_status().value}  ")
        lines.append(f"**Pipeline:** {self.generated_by}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Milestone | Status | Score | Pass | Fail | Skip |")
        lines.append("|-----------|--------|-------|------|------|------|")
        for r in self.results:
            status_badge = _status_badge(r.status)
            lines.append(
                f"| {r.milestone_id} | {status_badge} | "
                f"{r.score_pct:.1f}% | {r.n_pass} | {r.n_fail} | {r.n_skip} |"
            )
        lines.append("")
        for r in self.results:
            lines.append("---")
            lines.append("")
            lines.append(f"## {r.milestone_id} — {_status_badge(r.status)}")
            lines.append("")
            lines.append(f"**Logic:** {r.logic.value}  ")
            lines.append(f"**Score:** {r.score_pct:.1f}%  ")
            lines.append("")
            lines.append("### Criteria")
            lines.append("")
            lines.append("| ID | Metric | Label | Observed | Threshold | Status | Critical |")
            lines.append("|----|--------|-------|----------|-----------|--------|----------|")
            for cr in r.criterion_results:
                obs_fmt = f"{cr.observed_value:.4g}" if isinstance(cr.observed_value, float) else str(cr.observed_value)
                thr_fmt = f"{cr.threshold:.4g}" if isinstance(cr.threshold, float) else str(cr.threshold)
                crit_mark = "✓" if cr.critical else ""
                lines.append(
                    f"| {cr.criterion_id} | `{cr.metric}` | {cr.label} | "
                    f"{obs_fmt} | {cr.operator} {thr_fmt} | "
                    f"{_status_badge(cr.status)} | {crit_mark} |"
                )
            lines.append("")
            if r.recommendations:
                lines.append("### Recommendations")
                lines.append("")
                for rec in r.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        return "\n".join(lines)

    def save_markdown(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Milestone Markdown report saved to: %s", p)
        return p


def _status_badge(status: GateStatus) -> str:
    badges = {
        GateStatus.PASS: "**PASS** ✅",
        GateStatus.FAIL: "**FAIL** ❌",
        GateStatus.SKIP: "**SKIP** ⏭",
        GateStatus.WARNING: "**WARNING** ⚠️",
    }
    return badges.get(status, status.value)
