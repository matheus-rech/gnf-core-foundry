"""
src/milestone_runner/runner.py
YAML-driven milestone gating engine for NIH R21/R33 digital biomarker studies.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from .gating import GateLogic, GateResult, GatingCriterion, GateStatus, evaluate_gate
from .reporting import MilestoneReport

logger = logging.getLogger(__name__)


class MilestoneRunner:
    """Load YAML config and run milestone gating against a metrics dict."""

    def __init__(self, config_path: str | Path, verbose: bool = False) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.config = self._load_config()
        self.phase: str = self.config.get("phase", "unknown")
        self.study_title: str = self.config.get("study_title", "GNF Study")
        self.milestones_config: list[dict] = self.config.get("milestones", [])
        logger.info("MilestoneRunner loaded: %s (%s) -- %d milestone(s)", self.study_title, self.phase, len(self.milestones_config))

    def run(self, metrics: dict[str, Any], output_dir: Optional[str | Path] = None) -> MilestoneReport:
        """Evaluate all milestones against the provided metrics."""
        logger.info("Running %s milestones against %d metrics.", self.phase, len(metrics))
        gate_results: list[GateResult] = []
        for ms_cfg in self.milestones_config:
            gate_result = self._evaluate_milestone(ms_cfg, metrics)
            gate_results.append(gate_result)

        report = MilestoneReport(
            config_phase=self.phase,
            study_title=self.study_title,
            results=gate_results,
            generated_by="gnf-core-foundry/milestone_runner",
        )
        logger.info(report.summary())

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            timestamp = report.timestamp.replace(":", "-").replace(".", "-")[:19]
            report.save_json(out / f"milestone_report_{timestamp}.json")
            report.save_markdown(out / f"milestone_report_{timestamp}.md")

        return report

    def run_single_milestone(self, milestone_id: str, metrics: dict[str, Any]) -> GateResult:
        """Evaluate a single named milestone."""
        for ms_cfg in self.milestones_config:
            if ms_cfg.get("milestone_id") == milestone_id:
                return self._evaluate_milestone(ms_cfg, metrics)
        raise ValueError(
            f"Milestone '{milestone_id}' not found. "
            f"Available: {[m.get('milestone_id') for m in self.milestones_config]}"
        )

    def list_milestones(self) -> list[dict[str, Any]]:
        return [
            {"milestone_id": m.get("milestone_id"), "title": m.get("title"),
             "due_month": m.get("due_month"), "n_criteria": len(m.get("criteria", []))}
            for m in self.milestones_config
        ]

    def list_required_metrics(self) -> list[str]:
        metrics_set: set[str] = set()
        for ms_cfg in self.milestones_config:
            for crit in ms_cfg.get("criteria", []):
                metrics_set.add(crit["metric"])
        return sorted(metrics_set)

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if not isinstance(config, dict):
            raise ValueError(f"Config file must be a YAML mapping: {self.config_path}")
        if "milestones" not in config:
            raise ValueError("Config file must have a 'milestones' key.")
        return config

    def _evaluate_milestone(self, ms_cfg: dict, metrics: dict[str, Any]) -> GateResult:
        milestone_id: str = ms_cfg.get("milestone_id", "unknown")
        logic_str: str = ms_cfg.get("logic", "AND").upper()
        logic = GateLogic.AND if logic_str == "AND" else GateLogic.OR
        no_go_actions: list[str] = ms_cfg.get("no_go_actions", [])

        criteria: list[GatingCriterion] = []
        for crit_cfg in ms_cfg.get("criteria", []):
            criterion = _parse_criterion(crit_cfg)
            criteria.append(criterion)

        if not criteria:
            logger.warning("Milestone %s has no criteria -- marking SKIP.", milestone_id)
            return GateResult(milestone_id=milestone_id, status=GateStatus.SKIP, logic=logic)

        return evaluate_gate(
            criteria=criteria,
            metrics=metrics,
            milestone_id=milestone_id,
            logic=logic,
            no_go_actions=no_go_actions,
        )


def _parse_criterion(crit_cfg: dict) -> GatingCriterion:
    """Parse a criterion config dict into a GatingCriterion."""
    required = ("criterion_id", "metric", "operator", "threshold")
    for key in required:
        if key not in crit_cfg:
            raise ValueError(f"Criterion is missing required key '{key}': {crit_cfg}")

    threshold = crit_cfg["threshold"]
    if isinstance(threshold, str) and threshold.lower() in ("true", "false"):
        threshold = threshold.lower() == "true"

    return GatingCriterion(
        criterion_id=crit_cfg["criterion_id"],
        metric=crit_cfg["metric"],
        operator=crit_cfg["operator"],
        threshold=threshold,
        threshold_upper=crit_cfg.get("threshold_upper"),
        label=crit_cfg.get("label", crit_cfg["criterion_id"]),
        rationale=crit_cfg.get("rationale", ""),
        critical=bool(crit_cfg.get("critical", False)),
        weight=float(crit_cfg.get("weight", 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnf-run-milestone", description="GNF Milestone Runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--milestone-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-metrics", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")

    runner = MilestoneRunner(config_path=args.config, verbose=args.verbose)

    if args.list_metrics:
        for m in runner.list_required_metrics():
            print(f"  - {m}")
        sys.exit(0)

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics: dict[str, Any] = json.load(fh)

    if args.milestone_id:
        gate = runner.run_single_milestone(args.milestone_id, metrics)
        print(gate.summary())
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.milestone_id}_gate.json").write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    else:
        report = runner.run(metrics, output_dir=args.output_dir)
        print(report.summary())
        sys.exit(0 if report.overall_status() == GateStatus.PASS else 1)


if __name__ == "__main__":
    main()
