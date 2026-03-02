"""
tests/test_milestone_runner.py
Tests for the milestone runner, gating, and reporting modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from milestone_runner.gating import GateLogic, GateStatus, GatingCriterion, evaluate_gate
from milestone_runner.reporting import MilestoneReport
from milestone_runner.runner import MilestoneRunner


class TestGatingCriterion:
    def _make_criterion(self, metric, operator, threshold, critical=False):
        return GatingCriterion(criterion_id="C_test", metric=metric, operator=operator,
                               threshold=threshold, label="Test criterion", critical=critical)

    def test_gte_pass(self):
        c = self._make_criterion("icc", "gte", 0.75)
        result = evaluate_gate([c], {"icc": 0.80}, milestone_id="M1")
        assert result.status == GateStatus.PASS

    def test_gte_fail(self):
        c = self._make_criterion("icc", "gte", 0.75)
        result = evaluate_gate([c], {"icc": 0.60}, milestone_id="M1")
        assert result.status == GateStatus.FAIL

    def test_lte_pass(self):
        c = self._make_criterion("bias_pct", "lte", 10.0)
        result = evaluate_gate([c], {"bias_pct": 5.0}, milestone_id="M1")
        assert result.status == GateStatus.PASS

    def test_lte_fail(self):
        c = self._make_criterion("bias_pct", "lte", 10.0)
        result = evaluate_gate([c], {"bias_pct": 15.0}, milestone_id="M1")
        assert result.status == GateStatus.FAIL

    def test_eq_boolean_pass(self):
        c = self._make_criterion("flag", "eq", True)
        result = evaluate_gate([c], {"flag": True}, milestone_id="M1")
        assert result.status == GateStatus.PASS

    def test_eq_boolean_fail(self):
        c = self._make_criterion("flag", "eq", True)
        result = evaluate_gate([c], {"flag": False}, milestone_id="M1")
        assert result.status == GateStatus.FAIL

    def test_between_pass(self):
        c = GatingCriterion(criterion_id="C_between", metric="score", operator="between",
                            threshold=[2.0, 8.0], label="Range test")
        result = evaluate_gate([c], {"score": 5.0}, milestone_id="M1")
        assert result.status == GateStatus.PASS

    def test_between_fail(self):
        c = GatingCriterion(criterion_id="C_between", metric="score", operator="between",
                            threshold=[2.0, 8.0], label="Range test")
        result = evaluate_gate([c], {"score": 10.0}, milestone_id="M1")
        assert result.status == GateStatus.FAIL

    def test_metric_not_found_skip(self):
        c = self._make_criterion("missing_metric", "gte", 0.5)
        result = evaluate_gate([c], {}, milestone_id="M1")
        assert result.n_skip == 1

    def test_critical_criterion_fails_overall(self):
        c_critical = GatingCriterion(criterion_id="C_crit", metric="icc", operator="gte",
                                     threshold=0.75, critical=True, label="Critical ICC")
        c_optional = GatingCriterion(criterion_id="C_opt", metric="d", operator="gte",
                                     threshold=0.5, critical=False, label="Effect size")
        result = evaluate_gate([c_critical, c_optional], {"icc": 0.60, "d": 0.70},
                               milestone_id="M1", logic=GateLogic.OR)
        assert result.status == GateStatus.FAIL


class TestGateLogic:
    def _make_pass_fail_criteria(self):
        c_pass = GatingCriterion("C1", "x", "gte", 1.0, label="Pass")
        c_fail = GatingCriterion("C2", "y", "gte", 10.0, label="Fail")
        return c_pass, c_fail

    def test_and_logic_all_pass(self):
        c1, _ = self._make_pass_fail_criteria()
        c2 = GatingCriterion("C2", "y", "gte", 0.5, label="Also pass")
        result = evaluate_gate([c1, c2], {"x": 5.0, "y": 1.0}, logic=GateLogic.AND, milestone_id="M1")
        assert result.status == GateStatus.PASS

    def test_and_logic_one_fail(self):
        c1, c2 = self._make_pass_fail_criteria()
        result = evaluate_gate([c1, c2], {"x": 5.0, "y": 5.0}, logic=GateLogic.AND, milestone_id="M1")
        assert result.status == GateStatus.FAIL

    def test_or_logic_one_pass(self):
        c1, c2 = self._make_pass_fail_criteria()
        result = evaluate_gate([c1, c2], {"x": 5.0, "y": 5.0}, logic=GateLogic.OR, milestone_id="M1")
        assert result.status == GateStatus.PASS

    def test_or_logic_all_fail(self):
        c1, c2 = self._make_pass_fail_criteria()
        result = evaluate_gate([c1, c2], {"x": 0.5, "y": 5.0}, logic=GateLogic.OR, milestone_id="M1")
        assert result.status == GateStatus.FAIL


class TestMilestoneRunner:
    @pytest.fixture
    def r21_config_path(self):
        return Path(__file__).parent.parent / "configs" / "r21_template.yaml"

    def test_runner_loads_config(self, r21_config_path):
        if not r21_config_path.exists():
            pytest.skip("R21 config not found")
        runner = MilestoneRunner(r21_config_path)
        assert runner.phase == "R21"
        assert len(runner.milestones_config) > 0

    def test_runner_passing_metrics(self, r21_config_path, passing_r21_metrics):
        if not r21_config_path.exists():
            pytest.skip("R21 config not found")
        runner = MilestoneRunner(r21_config_path)
        report = runner.run(passing_r21_metrics)
        r21_m1 = next((r for r in report.results if r.milestone_id == "R21-M1"), None)
        if r21_m1:
            assert r21_m1.status == GateStatus.PASS

    def test_runner_failing_metrics(self, r21_config_path, failing_r21_metrics):
        if not r21_config_path.exists():
            pytest.skip("R21 config not found")
        runner = MilestoneRunner(r21_config_path)
        report = runner.run(failing_r21_metrics)
        r21_m1 = next((r for r in report.results if r.milestone_id == "R21-M1"), None)
        if r21_m1:
            assert r21_m1.status == GateStatus.FAIL

    def test_list_milestones(self, r21_config_path):
        if not r21_config_path.exists():
            pytest.skip("R21 config not found")
        runner = MilestoneRunner(r21_config_path)
        milestones = runner.list_milestones()
        assert len(milestones) >= 1
        assert "milestone_id" in milestones[0]

    def test_list_required_metrics(self, r21_config_path):
        if not r21_config_path.exists():
            pytest.skip("R21 config not found")
        runner = MilestoneRunner(r21_config_path)
        metrics = runner.list_required_metrics()
        assert "icc_2k" in metrics

    def test_runner_not_found_config_raises(self):
        with pytest.raises(FileNotFoundError):
            MilestoneRunner("nonexistent_config.yaml")


class TestMilestoneReport:
    def test_json_output(self, r21_config_path=None):
        c = GatingCriterion("C1", "icc", "gte", 0.75, label="ICC")
        gate = evaluate_gate([c], {"icc": 0.82}, milestone_id="R21-M1")
        report = MilestoneReport("R21", "Test Study", [gate])
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "report_metadata" in data
        assert "milestones" in data
        assert data["report_metadata"]["overall_status"] == "PASS"

    def test_markdown_output(self):
        c = GatingCriterion("C1", "icc", "gte", 0.75, label="ICC")
        gate = evaluate_gate([c], {"icc": 0.82}, milestone_id="R21-M1")
        report = MilestoneReport("R21", "Test Study", [gate])
        md = report.to_markdown()
        assert "# Milestone Report" in md
        assert "R21-M1" in md
        assert "PASS" in md

    def test_save_json(self, tmp_path):
        c = GatingCriterion("C1", "icc", "gte", 0.75, label="ICC")
        gate = evaluate_gate([c], {"icc": 0.82}, milestone_id="TEST-M1")
        report = MilestoneReport("TEST", "Test", [gate])
        p = report.save_json(tmp_path / "test_report.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["report_metadata"]["phase"] == "TEST"
