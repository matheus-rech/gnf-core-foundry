"""
tests/test_icc.py
Tests for src/stability/icc.py.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest

from stability.icc import ICCResult, _compute_icc_numpy, _compute_sem_mdc, _normalise_icc_type, compute_all_icc_forms, compute_icc, icc_summary_table


class TestNormaliseICCType:
    def test_shorthand_forms(self):
        assert _normalise_icc_type("ICC11") == "ICC11"
        assert _normalise_icc_type("ICC2k") == "ICC2k"
        assert _normalise_icc_type("ICC3k") == "ICC3k"

    def test_shrout_fleiss_notation(self):
        assert _normalise_icc_type("ICC(1,1)") == "ICC(1,1)"
        assert _normalise_icc_type("ICC(2,k)") == "ICC(2,k)"

    def test_case_insensitive(self):
        assert _normalise_icc_type("icc2k") == "ICC2k"
        assert _normalise_icc_type("Icc21") == "ICC21"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ICC type"):
            _normalise_icc_type("ICC99")


class TestSemMdc:
    def test_perfect_reliability(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sem, mdc95, mdc_pct = _compute_sem_mdc(values, icc=1.0)
        assert sem == pytest.approx(0.0, abs=1e-9)
        assert mdc95 == pytest.approx(0.0, abs=1e-9)

    def test_zero_reliability(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sd = float(np.std(values, ddof=1))
        sem, mdc95, _ = _compute_sem_mdc(values, icc=0.0)
        assert sem == pytest.approx(sd, rel=1e-6)

    def test_mdc_formula(self):
        values = np.array([3.0, 4.5, 5.0, 3.5, 4.0])
        sem, mdc95, _ = _compute_sem_mdc(values, icc=0.80)
        expected = 1.96 * math.sqrt(2.0) * sem
        assert mdc95 == pytest.approx(expected, rel=1e-6)


class TestNumpyICC:
    def _make_wide(self, n_subjects=10, noise=0.3, seed=0):
        rng = np.random.default_rng(seed)
        true_vals = rng.uniform(1, 10, n_subjects)
        r1 = true_vals + rng.normal(0, noise, n_subjects)
        r2 = true_vals + rng.normal(0, noise, n_subjects)
        return np.column_stack([r1, r2])

    @pytest.mark.parametrize("icc_type", ["ICC11", "ICC21", "ICC31", "ICC1k", "ICC2k", "ICC3k"])
    def test_all_forms_return_value_in_range(self, icc_type):
        wide = self._make_wide(n_subjects=20, noise=0.3)
        icc, ci_lo, ci_hi, F, df1, df2 = _compute_icc_numpy(wide, icc_type, 0.95)
        assert -1.0 <= icc <= 1.0
        assert ci_lo <= icc <= ci_hi
        assert F > 0

    def test_perfect_agreement(self):
        true_vals = np.linspace(1, 10, 15)
        wide = np.column_stack([true_vals, true_vals])
        icc, *_ = _compute_icc_numpy(wide, "ICC21", 0.95)
        assert icc == pytest.approx(1.0, abs=0.01)

    def test_no_agreement(self):
        rng = np.random.default_rng(77)
        wide = rng.uniform(0, 10, (20, 2))
        icc, *_ = _compute_icc_numpy(wide, "ICC21", 0.95)
        assert icc < 0.5


class TestComputeICCPython:
    def test_high_icc_data(self, high_icc_data):
        result = compute_icc(high_icc_data, "subject", "rater", "score", icc_type="ICC2k", use_r=False)
        assert isinstance(result, ICCResult)
        assert result.icc_value > 0.7
        assert result.ci_lower < result.icc_value < result.ci_upper
        assert result.n_subjects == 20
        assert result.n_raters == 2
        assert result.method == "pingouin_python"

    def test_moderate_icc_data(self, moderate_icc_data):
        result = compute_icc(moderate_icc_data, "subject", "rater", "score", icc_type="ICC2k", use_r=False)
        assert 0.3 <= result.icc_value <= 0.9

    def test_perfect_agreement(self, perfect_agreement_data):
        result = compute_icc(perfect_agreement_data, "subject", "rater", "score", icc_type="ICC2k", use_r=False)
        assert result.icc_value == pytest.approx(1.0, abs=0.01)

    def test_sem_is_positive(self, high_icc_data):
        result = compute_icc(high_icc_data, "subject", "rater", "score", icc_type="ICC2k", use_r=False)
        assert result.sem >= 0
        assert result.mdc95 >= 0

    def test_all_icc_types_run(self, multi_rater_data):
        results = compute_all_icc_forms(multi_rater_data, "subject", "rater", "score", use_r=False)
        assert len(results) == 6
        for form, res in results.items():
            assert isinstance(res, ICCResult)
            assert -1.0 <= res.icc_value <= 1.0

    def test_summary_table(self, high_icc_data):
        results = compute_all_icc_forms(high_icc_data, "subject", "rater", "score", use_r=False)
        table = icc_summary_table(results)
        assert len(table) == 6
        assert "ICC" in table.columns
        assert "MDC95" in table.columns

    def test_missing_column_raises(self, high_icc_data):
        with pytest.raises(ValueError, match="Column 'nonexistent'"):
            compute_icc(high_icc_data, "subject", "nonexistent", "score", use_r=False)

    def test_non_numeric_measurements_raises(self, high_icc_data):
        df = high_icc_data.copy()
        df["score"] = df["score"].astype(str)
        with pytest.raises(TypeError, match="numeric"):
            compute_icc(df, "subject", "rater", "score", use_r=False)

    def test_too_few_subjects_raises(self):
        df = pd.DataFrame({"subject": ["S1", "S1"], "rater": ["R1", "R2"], "score": [3.0, 3.1]})
        with pytest.raises(ValueError, match=">= 2"):
            compute_icc(df, "subject", "rater", "score", use_r=False)

    def test_unknown_icc_type_raises(self, high_icc_data):
        with pytest.raises(ValueError, match="Unknown ICC type"):
            compute_icc(high_icc_data, "subject", "rater", "score", icc_type="ICC99", use_r=False)
