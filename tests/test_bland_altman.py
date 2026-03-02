"""
tests/test_bland_altman.py
Tests for src/stability/bland_altman.py.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from stability.bland_altman import BlandAltmanResult, bland_altman_analysis, plot_bland_altman


class TestBlandAltmanAnalysis:
    def test_zero_bias(self, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert abs(result.bias) < 0.3
        assert result.n == len(m1)
        assert result.loa_lower < result.bias < result.loa_upper
        assert result.percent_within_loa > 90.0

    def test_known_bias(self, known_bias_ba):
        m1, m2 = known_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert result.bias == pytest.approx(-1.0, abs=0.15)
        assert result.bias_ci_upper < 0.0

    def test_proportional_bias_detected(self, proportional_bias_ba):
        m1, m2 = proportional_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert result.proportional_bias_present is True
        assert result.proportional_bias_p < 0.05

    def test_no_proportional_bias(self, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert result.proportional_bias_p > 0.01

    def test_result_dataclass_fields(self, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert isinstance(result, BlandAltmanResult)
        assert result.sd_diff > 0
        assert result.loa_lower < result.loa_upper
        assert 0.0 <= result.percent_within_loa <= 100.0

    def test_confidence_interval_for_bias(self, known_bias_ba):
        m1, m2 = known_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert result.bias_ci_lower < result.bias < result.bias_ci_upper

    def test_loa_ci_structure(self, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        assert result.loa_lower_ci_lower < result.loa_lower < result.loa_lower_ci_upper
        assert result.loa_upper_ci_lower < result.loa_upper < result.loa_upper_ci_upper

    def test_nan_removal(self):
        m1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        m2 = np.array([1.1, 2.1, 3.1, np.nan, 5.1])
        result = bland_altman_analysis(m1, m2)
        assert result.n == 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            bland_altman_analysis(np.array([1, 2, 3]), np.array([1, 2]))

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError, match=">= 3"):
            bland_altman_analysis(np.array([1.0, 2.0]), np.array([1.1, 2.1]))

    def test_summary_dict(self, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        d = result.summary_dict()
        assert "bias" in d
        assert "loa_lower" in d
        assert "loa_upper" in d
        assert "proportional_bias_present" in d


class TestPlotBlandAltman:
    def test_returns_figure(self, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        fig = plot_bland_altman(result, title="Test Plot")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_save_to_file(self, tmp_path, zero_bias_ba):
        m1, m2 = zero_bias_ba
        result = bland_altman_analysis(m1, m2)
        save_path = str(tmp_path / "ba_plot.png")
        fig = plot_bland_altman(result, title="Save Test", save_path=save_path)
        assert (tmp_path / "ba_plot.png").exists()
        plt.close(fig)

    def test_proportional_bias_plot(self, proportional_bias_ba):
        m1, m2 = proportional_bias_ba
        result = bland_altman_analysis(m1, m2)
        fig = plot_bland_altman(result, show_regression=True, title="Prop Bias")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
