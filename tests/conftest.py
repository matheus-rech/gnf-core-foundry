"""
tests/conftest.py
Shared pytest fixtures for gnf-core-foundry test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def perfect_agreement_data() -> pd.DataFrame:
    subjects = [f"S{i:02d}" for i in range(1, 11)]
    scores = [3.0, 4.5, 2.1, 5.0, 3.8, 4.2, 1.5, 5.5, 2.8, 4.0]
    return pd.DataFrame({"subject": subjects * 2, "rater": ["R1"] * 10 + ["R2"] * 10, "score": scores + scores})


@pytest.fixture
def high_icc_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_subjects = 20
    true_scores = rng.uniform(1, 10, n_subjects)
    r1 = true_scores + rng.normal(0, 0.3, n_subjects)
    r2 = true_scores + rng.normal(0, 0.3, n_subjects)
    subjects = [f"S{i:03d}" for i in range(n_subjects)]
    return pd.DataFrame({"subject": subjects * 2, "rater": ["session1"] * n_subjects + ["session2"] * n_subjects, "score": np.concatenate([r1, r2])})


@pytest.fixture
def moderate_icc_data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n_subjects = 20
    true_scores = rng.uniform(1, 10, n_subjects)
    r1 = true_scores + rng.normal(0, 1.5, n_subjects)
    r2 = true_scores + rng.normal(0, 1.5, n_subjects)
    subjects = [f"S{i:03d}" for i in range(n_subjects)]
    return pd.DataFrame({"subject": subjects * 2, "rater": ["session1"] * n_subjects + ["session2"] * n_subjects, "score": np.concatenate([r1, r2])})


@pytest.fixture
def no_agreement_data() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n_subjects = 20
    r1 = rng.uniform(0, 10, n_subjects)
    r2 = rng.uniform(0, 10, n_subjects)
    subjects = [f"S{i:03d}" for i in range(n_subjects)]
    return pd.DataFrame({"subject": subjects * 2, "rater": ["R1"] * n_subjects + ["R2"] * n_subjects, "score": np.concatenate([r1, r2])})


@pytest.fixture
def multi_rater_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_subjects = 15
    true_scores = rng.uniform(2, 8, n_subjects)
    dfs = []
    for rater_id in range(1, 5):
        scores = true_scores + rng.normal(0, 0.4, n_subjects)
        dfs.append(pd.DataFrame({"subject": [f"S{i:03d}" for i in range(n_subjects)], "rater": [f"R{rater_id}"] * n_subjects, "score": scores}))
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def zero_bias_ba() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 30
    true_vals = rng.uniform(2, 8, n)
    return true_vals + rng.normal(0, 0.3, n), true_vals + rng.normal(0, 0.3, n)


@pytest.fixture
def known_bias_ba() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 30
    m1 = rng.uniform(2, 8, n)
    return m1, m1 + 1.0 + rng.normal(0, 0.2, n)


@pytest.fixture
def proportional_bias_ba() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(55)
    n = 40
    m1 = rng.uniform(1, 10, n)
    return m1, m1 * 1.15 + rng.normal(0, 0.2, n)


@pytest.fixture
def passing_r21_metrics() -> dict:
    return {"icc_2k": 0.85, "bland_altman_bias_pct": 4.5, "bland_altman_proportional_bias_p": 0.41, "data_completeness_pct": 97.0, "sample_size": 25}


@pytest.fixture
def failing_r21_metrics() -> dict:
    return {"icc_2k": 0.62, "bland_altman_bias_pct": 12.0, "bland_altman_proportional_bias_p": 0.02, "data_completeness_pct": 85.0, "sample_size": 12}


@pytest.fixture
def synthetic_accel_signal() -> tuple[np.ndarray, float]:
    fs = 100.0
    t = np.linspace(0, 10, int(fs * 10))
    sig = 1.0 * np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 2.0 * t) + 0.2 * np.random.default_rng(1).normal(0, 1, len(t))
    return sig, fs


@pytest.fixture
def synthetic_tremor_signal() -> tuple[np.ndarray, float]:
    fs = 200.0
    t = np.linspace(0, 5, int(fs * 5))
    rng = np.random.default_rng(2)
    sig = 0.8 * np.sin(2 * np.pi * 5.0 * t) + 0.2 * np.sin(2 * np.pi * 10.0 * t) + 0.05 * rng.normal(0, 1, len(t))
    return sig, fs
