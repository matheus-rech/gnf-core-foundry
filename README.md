# gnf-core-foundry

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![R 4.x](https://img.shields.io/badge/R-4.x-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

**gnf-core-foundry** is the canonical digital biomarker pipeline engine for the Global NeuroFoundry translational neuroscience platform. It provides end-to-end infrastructure for ingesting wearable/sensor data, extracting validated digital biomarkers, assessing test-retest stability, and evaluating NIH R21/R33 milestone criteria with reproducible, auditable gating decisions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Modules](#modules)
- [Configuration](#configuration)
- [Milestone Runner](#milestone-runner)
- [API](#api)
- [Testing](#testing)
- [Docker](#docker)
- [License](#license)

---

## Overview

The pipeline is designed for translational neuroscience studies that require:

- **Rigorous test-retest reliability** via ICC and Bland-Altman analysis
- **Publication-quality statistical outputs** with confidence intervals
- **R/Python interoperability** - R's `irr` package for gold-standard ICC, Python for everything else
- **Milestone gating** - automated go/no-go decisions for NIH R21 and R33 milestones
- **Schema-validated data** - JSON Schema enforcement on all biomarker records

---

## Installation

### Prerequisites

- Python 3.11+
- R 4.x with `irr` package
- pip

### Standard Install

```bash
git clone https://github.com/GlobalNeuroFoundry/gnf-core-foundry.git
cd gnf-core-foundry
pip install -e .
```

### Install with all extras

```bash
pip install -e ".[dev,api]"
```

### Install R dependency

```r
install.packages("irr")
```

---

## Quickstart

```python
import pandas as pd
import numpy as np
from gnf_core_foundry.stability.icc import compute_icc
from gnf_core_foundry.stability.bland_altman import bland_altman_analysis, plot_bland_altman

result = compute_icc(
    data=data,
    subjects="subject",
    raters="rater",
    measurements="score",
    icc_type="ICC2k",
)
print(f"ICC(2,k) = {result.icc_value:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

---

## Modules

| Module | Description |
|--------|-------------|
| `ingestion.data_loader` | Load CSV/Parquet/JSON, validate against biomarker schema |
| `preprocessing.signal_cleaning` | Butterworth filter, outlier removal, interpolation |
| `preprocessing.normalization` | Z-score, min-max, baseline normalization |
| `feature_engine.temporal_features` | RMS, ZCR, peak features for time-series |
| `feature_engine.spectral_features` | FFT, PSD, spectral entropy |
| `feature_engine.kinematic_features` | Velocity, jerk, SPARC smoothness |
| `stability.icc` | All 6 ICC forms (rpy2/R + pingouin fallback) |
| `stability.bland_altman` | Bias, LoA, proportional bias, publication plots |
| `stability.test_retest` | Combined reliability assessment + classification |
| `stability.mdc` | MDC, MDC%, SRD computation |
| `validation.schema_validator` | JSON Schema DataFrame validation |
| `validation.data_quality` | Completeness, consistency, plausibility reports |
| `evaluation.responsiveness` | SRM, Cohen's d, Hedges' g |
| `evaluation.convergent_validity` | Pearson/Spearman/Kendall correlations |
| `milestone_runner.runner` | YAML-driven milestone gating engine |
| `milestone_runner.gating` | Gate criteria evaluation with evidence trail |
| `milestone_runner.reporting` | JSON + Markdown milestone reports |

---

## License

MIT - see [LICENSE](LICENSE).
