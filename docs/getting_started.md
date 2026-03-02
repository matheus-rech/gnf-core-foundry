# Getting Started -- gnf-core-foundry

## Prerequisites

| Dependency | Minimum Version | Notes |
|-----------|----------------|-------|
| Python | 3.11 | 3.12 also supported |
| R | 4.1 | Required for gold-standard ICC via rpy2 |
| pip | 23.0 | |

---

## 1. Clone the Repository

```bash
git clone https://github.com/GlobalNeuroFoundry/gnf-core-foundry.git
cd gnf-core-foundry
```

---

## 2. Set Up a Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\activate            # Windows
```

---

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev,api]"
```

---

## 4. Install R and the `irr` Package

### macOS (Homebrew)

```bash
brew install r
Rscript -e "install.packages('irr', repos='https://cloud.r-project.org')"
```

### Ubuntu / Debian

```bash
sudo apt-get install r-base r-base-dev
Rscript -e "install.packages('irr', repos='https://cloud.r-project.org')"
```

---

## 5. Verify Installation

```bash
python -c "from gnf_core_foundry.stability.icc import compute_icc; print('ICC module OK')"
python -c "from gnf_core_foundry.stability.bland_altman import bland_altman_analysis; print('BA module OK')"
python -c "from gnf_core_foundry.milestone_runner.runner import MilestoneRunner; print('Runner OK')"
```

---

## 6. Run the Tests

```bash
pytest tests/ -v
```

---

## 7. Quickstart -- ICC Analysis

```python
import pandas as pd
from gnf_core_foundry.stability.icc import compute_icc

data = pd.DataFrame({
    "subject": ["S01"]*2 + ["S02"]*2 + ["S03"]*2,
    "session": ["SES1", "SES2"] * 3,
    "gait_speed": [1.12, 1.10, 0.98, 0.95, 1.25, 1.22],
})

result = compute_icc(data=data, subjects="subject", raters="session",
                     measurements="gait_speed", icc_type="ICC2k", use_r=True)
print(result)
print(f"Reliability: {result.reliability_label()}")
```

---

## 8. Quickstart -- Bland-Altman Analysis

```python
import numpy as np
from gnf_core_foundry.stability.bland_altman import bland_altman_analysis, plot_bland_altman

device_a = np.array([1.10, 0.95, 1.25, 0.88, 1.42, 1.18, 0.79, 1.55])
device_b = np.array([1.08, 0.93, 1.22, 0.91, 1.40, 1.21, 0.82, 1.52])

result = bland_altman_analysis(device_a, device_b)
print(f"Bias: {result.bias:.4f}")
print(f"LoA:  [{result.loa_lower:.4f}, {result.loa_upper:.4f}]")

fig = plot_bland_altman(result, title="Device A vs Device B")
fig.savefig("bland_altman_plot.png", dpi=150, bbox_inches="tight")
```

---

## 9. Quickstart -- Milestone Evaluation

```python
from gnf_core_foundry.milestone_runner.runner import MilestoneRunner

runner = MilestoneRunner("configs/r21_template.yaml")
metrics = {
    "icc_2k": 0.82,
    "bland_altman_bias_pct": 4.5,
    "bland_altman_proportional_bias_p": 0.41,
    "data_completeness_pct": 97.0,
    "sample_size": 25,
    "cohens_d": 0.68,
}
report = runner.run(metrics, output_dir="outputs/")
print(report.summary())
```

---

## 10. Start the REST API

```bash
uvicorn gnf_core_foundry.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

---

## 11. Docker

```bash
docker build -f docker/Dockerfile -t gnf-core-foundry:latest .
docker run -p 8000:8000 gnf-core-foundry:latest
```

---

## 12. CLI Reference

```
gnf-run-milestone --config configs/r21_template.yaml \
    --metrics metrics.json \
    --output-dir outputs/
```

---

## References

- Shrout PE, Fleiss JL (1979). Intraclass correlations. *Psychol Bull*, 86(2), 420-428.
- Koo TK, Mae AY (2016). A guideline for ICC selection. *J Chiropr Med*, 15(2), 155-163.
- Bland JM, Altman DG (1986). Statistical methods for assessing agreement. *Lancet*, 1(8476), 307-310.
