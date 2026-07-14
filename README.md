# Network Intrusion Detection with MLOps Pipeline

An AI-powered network intrusion detection system built on the CICIDS2017 dataset, 
wrapped in a full production-grade MLOps pipeline — covering model training, 
containerized deployment, CI/CD-governed retraining, explainability, monitoring, 
and alerting.

## Project Status: Weeks 1-4 Complete, Group E In Progress

## Overview

This project detects malicious network traffic (DDoS, Port Scans, Brute Force, 
Bot activity, Web Attacks, and more) using machine learning, wrapped in a 
production-style pipeline: Docker containerization, Kubernetes orchestration, 
CI/CD-governed model retraining with human approval gates, drift detection, 
SHAP explainability, and live monitoring.

The goal isn't just training an accurate model — it's demonstrating the full 
lifecycle of operating, governing, and honestly evaluating an ML system in 
production, including documented limitations discovered through rigorous testing.

## Dataset

- **Source:** CICIDS2017 (Canadian Institute for Cybersecurity, UNB)
- **Files used:** Tuesday through Friday (7 CSV files, covering all major attack categories)
- **Final dataset:** 2,018,473 rows after cleaning and deduplication
- **Attack categories:** DoS (Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed), 
  DDoS, PortScan, Brute Force (FTP/SSH), Bot, Web Attacks (Brute Force, XSS, SQL Injection), 
  Infiltration

### Data Cleaning
- Fixed character encoding issues in Thursday's Web Attack labels
- Removed 264+ rows with missing/infinite values across files
- Removed 279,922 duplicate rows to prevent train/test leakage
- Created two label formats: binary (Attack/Benign) and multiclass 
  (12 categories, with rare classes under 100 samples grouped into "Other")

## Modeling

Two models were trained and compared on the binary classification task:

| Model | Training Time | Overall Accuracy | Bot Detection | Notes |
|---|---|---|---|---|
| Random Forest | 833s | 99.9%+ | 74.3% | Best generalization on stealthy/rare attacks |
| XGBoost | 55s | 99.9%+ | 68.7% | 15x faster, weaker on Bot & rare attacks |

**Random Forest was selected as the primary classifier** despite slower training, 
due to stronger performance on the hardest known attack categories.

### Feature Importance Validation
Checked for data leakage before trusting results — top predictive features were 
legitimate traffic behavior signals (packet size variance, segment sizes), not 
shortcuts like port number (which contributed only 2.5% importance).

### Anomaly Detection Experiment (Isolation Forest)
Tested an unsupervised Isolation Forest as a hypothesis for catching stealthy 
attacks like Bot traffic that evade the supervised classifier.

**Result: Hypothesis disproven for stealthy attacks.** Isolation Forest performed 
worse on Bot traffic (2.4% detection) because Bot and similar stealthy attacks are 
specifically designed to mimic normal traffic statistically. It performed well on 
high-volume/flood attacks like DoS Hulk (81.8%), suggesting unsupervised anomaly 
detection is better suited to statistically extreme deviations than to subtle, 
mimicry-based attacks.

## MLOps Pipeline

### Serving Layer
- **FastAPI** application serving predictions via `/predict`, with `/health` and 
  `/model-info` endpoints
- **Dockerized** for consistent, portable deployment
- **Kubernetes (Minikube)** deployment with 2 replicas and a NodePort service, 
  demonstrating orchestration and horizontal scaling
- **GitHub Actions CI pipeline** automatically tests the API on every push (lean, 
  API-specific dependency set to keep CI fast and reliable)
- **Live traffic simulator** (`simulator.py`) replays real test data against the 
  running API at randomized intervals, mimicking live traffic

### Model Registry (MLflow)
- MLflow deployed as a **real tracking server** (Dockerized, SQLite-backed) rather 
  than local file logging
- Both models — the Random Forest classifier and the Isolation Forest anomaly 
  detector — are **registered with full versioning** in the MLflow Model Registry
- FastAPI loads the model dynamically via a **`production` alias** 
  (`models:/network-intrusion-classifier@production`) rather than a hardcoded 
  pickle file, so promoting a new model version requires no code changes
- `/model-info` endpoint exposes the currently-serving model's version, run ID, 
  and creation timestamp

### Drift Detection (Evidently AI)
- Validated the drift detection pipeline with two test cases:
  - **Sanity check:** two random samples from the same distribution → 2.56% of 
    columns flagged as drifted (expected: low, confirms no false positives)
  - **Real drift validation:** BENIGN vs. ATTACK traffic → 75.64% of columns 
    flagged as drifted (expected: high, confirms the pipeline detects genuine 
    distributional shift)

### Retraining, Validation Gate, and Human-in-the-Loop Promotion
A governed model update loop, proven working end-to-end:

1. **`retrain.py`** — trains a new candidate model and registers it in MLflow 
   (not auto-promoted)
2. **`validate_and_promote.py`** — loads the current production model and the 
   newest candidate, evaluates both on the same held-out set, and compares 
   **precision and recall** (not just accuracy). The candidate must meet or 
   exceed production on both metrics to be eligible for promotion.
3. **`promote_to_production.py`** — requires a typed human confirmation before 
   moving the `production` alias to a new version — no automatic promotion.
4. **GitHub Actions workflow** (`retrain-validate.yml`) demonstrates this pattern 
   at the CI/CD level: triggered manually, connects to the MLflow server via an 
   ngrok tunnel, reads the live production model version, and pauses on a 
   **GitHub Environment protection rule** requiring manual review before 
   proceeding — proven to genuinely reach the local MLflow server from GitHub's 
   cloud runners.

**Proven result:** a retrained candidate (v3) beat the existing production model 
(v2) on both precision (0.9985 vs 0.9963) and recall (0.9988 vs 0.9953), passed 
the validation gate, and was promoted only after explicit human confirmation — 
after which the live API automatically began serving v3 with zero code changes.

*Note: for this local portfolio deployment, MLflow runs on local infrastructure 
tunneled via ngrok for CI/CD connectivity. In a production deployment, MLflow 
would be hosted on persistent, directly-reachable cloud infrastructure.*

### Explainability (SHAP)
- SHAP TreeExplainer computes per-feature contribution to ATTACK predictions
- Global summary plot and ranked feature importance saved (`shap_summary.png`, 
  `shap_feature_importance.csv`)
- `/explain` endpoint surfaces the top features driving attack classifications
- **Notable finding:** SHAP ranked `Destination Port` as the single most decisive 
  feature per-prediction, while global tree-based importance ranked it much lower 
  (2.5%) — showing it's rarely the deciding factor, but decisive when it is, 
  consistent with certain ports being strong standalone attack indicators.

### Alerting
- `alert_check.py` monitors two conditions: **data drift** (via Evidently) and 
  **attack-rate spikes** (share of recent traffic flagged as attacks)
- Alerts are logged with timestamp and severity (`alerts.log`) — a structured, 
  auditable format that could be wired to Slack/PagerDuty/email with minimal changes
- **Validated with deliberately skewed test data:** a constructed 66.67% attack-rate 
  sample correctly triggered a CRITICAL alert; normal data correctly triggered 
  no alert

## Key Finding: Novel Attack Detection Gap

To genuinely test detection of unseen attack types (not just held-out test rows 
of known types), an entire attack category — **PortScan (90,694 samples)** — was 
excluded from training from the start, for both the supervised classifier and 
the Isolation Forest.

**Result:**
| System | Detection rate on unseen PortScan |
|---|---|
| Supervised classifier (never trained on PortScan) | 0.17% |
| Isolation Forest (unsupervised, benign-only training) | 0.08% |
| Combined (either system flags it) | 0.25% |

This was rigorously verified to be a genuine finding, not a bug: feature alignment 
between the model and held-out data was confirmed exact, and the same holdout 
model correctly detected 100% of a known attack type (DDoS) on the same run, 
proving the model itself was functioning correctly — it simply had no learned 
concept of PortScan-shaped traffic.

**Root cause analysis:** comparing mean feature values showed PortScan traffic 
is *not* statistically close to BENIGN traffic — it's an extreme, distinct 
pattern (near-single-packet, ultra-short duration) in the opposite direction 
from DDoS's high-volume flood pattern. The failure isn't "PortScan blends into 
normal traffic" — it's that **neither model had any labeled example occupying 
that region of feature space**, so the decision boundary simply never learned 
to treat "minimal, ultra-short connections" as attack-like. This reveals that 
detection coverage is fundamentally bounded by the diversity of attack types 
represented in training data, regardless of how statistically distinct an 
unseen attack may be from other classes.

This finding directly motivates the retraining/validation/promotion pipeline 
built in this project: periodic retraining on newly-labeled attack types is 
necessary, not optional, for maintaining detection coverage over time.

## Tech Stack

- **ML:** Python, scikit-learn, XGBoost, pandas, numpy, SHAP
- **Serving:** FastAPI, Uvicorn, Docker, Kubernetes (Minikube)
- **MLOps:** MLflow (Model Registry + Tracking Server), GitHub Actions, ngrok
- **Monitoring:** Prometheus, Grafana, Evidently AI (drift detection)
- **Experiment tracking:** MLflow (SQLite backend)

## Project Structure

    network-intrusion-mlops/
    ├── data/                          # Raw and processed datasets (not tracked in git)
    ├── models/                        # Saved trained models (not tracked in git)
    ├── notebooks/                     # EDA and modeling notebooks
    ├── src/
    │   ├── app.py                     # FastAPI serving layer
    │   ├── Dockerfile
    │   ├── deployment.yaml            # Kubernetes deployment + service
    │   ├── simulator.py               # Live traffic simulator
    │   ├── drift_check.py             # Drift detection (sanity check)
    │   ├── drift_check_real.py        # Drift detection (real drift validation)
    │   ├── retrain.py                 # Retraining script
    │   ├── validate_and_promote.py    # Validation gate
    │   ├── promote_to_production.py   # Human-confirmed promotion
    │   ├── explain.py                 # SHAP explainability
    │   ├── alert_check.py             # Drift/attack-rate alerting
    │   ├── train_holdout.py           # Novel-attack holdout training
    │   └── test_novel_attack.py       # Novel-attack detection test
    ├── .github/workflows/
    │   ├── ci.yml                     # API smoke test on every push
    │   └── retrain-validate.yml       # Manual retrain/validation workflow
    ├── requirements.txt
    └── README.md

## Setup

    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Running Locally

    cd src
    uvicorn app:app --reload

## Next Steps

- **Group E (in progress):** rewrite traffic simulator with variable attack/benign 
  ratios and burst-pattern timing
- **Group F:** adversarial/robustness evaluation — perturb existing attack samples 
  to test evasion resistance
- **Group G:** updated architecture diagram, full end-to-end demo recording, 
  final documentation polish