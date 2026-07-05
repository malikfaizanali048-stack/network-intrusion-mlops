# Network Intrusion Detection with MLOps Pipeline

An AI-powered network intrusion detection system built on the CICIDS2017 dataset, 
wrapped in a full production-grade MLOps pipeline — from model training to 
containerized deployment, CI/CD automation, and live monitoring.

## Project Status: Week 1 Complete (Data + Modeling)

## Overview

This project detects malicious network traffic (DDoS, Port Scans, Brute Force, 
Bot activity, Web Attacks, and more) using machine learning, then wraps the model 
in a production-style pipeline: Docker containerization, Kubernetes orchestration, 
automated CI/CD, and real-time monitoring with Prometheus + Grafana.

The goal isn't just training an accurate model — it's demonstrating the full 
lifecycle of operating an ML system in production, including honest evaluation 
of where the model struggles and why.

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
due to stronger performance on the hardest attack categories.

### Feature Importance Validation
Checked for data leakage before trusting results — top predictive features were 
legitimate traffic behavior signals (packet size variance, segment sizes), not 
shortcuts like port number (which contributed only 2.5% importance).

### Anomaly Detection Experiment (Isolation Forest)
Tested an unsupervised Isolation Forest as a hypothesis for catching stealthy 
attacks like Bot traffic that evade the supervised classifier.

**Result: Hypothesis disproven.** Isolation Forest performed worse on Bot traffic 
(2.4% detection) because Bot and similar stealthy attacks are specifically designed 
to mimic normal traffic statistically. It did perform well on high-volume/flood 
attacks like DoS Hulk (81.8%), suggesting unsupervised anomaly detection is better 
suited as a safety net for novel, high-deviation attacks rather than stealthy, 
known attack patterns.

## Tech Stack

- **ML:** Python, scikit-learn, XGBoost, pandas, numpy
- **Experiment Tracking:** MLflow (SQLite backend)
- **Coming in Week 2-4:** FastAPI, Docker, Kubernetes, GitHub Actions, Prometheus, Grafana

## Project Structure

    network-intrusion-mlops/
    ├── data/               # Raw and processed datasets (not tracked in git)
    ├── models/             # Saved trained models (not tracked in git)
    ├── notebooks/          # Jupyter notebooks for EDA and modeling
    ├── src/                # Pipeline source code (API, training scripts)
    ├── requirements.txt    # Python dependencies
    └── README.md

## Setup

    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Next Steps

- Week 2: FastAPI serving layer + Docker containerization
- Week 3: Kubernetes deployment + CI/CD with GitHub Actions
- Week 4: Prometheus + Grafana monitoring, drift detection with Evidently AIs