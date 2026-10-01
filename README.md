# 🛡️ Network Intrusion Detection System (NIDS)

A machine learning-based network intrusion detection system with explainable predictions, a live monitoring dashboard, a real-time packet capture pipeline, and a production-style inference API — built to classify network traffic into 15 categories (14 attack types + benign) using real network flow data.

---

## Overview

This project goes beyond a typical "train a model on a static dataset" exercise by combining:

- A **custom flow-based feature extraction pipeline** built from scratch using Scapy, applied to self-captured network traffic
- A **multi-class classifier comparison** (Random Forest vs. XGBoost) trained on CICIDS2017, a large-scale, peer-reviewed intrusion detection research dataset (2.5M+ flows after cleaning)
- **SHAP-based explainability** — every prediction can be traced back to the specific features that drove it, not just a black-box label
- A **live-updating, multi-page Streamlit dashboard** (Dashboard, Alerts, Traffic Monitor, System Analytics, Reports, Settings) streaming real model predictions with live, on-demand SHAP explanations
- A **standalone FastAPI inference service**, decoupled from the dashboard
- A **real-time packet capture pipeline** (Scapy + Npcap) that reconstructs flows and generates live predictions from genuinely live traffic
- **Slack alerting** for high-severity detections
- A **retraining pipeline** with compare-and-promote logic, so a new model only replaces the deployed one if it's actually better

---

## Architecture

```
Raw traffic (pcap)                 Custom Scapy flow extractor        Feature CSV
   (self-captured)         --->     (5-tuple grouping,          --->  (own pipeline)
                                     flow timeout, time-window
                                     aggregate features)

CICIDS2017 raw CSVs        --->     Cleaning & preprocessing     --->  Combined,
   (8 files, 2.8M rows)             (memory-safe, chunked;              cleaned,
                                     dedup, encoding fixes,              balanced
                                     inf/NaN handling)                   feature CSV
                                                                              |
                                                                              v
                                                          Random Forest  /  XGBoost
                                                             Classifier  (compared)
                                                                              |
                              ------------------------------------------------------------------
                              |                    |                    |                      |
                              v                    v                    v                      v
                     Streamlit Dashboard      FastAPI Service    SHAP Explainability    Retraining Pipeline
                     (live predictions,       (/predict,         (global + per-         (compare-and-
                      confusion matrix,        /predict_sample)   prediction reasoning)   promote logic)
                      severity alerts)

Live traffic (NIC)  --->  Scapy live sniffer  --->  Real-time flow    --->  Model  --->  Slack Alert
  (Npcap driver)           (live_capture.py)         reconstruction              (high-severity only)
```

---

## Key Results

Trained on a class-balanced subset of CICIDS2017 (625,740 flows across 15 classes):

| Model | Accuracy | Notes |
|---|---|---|
| Random Forest (class-weighted) | 99.66% | Baseline; strong on all well-represented classes |
| XGBoost | **99.82%** | Better on rare classes (e.g., Infiltration recall 0.43 → 0.57, Bot precision 0.68 → 0.95) |

- Honest limitations on extremely rare classes (e.g., Heartbleed: 11 total samples, Infiltration: 36 total samples) — lower recall here is an expected, documented consequence of insufficient training examples, not a modeling flaw. This mirrors a real, known challenge in intrusion detection: rare or novel attacks are inherently hard to learn reliably from limited historical data.
- Confusion matrices (linear and log-scale) available in [`confusion_matrix.png`](confusion_matrix.png) and [`confusion_matrix_log.png`](confusion_matrix_log.png) — the log-scale version specifically surfaces misclassifications in rare classes that a linear color scale would hide.

---

## Explainability (SHAP)

Rather than treat the model as a black box, every prediction can be explained using SHAP (SHapley Additive exPlanations) — both offline and **live, inside the running dashboard**:

- **Global feature importance** ([`shap_summary.png`](shap_summary.png)) shows which features matter most across all predictions
- **Live per-alert explanations**: the dashboard includes an "Explain an Alert" panel where any alert in the current session can be selected, and SHAP computes a real, on-demand explanation showing exactly which features drove that specific prediction (and in which direction) — not a static example, but live computation on the actual flow data behind that alert

This directly answers the "how do you trust an ML-based security decision?" question that black-box classifiers can't.

---

## Custom Packet Capture & Real-Time Pipeline

In addition to CICIDS2017 for training volume and attack-type coverage, this project includes two self-built, real, working components:

**1. Offline flow extraction** ([`extract_features.py`](extract_features.py)):
- Captured real benign and attack traffic (via `tcpdump` + Nmap SYN scans) in an isolated lab environment
- Built a custom flow-reconstruction engine in Python/Scapy — canonicalized 5-tuple grouping, configurable inactivity timeout, and engineered time-window aggregate features (e.g., distinct destination ports contacted by the same source IP in a trailing 2-second window) — this single feature averaged 0 for benign traffic and ~144 for port-scan traffic, cleanly separating the classes

**2. Real-time live capture** ([`live_capture.py`](live_capture.py)):
- Uses Scapy with the Npcap driver to sniff live packets directly off a network interface
- Reconstructs flows in real time using the same timeout-based logic as the offline extractor
- Feeds completed flows to the trained model and prints live predictions as traffic occurs
- Sends Slack alerts for high-confidence, high-severity detections
- **Honest limitation**: live features are a simplified subset of the 78 features used in offline training (full real-time replication of all 78 CICFlowMeter-style features was out of scope for this iteration) — live predictions demonstrate the real-time architecture end-to-end, but are not benchmarked at the same accuracy as the offline evaluation.

---

## Inference API

A standalone FastAPI service ([`api.py`](api.py)) decouples model serving from the dashboard:

- `GET /` — health check
- `GET /classes` — list of the 15 classification labels
- `POST /predict` — submit feature values, get a prediction + confidence
- `POST /predict_sample` — convenience endpoint for demos: predicts a random real held-out row

Run with `uvicorn api:app --reload` and test with `curl`.

---

## Retraining Pipeline

[`retrain_pipeline.py`](retrain_pipeline.py) demonstrates a real MLOps pattern, not just "run training again":

1. Load current training data (optionally merged with newly labeled data)
2. Train a candidate model
3. Evaluate the candidate against the currently deployed model on the same held-out set
4. **Only promote the candidate if it performs at least as well** — the previous model is backed up automatically before any replacement

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | CICIDS2017, self-captured pcap traffic |
| Feature Engineering | Python, Scapy, Pandas, NumPy |
| Modeling | scikit-learn (Random Forest), XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Live Capture | Scapy, Npcap |
| Alerting | Slack (Incoming Webhooks) |
| Traffic Capture | tcpdump, Nmap |

---

## Project Structure

```
├── app.py                          # Streamlit live dashboard
├── api.py                          # FastAPI inference service
├── live_capture.py                 # Real-time packet capture + prediction + Slack alerting
├── extract_features.py             # Custom Scapy-based flow feature extractor (offline)
├── preprocess_cicids.py            # CICIDS2017 cleaning pipeline (memory-safe, chunked)
├── downsample_benign.py            # Class balancing for the dominant BENIGN class
├── train_model.py                  # Random Forest training + evaluation
├── train_xgboost.py                # XGBoost training + comparison vs Random Forest
├── evaluate_model.py               # SHAP explainability + confusion matrix generation
├── retrain_pipeline.py             # Compare-and-promote retraining logic
├── make_sample_data.py             # Generates the small committed dashboard sample dataset
├── nids_model.pkl                  # Trained Random Forest model
├── nids_model_xgboost.pkl          # Trained XGBoost model
├── label_encoder.pkl               # Label encoder for XGBoost
├── dashboard_sample_data.csv       # Small sample used by the deployed dashboard
├── normal_traffic.pcap             # Self-captured benign traffic sample
├── portscan_traffic.pcap           # Self-captured attack traffic sample (Nmap SYN scan)
├── traffic_features.csv            # Output of the custom flow extractor
├── confusion_matrix.png / confusion_matrix_log.png
├── shap_summary.png
├── requirements.txt
└── .env                            # Slack webhook URL (not committed — see below)
```

---

## Reproducing This Project

```bash
# 1. Clone the repo
git clone https://github.com/Somya0207/network-intrusion-detection-system.git
cd network-intrusion-detection-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional, for full retraining) Download CICIDS2017's MachineLearningCSV.zip:
#    https://www.unb.ca/cic/datasets/ids-2017.html
#    Extract into a folder named MachineLearningCVE/ in the project root

# 4. (Optional) Set up Slack alerting - create .env with:
#    SLACK_WEBHOOK_URL=your_webhook_url_here

# 5. Launch the dashboard (uses committed dashboard_sample_data.csv - no large downloads needed)
streamlit run app.py

# 6. (Optional) Run the inference API separately
uvicorn api:app --reload

# 7. (Optional, requires Npcap on Windows) Run live capture
python live_capture.py
```

---

## Known Limitations & Future Work

- Rare attack classes (Heartbleed, SQL Injection, Infiltration) have too few real-world samples for statistically reliable evaluation — a documented, honest limitation rather than a hidden one
- Live capture cannot run on cloud hosting (requires direct NIC access) — it's a local-only component by nature; the deployed dashboard uses simulated live predictions on real held-out data instead
- Live real-time features are a simplified subset of the full 78-feature offline pipeline — full real-time feature parity is a planned enhancement
- Dashboard source/destination IPs shown in the simulated stream are synthesized for display purposes, since CICIDS2017's ML-ready CSV format does not retain real IP addresses
- Slack alerting currently uses a fixed confidence/class threshold rather than a tuned, adaptive alerting policy

---

## Live Demo

**Dashboard**: https://network-intrusion-detection-system-eflspv5g9cnxf4rzfwuky2.streamlit.app/

---

## Author

Built as a hands-on deep dive into network security and applied machine learning — from raw packet capture through model deployment, explainability, and production-style serving.
