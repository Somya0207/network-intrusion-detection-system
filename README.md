# 🛡️ Network Intrusion Detection System (NIDS)

A machine learning-based network intrusion detection system with a live monitoring dashboard, built to classify network traffic into 15 categories (14 attack types + benign) using real network flow data.

---

## Overview

This project goes beyond a typical "train a model on a static dataset" exercise by combining:

- A **custom flow-based feature extraction pipeline** built from scratch using Scapy, applied to self-captured network traffic (via `tcpdump` and real Nmap scans, generated in an isolated lab VM)
- A **multi-class Random Forest classifier** trained on CICIDS2017, a large-scale, peer-reviewed intrusion detection research dataset (2.5M+ flows after cleaning)
- A **live-updating Streamlit dashboard** that streams real model predictions on held-out traffic, simulating a live SOC (Security Operations Center) monitoring view

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
                                                                    Random Forest
                                                                     Classifier
                                                                              |
                                                                              v
                                                                    Streamlit Live
                                                                       Dashboard
```

---

## Key Results

Trained on a class-balanced subset of CICIDS2017 (625,740 flows across 15 classes, after downsampling the dominant benign class):

- **99.66% overall test accuracy**
- Near-perfect precision/recall (>0.99) on well-represented attack types: DDoS, DoS Hulk, DoS GoldenEye, PortScan, FTP-Patator, SSH-Patator
- Honest limitations on extremely rare classes (e.g., Heartbleed: 11 total samples, Infiltration: 36 total samples) — these show lower recall, which is an expected and documented consequence of insufficient training examples rather than a modeling flaw. This mirrors a real, known challenge in intrusion detection research: rare or novel attacks are inherently hard to learn reliably from limited historical data.

Full classification report available in [`docs/model_evaluation.md`](docs/model_evaluation.md).

---

## Custom Packet Capture & Feature Extraction

In addition to using CICIDS2017 for training volume and attack-type coverage, this project includes a self-built pipeline for capturing and analyzing real network traffic end-to-end:

- Captured live traffic (`tcpdump`) and generated real attack traffic (Nmap SYN scans against a designated test target) in an isolated Kali Linux lab VM
- Built a custom flow-reconstruction engine in Python/Scapy — grouping packets into bidirectional flows using canonicalized 5-tuples (source/destination IP and port, protocol), with a configurable inactivity timeout that mirrors how production tools like Zeek define flow boundaries
- Engineered time-window aggregate features (e.g., count of distinct destination ports contacted by the same source IP in a trailing 2-second window) to capture port-scan behavior that isn't visible from any single flow viewed in isolation — this feature alone cleanly separated benign traffic (always 0) from port-scan traffic (mean ~144) in testing

See [`extract_features.py`](extract_features.py) for the full implementation.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | CICIDS2017, self-captured pcap traffic |
| Feature Engineering | Python, Scapy, Pandas, NumPy |
| Modeling | scikit-learn (Random Forest) |
| Dashboard | Streamlit, Plotly |
| Traffic Capture | tcpdump, Nmap |

---

## Project Structure

```
├── app.py                      # Streamlit live dashboard
├── extract_features.py         # Custom Scapy-based flow feature extractor
├── preprocess_cicids.py        # CICIDS2017 cleaning pipeline (memory-safe, chunked)
├── downsample_benign.py        # Class balancing for the dominant BENIGN class
├── train_model.py              # Model training + evaluation
├── nids_model.pkl              # Trained Random Forest model
├── normal_traffic.pcap         # Self-captured benign traffic sample
├── portscan_traffic.pcap       # Self-captured attack traffic sample (Nmap SYN scan)
├── traffic_features.csv        # Output of the custom flow extractor
├── requirements.txt
└── docs/
    └── model_evaluation.md     # Full classification report + analysis
```

---

## Reproducing This Project

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd network-intrusion-detection-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CICIDS2017's MachineLearningCSV.zip from the official CIC dataset page:
#    https://www.unb.ca/cic/datasets/ids-2017.html
#    Extract it into a folder named MachineLearningCVE/ in the project root

# 4. Run the pipeline
python preprocess_cicids.py
python downsample_benign.py
python train_model.py

# 5. Launch the dashboard
streamlit run app.py
```

To regenerate the self-captured pcap data, use `tcpdump`/`nmap` on a Linux machine or VM and feed the resulting `.pcap` files into `extract_features.py`.

---

## Known Limitations & Future Work

- Rare attack classes (Heartbleed, SQL Injection, Infiltration) have too few real-world samples for statistically reliable evaluation — a documented, honest limitation rather than a hidden one
- Model explainability (SHAP-based "why was this flagged" reasoning) is planned but not yet implemented
- Currently runs locally; cloud deployment for a public live demo link is a planned next step
- The self-captured flow extractor is intentionally simpler than production-grade tools (e.g., no service/protocol detection); a known edge case exists where a delayed reply arriving after the flow timeout can be misattributed to the wrong originating IP in that specific sub-flow
- Dashboard source/destination IPs are synthesized for display purposes, since CICIDS2017's ML-ready CSV format does not retain real IP addresses

---

## Author

Built as a hands-on deep dive into network security and applied machine learning — from raw packet capture through model deployment.
