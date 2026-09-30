from scapy.all import sniff, IP, TCP, UDP, ICMP
from collections import defaultdict, deque
import pandas as pd
import joblib
import time
import threading
import requests
import os
from dotenv import load_dotenv

MODEL = joblib.load("nids_model_xgboost.pkl")
LABEL_ENCODER = joblib.load("label_encoder.pkl")
FEATURE_COLUMNS = pd.read_csv("cicids2017_downsampled.csv", nrows=1).drop(columns=['Label']).columns.tolist()

FLOW_TIMEOUT = 5.0
active_flows = defaultdict(list)
recent_alerts = deque(maxlen=200)
recent_flow_starts = deque(maxlen=5000)  # for time-window features


load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
ALERT_THRESHOLD_CLASSES = {"DDoS", "Bot", "Infiltration", "Heartbleed", "Web Attack - Sql Injection"}

def send_slack_alert(alert):
    if alert["prediction"] in ALERT_THRESHOLD_CLASSES and alert["confidence"] > 0.8:
        message = {
            "text": f"🚨 *NIDS Alert*: `{alert['prediction']}` detected\n"
                    f"Source: `{alert['src_ip']}` -> `{alert['dst_ip']}:{alert['dst_port']}`\n"
                    f"Confidence: {alert['confidence']:.1%} | Time: {alert['timestamp']}"
        }
        try:
            requests.post(SLACK_WEBHOOK_URL, json=message, timeout=3)
        except Exception as e:
            print(f"Slack alert failed: {e}")

def canonical_key(ip1, port1, ip2, port2, proto):
    a, b = (ip1, port1), (ip2, port2)
    return (a, b, proto) if a <= b else (b, a, proto)

def process_packet(pkt):
    if IP not in pkt:
        return
    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst

    if TCP in pkt:
        proto, sport, dport = "tcp", pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        proto, sport, dport = "udp", pkt[UDP].sport, pkt[UDP].dport
    elif ICMP in pkt:
        proto, sport, dport = "icmp", 0, 0
    else:
        return

    key = canonical_key(src_ip, sport, dst_ip, dport, proto)
    now = time.time()
    active_flows[key].append({"time": now, "size": len(pkt), "src": src_ip, "dst": dst_ip, "dport": dport})

def flush_expired_flows():
    """Runs periodically: close out flows that have gone quiet, build a
    simplified feature vector, and get a real model prediction."""
    now = time.time()
    to_remove = []
    for key, pkts in list(active_flows.items()):
        if now - pkts[-1]["time"] > FLOW_TIMEOUT:
            if len(pkts) >= 1:
                emit_prediction(pkts, key)
            to_remove.append(key)
    for key in to_remove:
        del active_flows[key]

def emit_prediction(pkts, key):
    """NOTE: Our live features are a simplified subset compared to CICFlowMeter's
    78 features used for training. For a real production match we'd need to compute
    the exact same 78 features live. Here we build a best-effort row with sane
    defaults for missing columns, which is a known simplification - documented
    in the README as a limitation of the real-time path."""
    src_ip = pkts[0]["src"]
    duration = pkts[-1]["time"] - pkts[0]["time"]
    total_bytes = sum(p["size"] for p in pkts)

    row = {col: 0 for col in FEATURE_COLUMNS}
    if "Flow Duration" in row:
        row["Flow Duration"] = duration * 1_000_000  # CICFlowMeter uses microseconds
    if "Total Fwd Packets" in row:
        row["Total Fwd Packets"] = len(pkts)
    if "Total Length of Fwd Packets" in row:
        row["Total Length of Fwd Packets"] = total_bytes
    if "Destination Port" in row:
        row["Destination Port"] = pkts[0]["dport"]

    X_row = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred_encoded = MODEL.predict(X_row)[0]
    pred_label = LABEL_ENCODER.inverse_transform([pred_encoded])[0]
    confidence = MODEL.predict_proba(X_row).max()

    alert = {
        "timestamp": time.strftime("%H:%M:%S"),
        "src_ip": src_ip,
        "dst_ip": pkts[0]["dst"],
        "dst_port": pkts[0]["dport"],
        "prediction": pred_label,
        "confidence": round(float(confidence), 3),
    }
    recent_alerts.append(alert)
    send_slack_alert(alert)
    print(f"[{alert['timestamp']}] {src_ip} -> {alert['dst_ip']}:{alert['dst_port']} => {pred_label} ({alert['confidence']})")

def background_flusher():
    while True:
        time.sleep(2)
        flush_expired_flows()

if __name__ == "__main__":
    print("Starting live capture... (Ctrl+C to stop)")
    print("NOTE: Live features are simplified vs. training data - predictions are")
    print("      illustrative of the real-time pipeline, not production-accuracy.\n")

    flusher = threading.Thread(target=background_flusher, daemon=True)
    flusher.start()

    sniff(prn=process_packet, store=False)