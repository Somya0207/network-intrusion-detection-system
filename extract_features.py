from scapy.all import rdpcap, TCP, UDP, ICMP, IP
from collections import defaultdict
import pandas as pd

FLOW_TIMEOUT = 5.0  # seconds of inactivity before a "flow" is considered ended

def extract_flows(pcap_path, label):
    print(f"Reading {pcap_path} ...")
    packets = rdpcap(pcap_path)
    print(f"  {len(packets)} packets loaded")

    # Step 1: collect raw packets per 5-tuple bucket (same as before)
    raw_buckets = defaultdict(list)

    for pkt in packets:
        if IP not in pkt:
            continue
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst

        if TCP in pkt:
            proto = "tcp"
            src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
            flags = str(pkt[TCP].flags)
        elif UDP in pkt:
            proto = "udp"
            src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
            flags = ""
        elif ICMP in pkt:
            proto = "icmp"
            src_port, dst_port = 0, 0
            flags = ""
        else:
            continue

        endpoint_a, endpoint_b = (src_ip, src_port), (dst_ip, dst_port)
        if endpoint_a <= endpoint_b:
            bucket_key = (endpoint_a, endpoint_b, proto)
            direction = "fwd"
        else:
            bucket_key = (endpoint_b, endpoint_a, proto)
            direction = "bwd"

        raw_buckets[bucket_key].append({
            "time": float(pkt.time), "size": len(pkt), "flags": flags,
            "direction": direction, "real_src": src_ip,
            "real_dst": dst_ip, "real_dst_port": dst_port,
        })

    # Step 2: within each bucket, SPLIT into separate flows using the timeout
    rows = []
    for bucket_key, pkts in raw_buckets.items():
        pkts_sorted = sorted(pkts, key=lambda p: p["time"])

        current_flow = [pkts_sorted[0]]
        for p in pkts_sorted[1:]:
            gap = p["time"] - current_flow[-1]["time"]
            if gap > FLOW_TIMEOUT:
                rows.append(build_flow_row(current_flow, proto=bucket_key[2], label=label))
                current_flow = [p]
            else:
                current_flow.append(p)
        rows.append(build_flow_row(current_flow, proto=bucket_key[2], label=label))

    df = pd.DataFrame(rows)
    df = df.sort_values("start_time").reset_index(drop=True)
    df = add_time_window_features(df)
    return df

def build_flow_row(pkts_sorted, proto, label):
    start_time = pkts_sorted[0]["time"]
    duration = pkts_sorted[-1]["time"] - start_time
    fwd_bytes = sum(p["size"] for p in pkts_sorted if p["direction"] == "fwd")
    bwd_bytes = sum(p["size"] for p in pkts_sorted if p["direction"] == "bwd")
    fwd_pkts = sum(1 for p in pkts_sorted if p["direction"] == "fwd")
    bwd_pkts = sum(1 for p in pkts_sorted if p["direction"] == "bwd")
    all_flags = "".join(sorted(set("".join(p["flags"] for p in pkts_sorted))))
    origin = pkts_sorted[0]["real_src"]
    dest = pkts_sorted[0]["real_dst"]
    dest_port = pkts_sorted[0]["real_dst_port"]

    return {
        "start_time": start_time, "src_ip": origin, "dst_ip": dest,
        "dst_port": dest_port, "protocol": proto, "duration": round(duration, 6),
        "fwd_packets": fwd_pkts, "bwd_packets": bwd_pkts,
        "fwd_bytes": fwd_bytes, "bwd_bytes": bwd_bytes,
        "total_packets": fwd_pkts + bwd_pkts, "total_bytes": fwd_bytes + bwd_bytes,
        "tcp_flags": all_flags, "label": label
    }

def add_time_window_features(df):
    WINDOW = 2.0
    times = df["start_time"].values
    src_ips = df["src_ip"].values
    dst_ips = df["dst_ip"].values
    dst_ports = df["dst_port"].values

    conn_count, same_dst_count, dst_port_count = [], [], []
    for i in range(len(df)):
        window_start = times[i] - WINDOW
        mask = (times[:i] >= window_start) & (src_ips[:i] == src_ips[i])
        conn_count.append(mask.sum())
        mask_same_dst = mask & (dst_ips[:i] == dst_ips[i])
        same_dst_count.append(mask_same_dst.sum())
        mask_diff_port = mask_same_dst & (dst_ports[:i] != dst_ports[i])
        dst_port_count.append(mask_diff_port.sum())

    df["conn_count_2s"] = conn_count
    df["same_dst_count_2s"] = same_dst_count
    df["diff_port_count_2s"] = dst_port_count
    return df

if __name__ == "__main__":
    normal_df = extract_flows("normal_traffic.pcap", "normal")
    attack_df = extract_flows("portscan_traffic.pcap", "portscan")
    full_df = pd.concat([normal_df, attack_df], ignore_index=True)
    full_df.to_csv("traffic_features.csv", index=False)
    print(f"\nSaved {len(full_df)} flows to traffic_features.csv")
    print(f"  normal: {len(normal_df)}, portscan: {len(attack_df)}")