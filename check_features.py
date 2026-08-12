import pandas as pd
df = pd.read_csv("traffic_features.csv")

print(df.groupby("label")[["duration", "total_packets", "diff_port_count_2s"]].describe().T)
print()
print(df[df["label"]=="portscan"]["tcp_flags"].value_counts())