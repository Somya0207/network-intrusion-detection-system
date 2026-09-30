import pandas as pd

df = pd.read_csv("cicids2017_downsampled.csv")
sample = df.sample(n=2000, random_state=1)
sample.to_csv("dashboard_sample_data.csv", index=False)
print(f"Saved {len(sample)} rows to dashboard_sample_data.csv")