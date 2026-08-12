import pandas as pd
import glob

csv_files = glob.glob("MachineLearningCVE/*.csv")
print(f"Found {len(csv_files)} files\n")

dfs = []
for f in csv_files:
    temp = pd.read_csv(f)
    temp.columns = temp.columns.str.strip()  # remove leading/trailing whitespace from ALL column names
    print(f"{f}: {temp.shape[0]} rows")
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined shape: {df.shape}")
print(f"\nFull label distribution:\n{df['Label'].value_counts()}")

df.to_csv("cicids2017_combined.csv", index=False)
print("\nSaved combined dataset to cicids2017_combined.csv")