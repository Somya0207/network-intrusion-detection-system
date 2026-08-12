import pandas as pd

label_counts = pd.Series(dtype=int)

for chunk in pd.read_csv("cicids2017_cleaned.csv", chunksize=100000):
    label_counts = label_counts.add(chunk['Label'].value_counts(), fill_value=0)

print(label_counts.sort_values(ascending=False))