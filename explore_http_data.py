import pandas as pd

df = pd.read_csv("payload_full.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10))

label_col = 'label' if 'label' in df.columns else None
if label_col:
    print("\nLabel distribution:")
    print(df[label_col].value_counts())
else:
    print("\nNo 'label' column found — check the actual column names printed above")