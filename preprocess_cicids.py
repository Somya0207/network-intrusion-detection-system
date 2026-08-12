import pandas as pd
import numpy as np
import glob
import os

output_file = "cicids2017_cleaned.csv"
if os.path.exists(output_file):
    os.remove(output_file)  # start fresh each run

csv_files = glob.glob("MachineLearningCVE/*.csv")

label_fixes = {
    'Web Attack \x96 Brute Force': 'Web Attack - Brute Force',
    'Web Attack \x96 XSS': 'Web Attack - XSS',
    'Web Attack \x96 Sql Injection': 'Web Attack - Sql Injection',
}

total_rows_before = 0
total_rows_after = 0
total_inf = 0
total_na = 0
first_file = True

for f in csv_files:
    print(f"\nProcessing {f} ...")
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    # Downcast numeric types to cut memory usage roughly in half
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    df['Label'] = df['Label'].str.replace(r'[^\x00-\x7F]+', '-', regex=True)

    before = len(df)
    total_rows_before += before

    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    total_inf += inf_count
    df = df.replace([np.inf, -np.inf], np.nan)

    na_count = df.isna().sum().sum()
    total_na += na_count
    df = df.dropna()

    df = df.drop_duplicates()
    after = len(df)
    total_rows_after += after

    print(f"  {before} -> {after} rows (dropped {before - after})")

    # Append to output CSV (write header only for the first file)
    df.to_csv(output_file, mode='a', header=first_file, index=False)
    first_file = False

    del df  # free memory explicitly before moving to next file

print(f"\n=== Summary ===")
print(f"Total rows before cleaning: {total_rows_before}")
print(f"Total rows after cleaning:  {total_rows_after}")
print(f"Total inf values found:     {total_inf}")
print(f"Total NaN values found:     {total_na}")
print(f"Saved cleaned data to {output_file}")