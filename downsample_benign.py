import pandas as pd
import numpy as np

TOTAL_BENIGN = 2146899   # from our earlier label count
TARGET_BENIGN = 200000
KEEP_FRACTION = TARGET_BENIGN / TOTAL_BENIGN
print(f"Keeping {KEEP_FRACTION:.4f} of BENIGN rows per chunk")

output_file = "cicids2017_downsampled.csv"
first_chunk = True
benign_kept = 0
attack_kept = 0

np.random.seed(42)  # reproducibility

for chunk in pd.read_csv("cicids2017_cleaned.csv", chunksize=100000):
    is_benign = chunk['Label'] == 'BENIGN'

    attacks = chunk[~is_benign]
    benign = chunk[is_benign]

    # Randomly keep only KEEP_FRACTION of benign rows in this chunk
    benign_sampled = benign.sample(frac=KEEP_FRACTION, random_state=42)

    combined = pd.concat([attacks, benign_sampled])
    attack_kept += len(attacks)
    benign_kept += len(benign_sampled)

    combined.to_csv(output_file, mode='a', header=first_chunk, index=False)
    first_chunk = False

print(f"\nTotal attack rows kept: {attack_kept}")
print(f"Total benign rows kept (approx): {benign_kept}")
print(f"Saved to {output_file}")