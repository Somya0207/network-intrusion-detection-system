import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading downsampled dataset...")
df = pd.read_csv("cicids2017_downsampled.csv")
print(f"Shape: {df.shape}")

# Separate features (X) from label (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Train/test split — stratify to keep class proportions consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Train a Random Forest — strong baseline for tabular data, no scaling needed
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',  # compensates for remaining imbalance
    random_state=42,
    n_jobs=-1  # use all CPU cores
)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating on test set...")
y_pred = model.predict(X_test)

print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model for later use (dashboard integration)
joblib.dump(model, "nids_model.pkl")
print("\nModel saved to nids_model.pkl")