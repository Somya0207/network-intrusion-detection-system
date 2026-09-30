import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import time

print("Loading data...")
df = pd.read_csv("cicids2017_downsampled.csv")
X = df.drop(columns=['Label'])
y = df['Label']

# XGBoost requires numeric labels, not strings
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training XGBoost...")
start = time.time()
model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Training took {train_time:.1f}s")

y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print(f"\nXGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

joblib.dump(model, "nids_model_xgboost.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\nSaved nids_model_xgboost.pkl and label_encoder.pkl")

# ---- Quick comparison table vs Random Forest ----
print("\n" + "="*50)
print("Loading Random Forest for side-by-side comparison...")
rf_model = joblib.load("nids_model.pkl")
rf_pred = rf_model.predict(X.iloc[X_test.index] if hasattr(X_test, 'index') else X_test)
# Note: RF was trained on string labels directly, XGBoost on encoded - predict on same X_test
rf_pred_on_test = rf_model.predict(pd.DataFrame(X_test, columns=X.columns))
rf_acc = accuracy_score(y_test_labels, rf_pred_on_test)

print(f"\nRandom Forest accuracy: {rf_acc:.4f}")
print(f"XGBoost accuracy:       {accuracy_score(y_test, y_pred):.4f}")
print(f"XGBoost train time:     {train_time:.1f}s")