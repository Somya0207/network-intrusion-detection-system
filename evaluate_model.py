import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

print("Loading model and data...")
model = joblib.load("nids_model.pkl")
df = pd.read_csv("cicids2017_downsampled.csv")

X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 1. Confusion Matrix ----
print("Generating confusion matrix...")
y_pred = model.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('NIDS Confusion Matrix - 15 Classes')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Saved confusion_matrix.png")

# Log-scale version (raw counts span from 2 to 40,000 - linear scale hides small classes)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
            norm=plt.matplotlib.colors.LogNorm())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('NIDS Confusion Matrix - Log Scale (reveals rare-class errors)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_log.png', dpi=150)
print("Saved confusion_matrix_log.png")

# ---- 2. SHAP Explainability ----
print("\nComputing SHAP values (this takes a few minutes)...")
# Use a SMALL sample - SHAP on RandomForest is expensive, and we only need
# enough to explain patterns, not the whole test set
sample = X_test.sample(n=500, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Global feature importance summary (which features matter most, overall)
plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
print("Saved shap_summary.png")

# ---- 3. Explain ONE specific prediction (the "why was this flagged" moment) ----
print("\nExplaining a single PortScan prediction...")
portscan_rows = X_test[y_test == 'PortScan']
if len(portscan_rows) > 0:
    example = portscan_rows.iloc[[0]]
    example_pred = model.predict(example)[0]
    example_class_idx = list(model.classes_).index(example_pred)

    print(f"Prediction: {example_pred}")
    print("\nTop features driving this prediction:")

    example_shap = explainer.shap_values(example)
    if isinstance(example_shap, list):
        vals = example_shap[example_class_idx][0]
    else:
        vals = example_shap[0, :, example_class_idx]

    feature_impact = pd.Series(vals, index=X.columns).sort_values(key=abs, ascending=False)
    print(feature_impact.head(10))

print("\nDone. Files saved: confusion_matrix.png, confusion_matrix_log.png, shap_summary.png")