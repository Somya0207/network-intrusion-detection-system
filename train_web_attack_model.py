import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading HTTP payload dataset...")
df = pd.read_csv("payload_full.csv")
df = df.dropna(subset=["payload"])  # a few payloads may be empty/NaN
df["payload"] = df["payload"].astype(str)

print(f"Shape: {df.shape}")
print(f"Attack type distribution:\n{df['attack_type'].value_counts()}")

X = df["payload"]
y = df["attack_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- TF-IDF with CHARACTER n-grams, not word n-grams ----
# SQL injection / XSS patterns rely on specific character sequences
# (quotes, angle brackets, semicolons) rather than whole "words" -
# character-level n-grams capture this much better than word-level.
print("\nVectorizing with character-level TF-IDF...")
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=5000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "web_attack_model.pkl")
joblib.dump(vectorizer, "web_attack_vectorizer.pkl")
print("\nSaved web_attack_model.pkl and web_attack_vectorizer.pkl")

# ---- Quick sanity check with a few hand-written examples ----
print("\n--- Sanity check on example payloads ---")
examples = [
    "normal search query",
    "' OR '1'='1",
    "<script>alert('xss')</script>",
    "; cat /etc/passwd",
    "../../etc/passwd",
]
examples_vec = vectorizer.transform(examples)
predictions = model.predict(examples_vec)
for ex, pred in zip(examples, predictions):
    print(f"  '{ex}' -> {pred}")