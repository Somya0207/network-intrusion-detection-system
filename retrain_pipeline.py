import pandas as pd
import joblib
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def retrain(new_data_path=None, base_data_path="cicids2017_downsampled.csv"):
    """
    Simulates a real retraining cycle:
    1. Load existing training data
    2. Optionally merge in newly labeled data (e.g. analyst-confirmed alerts)
    3. Retrain
    4. Compare new model vs old model on a held-out set
    5. Only replace the deployed model if the new one is at least as good
    """
    print("Loading base training data...")
    df = pd.read_csv(base_data_path)

    if new_data_path:
        print(f"Merging new labeled data from {new_data_path}...")
        new_df = pd.read_csv(new_data_path)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"Combined dataset: {len(df)} rows")

    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training candidate model...")
    new_model = RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    new_model.fit(X_train, y_train)
    new_accuracy = accuracy_score(y_test, new_model.predict(X_test))
    print(f"Candidate model accuracy: {new_accuracy:.4f}")

    # Compare against currently deployed model
    try:
        old_model = joblib.load("nids_model.pkl")
        old_accuracy = accuracy_score(y_test, old_model.predict(X_test))
        print(f"Currently deployed model accuracy: {old_accuracy:.4f}")
    except FileNotFoundError:
        old_accuracy = 0
        print("No existing deployed model found - deploying candidate unconditionally.")

    if new_accuracy >= old_accuracy:
        # Back up the old model before replacing it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy("nids_model.pkl", f"nids_model_backup_{timestamp}.pkl")
        joblib.dump(new_model, "nids_model.pkl")
        print(f"\n✅ New model DEPLOYED (accuracy {new_accuracy:.4f} >= {old_accuracy:.4f})")
        print(f"Previous model backed up as nids_model_backup_{timestamp}.pkl")
    else:
        print(f"\n❌ New model REJECTED (accuracy {new_accuracy:.4f} < {old_accuracy:.4f}) - keeping current model")

if __name__ == "__main__":
    retrain()