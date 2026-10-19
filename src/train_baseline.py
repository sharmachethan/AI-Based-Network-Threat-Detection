
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "training_sample.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "Target"
EXCLUDED = {"Target", "Original_Label"}
RANDOM_STATE = 42


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("Loading the 100,000-row training sample...")
    df = pd.read_csv(DATA_FILE, low_memory=True)

    # Keep numeric network-flow features only.
    feature_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col not in EXCLUDED
    ]

    if not feature_columns:
        raise ValueError("No numeric input features found.")

    X = df[feature_columns]
    y = df[TARGET].astype("int8")

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_columns)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    print("Generating validation predictions...")
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 60)
    print("BASELINE VALIDATION RESULTS")
    print("=" * 60)
    print(classification_report(
        y_val,
        predictions,
        target_names=["Benign", "Attack"],
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

    model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
    }, model_path)

    print(f"\nSaved model: {model_path}")
    print("Validation only: the separate test set has not been used.")


if __name__ == "__main__":
    main()
