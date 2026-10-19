
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "training_sample.csv"
MODEL_PATH = ROOT / "models" / "logistic_regression_baseline.joblib"
REPORT_PATH = ROOT / "reports" / "balanced_sample_evaluation.json"

SAMPLE_SIZE_PER_CLASS = 250
RANDOM_STATE = 42


def main():
    print("Loading saved model...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    print("Loading existing training sample...")
    df = pd.read_csv(SAMPLE_PATH)

    # Select both classes without changing the saved model.
    benign = df[df["Target"] == 0].sample(
        n=SAMPLE_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )
    attacks = df[df["Target"] == 1].sample(
        n=SAMPLE_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    evaluation = pd.concat([benign, attacks]).sample(
        frac=1,
        random_state=RANDOM_STATE,
    )

    X = evaluation[feature_columns].copy()
    y_true = evaluation["Target"].astype(int)

    # Convert invalid feature values to NaN; the saved pipeline
    # handles missing values through its fitted imputer.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    print(f"Rows evaluated: {len(evaluation)}")
    print("Actual class counts:")
    print(y_true.value_counts().sort_index().to_dict())

    print("\nRunning predictions...")
    start = time.perf_counter()
    y_pred = model.predict(X)
    elapsed = time.perf_counter() - start

    probabilities = model.predict_proba(X)[:, 1]

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "evaluation_type": "Balanced sample from existing training sample",
        "rows_evaluated": int(len(evaluation)),
        "class_counts": {
            "benign": int((y_true == 0).sum()),
            "attack": int((y_true == 1).sum()),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_prediction_time_ms_per_row": (
            elapsed * 1000 / len(evaluation)
        ),
        "warning": (
            "This is a diagnostic sample drawn from the training sample, "
            "not an independent final test set."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n===== BALANCED SAMPLE RESULTS =====")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(
        "Average prediction time per row: "
        f"{results['average_prediction_time_ms_per_row']:.4f} ms"
    )
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
