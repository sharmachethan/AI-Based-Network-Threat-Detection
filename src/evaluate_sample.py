
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "training_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "sample_evaluation.json"

SAMPLE_SIZE = 500


def main():
    print("Loading saved model...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    print(f"Expected model features: {len(feature_columns)}")
    print(f"Reading up to {SAMPLE_SIZE} real dataset rows...")

    df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE)

    if "Target" not in df.columns:
        raise ValueError("The dataset must contain the Target column.")

    missing_features = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    y_true = pd.to_numeric(
        df["Target"], errors="raise"
    ).astype(int)

    if not y_true.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    print("Running predictions...")
    start = time.perf_counter()

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    elapsed = time.perf_counter() - start

    results = {
        "evaluation_type": (
            "Preliminary sanity check on training-sample rows; "
            "not an independent test evaluation"
        ),
        "rows_evaluated": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
        "average_prediction_time_ms": (
            float(elapsed * 1000 / len(df))
            if len(df) else 0.0
        ),
    }

    if len(set(y_true)) == 2:
        results["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
    else:
        results["roc_auc"] = None
        print(
            "Note: ROC-AUC cannot be calculated because "
            "this sample contains only one target class."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n===== PRELIMINARY SAMPLE RESULTS =====")
    print(f"Rows evaluated: {len(df)}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Confusion matrix: {results['confusion_matrix']}")
    print(f"ROC-AUC: {results['roc_auc']}")
    print(
        "Average prediction time per row: "
        f"{results['average_prediction_time_ms']:.4f} ms"
    )
    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
