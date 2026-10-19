
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
INDICES_FILE = PROJECT_ROOT / "data" / "threshold_validation_indices.npy"
MODEL_FILE = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "threshold_comparison.json"

CHUNK_SIZE = 5_000
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def main():
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    validation_indices = np.sort(np.load(INDICES_FILE))
    y_true_parts = []
    probability_parts = []
    position = 0
    processed = 0
    prediction_seconds = 0.0

    print("Scoring validation records in batches...", flush=True)

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        end_position = position + len(chunk)

        left = np.searchsorted(validation_indices, position)
        right = np.searchsorted(validation_indices, end_position)

        if left < right:
            local_indices = validation_indices[left:right] - position
            selected = chunk.iloc[local_indices]

            X = selected[feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            y = selected["Target"].to_numpy(dtype=np.int8)

            start_time = time.perf_counter()
            probabilities = model.predict_proba(X)[:, 1]
            prediction_seconds += time.perf_counter() - start_time

            y_true_parts.append(y)
            probability_parts.append(
                probabilities.astype(np.float32)
            )
            processed += len(selected)

            del selected, X, y, probabilities

        position = end_position
        del chunk

        if processed and processed % 10_000 < CHUNK_SIZE:
            print(f"Scored {processed:,} validation records", flush=True)

    if processed != len(validation_indices):
        raise RuntimeError(
            f"Expected {len(validation_indices):,} records; "
            f"scored {processed:,}."
        )

    y_true = np.concatenate(y_true_parts)
    scores = np.concatenate(probability_parts)
    results = []

    for threshold in THRESHOLDS:
        y_pred = (scores >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "attack_precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "attack_recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "attack_f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump({
            "validation_records": int(len(y_true)),
            "validation_benign": int((y_true == 0).sum()),
            "validation_attack": int((y_true == 1).sum()),
            "prediction_time_seconds": float(prediction_seconds),
            "threshold_results": results,
            "note": (
                "Thresholds are compared on a separate validation subset. "
                "The held-out test set was not used for threshold selection."
            ),
        }, file, indent=2)

    print("\nTHRESHOLD COMPARISON")
    print("=" * 95)
    print(
        f"{'Threshold':>9} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>9} {'F1':>9} {'False positives':>16} "
        f"{'Missed attacks':>15}"
    )

    for r in results:
        print(
            f"{r['threshold']:>9.2f} "
            f"{r['accuracy']:>9.4f} "
            f"{r['attack_precision']:>10.4f} "
            f"{r['attack_recall']:>9.4f} "
            f"{r['attack_f1']:>9.4f} "
            f"{r['false_positives']:>16,} "
            f"{r['false_negatives']:>15,}"
        )

    print(f"\nSaved report: {OUTPUT_FILE}")
    print(f"Prediction time: {prediction_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
