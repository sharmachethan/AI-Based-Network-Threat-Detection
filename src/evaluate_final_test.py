
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
TEST_INDICES_FILE = PROJECT_ROOT / "data" / "test_indices.npy"
MODEL_FILE = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "final_test_evaluation.json"

CHUNK_SIZE = 10_000


def main():
    print("Loading saved model...", flush=True)
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    print("Loading test indices...", flush=True)
    test_indices = np.load(TEST_INDICES_FILE)

    y_true_parts = []
    y_pred_parts = []
    y_score_parts = []
    total_prediction_seconds = 0.0
    processed = 0

    # Read the dataset sequentially, retaining only rows in the test split.
    # This avoids loading the full 970+ MB CSV into RAM.
    print("Reading test rows in chunks...", flush=True)

    test_mask_indices = np.sort(test_indices)
    test_position = 0

    for chunk_start, chunk in enumerate(
        pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE)
    ):
        start_row = chunk_start * CHUNK_SIZE
        end_row = start_row + len(chunk)

        left = np.searchsorted(test_mask_indices, start_row, side="left")
        right = np.searchsorted(test_mask_indices, end_row, side="left")

        if left == right:
            continue

        local_indices = test_mask_indices[left:right] - start_row
        selected = chunk.iloc[local_indices]

        X = selected[feature_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        y = selected["Target"].to_numpy(dtype=np.int8)

        start_time = time.perf_counter()
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        total_prediction_seconds += time.perf_counter() - start_time

        y_true_parts.append(y)
        y_pred_parts.append(predictions.astype(np.int8))
        y_score_parts.append(probabilities.astype(np.float32))

        processed += len(selected)
        test_position += len(selected)

        if processed % 50_000 < len(selected):
            print(
                f"Processed {processed:,} / {len(test_indices):,} "
                "test records",
                flush=True,
            )

        del chunk, selected, X, y, predictions, probabilities

    if processed != len(test_indices):
        raise RuntimeError(
            f"Expected {len(test_indices):,} test rows, "
            f"but processed {processed:,}."
        )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_score = np.concatenate(y_score_parts)

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
        "evaluation": "held-out random test split",
        "warning": (
            "Random split; related flows may occur across train and test. "
            "This does not establish generalization to unseen attack families."
        ),
        "test_records": int(len(y_true)),
        "benign_records": int((y_true == 0).sum()),
        "attack_records": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix_labels": ["Benign", "Attack"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "prediction_time_seconds": float(total_prediction_seconds),
        "average_prediction_time_ms_per_record": float(
            total_prediction_seconds / len(y_true) * 1000
        ),
        "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
        "feature_count": len(feature_columns),
        "batch_size": CHUNK_SIZE,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFINAL HELD-OUT TEST RESULTS", flush=True)
    print("=" * 45)
    print(f"Records evaluated: {len(y_true):,}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
    ))
    print(
        "Average prediction time: "
        f"{results['average_prediction_time_ms_per_record']:.4f} ms/record"
    )
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
