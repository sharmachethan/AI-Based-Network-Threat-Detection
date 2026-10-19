
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
TEST_INDICES_FILE = PROJECT_ROOT / "data" / "test_indices.npy"
MODEL_FILE = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
ORIGINAL_REPORT = PROJECT_ROOT / "reports" / "final_test_evaluation.json"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "threshold_060_test_evaluation.json"

THRESHOLD = 0.60
CHUNK_SIZE = 2_000


def main():
    print("Loading model and original report...", flush=True)

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    with ORIGINAL_REPORT.open("r", encoding="utf-8") as file:
        original_results = json.load(file)

    test_indices = np.sort(np.load(TEST_INDICES_FILE))
    total_records = len(test_indices)

    tn = fp = fn = tp = 0
    benign_count = attack_count = 0
    processed = 0
    total_prediction_seconds = 0.0
    row_start = 0

    print(
        f"Evaluating {total_records:,} held-out records "
        f"at threshold {THRESHOLD:.2f}...",
        flush=True,
    )

    reader = pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE)

    try:
        for chunk in reader:
            start_row = row_start
            end_row = start_row + len(chunk)
            row_start = end_row

            left = np.searchsorted(test_indices, start_row, side="left")
            right = np.searchsorted(test_indices, end_row, side="left")

            if left == right:
                del chunk
                continue

            local_indices = test_indices[left:right] - start_row
            selected = chunk.iloc[local_indices]

            X = selected[feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            y = selected["Target"].to_numpy(dtype=np.int8)

            start_time = time.perf_counter()
            scores = model.predict_proba(X)[:, 1]
            total_prediction_seconds += time.perf_counter() - start_time

            predicted_attack = scores >= THRESHOLD
            actual_attack = y == 1
            actual_benign = ~actual_attack

            tp += int(np.sum(predicted_attack & actual_attack))
            fn += int(np.sum(~predicted_attack & actual_attack))
            fp += int(np.sum(predicted_attack & actual_benign))
            tn += int(np.sum(~predicted_attack & actual_benign))

            attack_count += int(np.sum(actual_attack))
            benign_count += int(np.sum(actual_benign))
            processed += len(y)

            if processed % 50_000 < len(y):
                print(
                    f"Processed {processed:,} / {total_records:,}",
                    flush=True,
                )

            del chunk, selected, X, y, scores
            gc.collect()

    finally:
        reader.close()

    if processed != total_records:
        raise RuntimeError(
            f"Expected {total_records:,} test rows; processed {processed:,}."
        )

    total = tn + fp + fn + tp

    def metrics(precision, recall, support):
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

    benign_precision = tn / (tn + fn) if tn + fn else 0.0
    benign_recall = tn / (tn + fp) if tn + fp else 0.0
    attack_precision = tp / (tp + fp) if tp + fp else 0.0
    attack_recall = tp / (tp + fn) if tp + fn else 0.0

    benign = metrics(benign_precision, benign_recall, benign_count)
    attack = metrics(attack_precision, attack_recall, attack_count)

    macro_avg = {
        key: (benign[key] + attack[key]) / 2
        for key in ("precision", "recall", "f1-score")
    }
    macro_avg["support"] = total

    weighted_avg = {
        key: (
            benign[key] * benign_count + attack[key] * attack_count
        ) / total
        for key in ("precision", "recall", "f1-score")
    }
    weighted_avg["support"] = total

    results = {
        "evaluation": "held-out random test split",
        "decision_threshold": THRESHOLD,
        "warning": (
            "Random split; related flows may occur across train and test. "
            "This does not establish generalization to unseen attack families."
        ),
        "test_records": total_records,
        "benign_records": benign_count,
        "attack_records": attack_count,
        "accuracy": (tn + tp) / total,
        "roc_auc_reference": original_results["roc_auc"],
        "roc_auc_note": (
            "Reference only: copied from the original held-out evaluation "
            "using the same model and test split. ROC-AUC was not recalculated "
            "in this threshold-specific run."
        ),
        "confusion_matrix_labels": ["Benign", "Attack"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "classification_report": {
            "Benign": benign,
            "Attack": attack,
            "macro avg": macro_avg,
            "weighted avg": weighted_avg,
        },
        "prediction_time_seconds": total_prediction_seconds,
        "average_prediction_time_ms_per_record": (
            total_prediction_seconds / total_records * 1000
        ),
        "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
        "feature_count": len(feature_columns),
        "batch_size": CHUNK_SIZE,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nTHRESHOLD 0.60 — HELD-OUT TEST RESULTS")
    print("=" * 45)
    print(f"Records evaluated: {total_records:,}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC reference: {results['roc_auc_reference']:.4f}")
    print(f"Confusion matrix: {results['confusion_matrix']}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
