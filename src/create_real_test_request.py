
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "binary_network_flows.csv"
TEST_INDICES_FILE = ROOT / "data" / "test_indices.npy"
MODEL_FILE = ROOT / "models" / "logistic_regression_baseline.joblib"

REQUEST_FILE = ROOT / "data" / "real_test_prediction_request.json"
LABEL_FILE = ROOT / "data" / "real_test_expected_label.json"

CHUNK_SIZE = 10_000


def main():
    import joblib

    saved = joblib.load(MODEL_FILE)
    feature_columns = saved["feature_columns"]
    test_indices = np.load(TEST_INDICES_FILE)

    # Choose the first held-out test record.
    target_index = int(test_indices[0])

    print(f"Selected original row index: {target_index}", flush=True)
    print("Searching for that row in CSV chunks...", flush=True)

    position = 0

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        end_position = position + len(chunk)

        if position <= target_index < end_position:
            row = chunk.iloc[target_index - position]

            features = row[feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )

            # Preserve missing values as JSON null; the model pipeline
            # handles them using its fitted imputer.
            feature_values = {}
            for name, value in features.items():
                if pd.isna(value):
                    feature_values[name] = None
                else:
                    feature_values[name] = float(value)

            expected_label = int(row["Target"])
            original_label = str(row["Original_Label"])

            request = {
                "flow_id": f"cicids-test-row-{target_index}",
                "features": feature_values,
            }

            REQUEST_FILE.write_text(
                json.dumps(request, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            LABEL_FILE.write_text(
                json.dumps({
                    "source_row_index": target_index,
                    "expected_target": expected_label,
                    "original_label": original_label,
                    "note": "Reference label only; not sent to the API.",
                }, indent=2),
                encoding="utf-8",
            )

            print("\nRequest created successfully.")
            print(f"Feature count: {len(feature_values)}")
            print(f"Reference target: {expected_label}")
            print(f"Original dataset label: {original_label}")
            print(f"Request file: {REQUEST_FILE}")
            print(f"Reference label file: {LABEL_FILE}")
            print("The true label is stored separately from the API request.")
            return

        position = end_position
        del chunk

    raise RuntimeError(f"Could not find dataset row {target_index}.")


if __name__ == "__main__":
    main()
