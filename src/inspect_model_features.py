
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
CHUNK_SIZE = 50_000

EXCLUDED_COLUMNS = {"Target", "Original_Label"}


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(DATASET_FILE)

    header = pd.read_csv(DATASET_FILE, nrows=0)
    columns = [str(col).strip() for col in header.columns]

    duplicates = sorted(
        name for name, count in Counter(columns).items()
        if count > 1
    )

    feature_columns = [
        col for col in columns
        if col not in EXCLUDED_COLUMNS
    ]

    missing_counts = Counter()
    infinite_counts = Counter()
    total_rows = 0

    print("Inspecting model features in chunks...")

    for chunk in pd.read_csv(
        DATASET_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()
        total_rows += len(chunk)

        # Inspect only model features, not the target or original label.
        features = chunk[feature_columns]

        for column in feature_columns:
            values = pd.to_numeric(
                features[column], errors="coerce"
            )

            missing_counts[column] += int(values.isna().sum())
            infinite_counts[column] += int(
                np.isinf(values.to_numpy(dtype=np.float64)).sum()
            )

    print("\n" + "=" * 55)
    print("MODEL FEATURE QUALITY REPORT")
    print("=" * 55)
    print(f"Rows inspected: {total_rows:,}")
    print(f"Candidate feature count: {len(feature_columns)}")
    print(f"Duplicate column names: {duplicates or 'None'}")
    print(f"Excluded columns: {sorted(EXCLUDED_COLUMNS)}")

    print("\nFeatures with missing or non-numeric values:")
    print({
        col: count for col, count in missing_counts.items()
        if count > 0
    } or "None")

    print("\nFeatures with infinite values:")
    print({
        col: count for col, count in infinite_counts.items()
        if count > 0
    } or "None")

    print("\nInspection complete. No files were modified.")


if __name__ == "__main__":
    main()
