
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"

CHUNK_SIZE = 100_000

EXPECTED_TOTAL = 2_830_743
EXPECTED_BENIGN = 2_273_097
EXPECTED_ATTACK = 557_646


def main():
    if not DATASET_FILE.exists():
        print(f"Dataset not found: {DATASET_FILE}")
        return

    counts = Counter()
    total_rows = 0

    print("Verifying binary dataset...")

    for chunk in pd.read_csv(
        DATASET_FILE,
        usecols=["Target"],
        chunksize=CHUNK_SIZE,
    ):
        counts.update(chunk["Target"].value_counts().to_dict())
        total_rows += len(chunk)

    print("\n" + "=" * 50)
    print("BINARY DATASET VERIFICATION")
    print("=" * 50)
    print(f"Total rows: {total_rows:,}")
    print(f"Benign (0): {counts.get(0, 0):,}")
    print(f"Attack (1): {counts.get(1, 0):,}")
    print(f"Unexpected labels: {set(counts) - {0, 1}}")

    matches = (
        total_rows == EXPECTED_TOTAL
        and counts.get(0, 0) == EXPECTED_BENIGN
        and counts.get(1, 0) == EXPECTED_ATTACK
        and set(counts).issubset({0, 1})
    )

    print(f"Counts match expected: {matches}")


if __name__ == "__main__":
    main()
