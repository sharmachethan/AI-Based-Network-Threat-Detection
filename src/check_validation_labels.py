
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
INDICES_FILE = PROJECT_ROOT / "data" / "threshold_validation_indices.npy"

CHUNK_SIZE = 100_000


def main():
    validation_indices = np.sort(np.load(INDICES_FILE))
    labels = []
    position = 0

    print("Reading validation labels in chunks...", flush=True)

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=["Target"],
        chunksize=CHUNK_SIZE,
    ):
        end_position = position + len(chunk)

        left = np.searchsorted(validation_indices, position)
        right = np.searchsorted(validation_indices, end_position)

        if left < right:
            local_indices = (
                validation_indices[left:right] - position
            )
            selected_labels = chunk["Target"].to_numpy()[local_indices]
            labels.append(selected_labels)

        position = end_position

    if labels:
        y = np.concatenate(labels)
    else:
        y = np.array([])

    print("\nVALIDATION LABEL SUMMARY")
    print("=" * 35)
    print(f"Validation records: {len(y):,}")
    print(f"Benign: {int((y == 0).sum()):,}")
    print(f"Attack: {int((y == 1).sum()):,}")
    print(f"Invalid labels: {int((~np.isin(y, [0, 1])).sum()):,}")

    if len(y) != len(validation_indices):
        raise RuntimeError(
            "The number of retrieved labels does not match "
            "the validation-index count."
        )


if __name__ == "__main__":
    main()
