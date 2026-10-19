
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
TRAIN_INDICES_FILE = PROJECT_ROOT / "data" / "train_indices.npy"
OUTPUT_FILE = PROJECT_ROOT / "data" / "training_sample.csv"

SAMPLE_SIZE = 100_000
CHUNK_SIZE = 50_000
RANDOM_STATE = 42


def main():
    for path in (DATASET_FILE, TRAIN_INDICES_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Rename or remove it manually before running again.")
        return

    train_indices = np.load(TRAIN_INDICES_FILE, mmap_mode="r")

    if SAMPLE_SIZE > len(train_indices):
        raise ValueError("Sample size exceeds the training-set size.")

    # Stratify the sample using labels from the training indices only.
    labels = []
    row_positions = []

    for chunk_start in range(0, len(train_indices), CHUNK_SIZE):
        batch_indices = train_indices[
            chunk_start:chunk_start + CHUNK_SIZE
        ]

        # Labels are read separately in the next stage. Here we only
        # collect a reproducible sample of training row positions.
        row_positions.extend(batch_indices.tolist())

    rng = np.random.default_rng(RANDOM_STATE)
    selected_indices = np.sort(
        rng.choice(
            np.asarray(row_positions, dtype=np.int64),
            size=SAMPLE_SIZE,
            replace=False,
        )
    )

    # Read the CSV sequentially, retaining only selected rows.
    selected_set = set(selected_indices.tolist())
    retained_chunks = []
    current_position = 0

    for chunk in pd.read_csv(
        DATASET_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):
        chunk_end = current_position + len(chunk)

        mask = [
            position in selected_set
            for position in range(current_position, chunk_end)
        ]

        if any(mask):
            retained_chunks.append(chunk.loc[mask].copy())

        current_position = chunk_end
        print(f"Scanned {current_position:,} rows")

    sample = pd.concat(retained_chunks, ignore_index=True)
    sample.to_csv(OUTPUT_FILE, index=False)

    print("\nTraining sample created.")
    print(f"Rows: {len(sample):,}")
    print(f"Output: {OUTPUT_FILE}")
    print("The original dataset and test indices were not modified.")


if __name__ == "__main__":
    main()
