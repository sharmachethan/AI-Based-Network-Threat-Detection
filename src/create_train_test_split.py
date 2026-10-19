
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"

TARGET_COLUMN = "Target"
CHUNK_SIZE = 100_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(DATASET_FILE)

    train_indices_file = OUTPUT_DIR / "train_indices.npy"
    test_indices_file = OUTPUT_DIR / "test_indices.npy"

    if train_indices_file.exists() or test_indices_file.exists():
        print("Split-index files already exist.")
        print("Rename or remove them manually before creating a new split.")
        return

    # Read only the target column to keep memory usage low.
    target_chunks = []
    for chunk in pd.read_csv(
        DATASET_FILE,
        usecols=[TARGET_COLUMN],
        chunksize=CHUNK_SIZE,
    ):
        target_chunks.append(chunk[TARGET_COLUMN].to_numpy())

    y = np.concatenate(target_chunks)

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target contains values other than 0 and 1.")

    indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    np.save(train_indices_file, train_indices)
    np.save(test_indices_file, test_indices)

    print("=" * 55)
    print("TRAIN/TEST SPLIT CREATED")
    print("=" * 55)
    print(f"Total records: {len(y):,}")
    print(f"Training records: {len(train_indices):,}")
    print(f"Testing records: {len(test_indices):,}")
    print(f"Training benign: {(y[train_indices] == 0).sum():,}")
    print(f"Training attack: {(y[train_indices] == 1).sum():,}")
    print(f"Testing benign: {(y[test_indices] == 0).sum():,}")
    print(f"Testing attack: {(y[test_indices] == 1).sum():,}")
    print(f"Random state: {RANDOM_STATE}")
    print("\nOnly split indices were saved; the dataset was not duplicated.")


if __name__ == "__main__":
    main()
