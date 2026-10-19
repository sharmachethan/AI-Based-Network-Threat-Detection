
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "train_indices.npy"
SAMPLE_FILE = PROJECT_ROOT / "data" / "sample_indices.npy"
OUTPUT_FILE = PROJECT_ROOT / "data" / "threshold_validation_indices.npy"

VALIDATION_SIZE = 50_000
RANDOM_STATE = 2026


def main():
    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Keeping the existing validation indices unchanged.")
        return

    train_indices = np.load(TRAIN_FILE)
    sample_indices = np.load(SAMPLE_FILE)

    # Exclude the rows already used for baseline model development.
    remaining_indices = np.setdiff1d(
        train_indices,
        sample_indices,
        assume_unique=False,
    )

    if VALIDATION_SIZE > len(remaining_indices):
        raise ValueError("Not enough records for validation.")

    rng = np.random.default_rng(RANDOM_STATE)
    validation_indices = np.sort(
        rng.choice(
            remaining_indices,
            size=VALIDATION_SIZE,
            replace=False,
        )
    )

    np.save(OUTPUT_FILE, validation_indices)

    print("Threshold validation indices created.")
    print(f"Validation records: {len(validation_indices):,}")
    print(f"Output: {OUTPUT_FILE}")
    print("Validation indices overlap existing sample:",
          len(np.intersect1d(validation_indices, sample_indices)))
    print("Validation indices overlap test set:",
          len(np.intersect1d(
              validation_indices,
              np.load(PROJECT_ROOT / "data" / "test_indices.npy"),
          )))


if __name__ == "__main__":
    main()
