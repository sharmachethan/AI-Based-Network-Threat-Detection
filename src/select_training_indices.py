
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "train_indices.npy"
OUTPUT_FILE = PROJECT_ROOT / "data" / "sample_indices.npy"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Rename or remove it manually before running again.")
        return

    train_indices = np.load(TRAIN_FILE, mmap_mode="r")

    if SAMPLE_SIZE > len(train_indices):
        raise ValueError("Sample size exceeds training-set size.")

    rng = np.random.default_rng(RANDOM_STATE)

    # Select positions within the training-index array.
    selected_positions = rng.choice(
        len(train_indices),
        size=SAMPLE_SIZE,
        replace=False,
    )

    # Convert selected positions to original CSV row indices.
    selected_indices = np.sort(
        np.asarray(train_indices[selected_positions])
    )

    np.save(OUTPUT_FILE, selected_indices)

    print("Training sample indices created.")
    print(f"Selected records: {len(selected_indices):,}")
    print(f"Output: {OUTPUT_FILE}")
    print("The full dataset and test indices were not changed.")


if __name__ == "__main__":
    main()
