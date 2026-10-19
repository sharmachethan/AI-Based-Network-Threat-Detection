
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"
INDICES_FILE = PROJECT_ROOT / "data" / "sample_indices.npy"
OUTPUT_FILE = PROJECT_ROOT / "data" / "training_sample.csv"

CHUNK_SIZE = 50_000


def main():
    for path in (DATASET_FILE, INDICES_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Rename or remove it manually before running again.")
        return

    selected_indices = np.load(INDICES_FILE)
    selected_set = set(selected_indices.tolist())

    current_position = 0
    rows_written = 0
    first_chunk = True

    try:
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
                selected_rows = chunk.loc[mask]

                selected_rows.to_csv(
                    OUTPUT_FILE,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )

                rows_written += len(selected_rows)
                first_chunk = False

            current_position = chunk_end
            print(f"Scanned {current_position:,} rows")

        if rows_written != len(selected_indices):
            raise ValueError(
                f"Expected {len(selected_indices):,} rows, "
                f"but extracted {rows_written:,}."
            )

        print("\n" + "=" * 55)
        print("TRAINING SAMPLE EXTRACTED")
        print("=" * 55)
        print(f"Rows extracted: {rows_written:,}")
        print(f"Output: {OUTPUT_FILE}")
        print("The original dataset was not modified.")

    except Exception:
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        raise


if __name__ == "__main__":
    main()
