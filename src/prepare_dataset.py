
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "MachineLearningCSV"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_network_flows.csv"

CHUNK_SIZE = 50_000


def main():
    csv_files = sorted(DATASET_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {DATASET_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent accidentally appending to an old output.
    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Rename or remove it manually before running again.")
        return

    total_rows = 0
    skipped_rows = 0
    first_chunk = True

    try:
        for csv_file in csv_files:
            print(f"\nProcessing: {csv_file.name}")

            for chunk in pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                low_memory=True,
                encoding_errors="replace",
            ):
                # Standardize column names.
                chunk.columns = chunk.columns.astype(str).str.strip()

                # Standardize labels without changing attack categories.
                label_column = next(
                    (
                        col for col in chunk.columns
                        if col.lower() == "label"
                    ),
                    None,
                )

                if label_column is None:
                    raise ValueError(
                        f"No Label column found in {csv_file.name}"
                    )

                chunk[label_column] = (
                    chunk[label_column]
                    .astype("string")
                    .str.strip()
                )

                # Exclude records without a usable target label.
                valid_labels = (
                    chunk[label_column].notna()
                    & chunk[label_column].ne("")
                )
                skipped_rows += int((~valid_labels).sum())
                chunk = chunk.loc[valid_labels].copy()

                # Replace infinities in numeric columns with NaN.
                numeric_columns = chunk.select_dtypes(
                    include=[np.number]
                ).columns

                chunk[numeric_columns] = chunk[
                    numeric_columns
                ].replace([np.inf, -np.inf], np.nan)

                # Write incrementally; never load all files into RAM.
                chunk.to_csv(
                    OUTPUT_FILE,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )

                first_chunk = False
                total_rows += len(chunk)

            print(f"Rows written so far: {total_rows:,}")

        print("\n" + "=" * 60)
        print("DATASET CREATION COMPLETE")
        print("=" * 60)
        print(f"Output file: {OUTPUT_FILE}")
        print(f"Rows written: {total_rows:,}")
        print(f"Rows skipped due to missing labels: {skipped_rows:,}")
        print("Original CSV files were not modified.")

    except Exception:
        # Remove an incomplete output so it cannot be mistaken
        # for a completed dataset.
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        raise


if __name__ == "__main__":
    main()
