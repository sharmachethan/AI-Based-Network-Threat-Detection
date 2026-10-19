
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "MachineLearningCSV"
CHUNK_SIZE = 50_000


def main():
    csv_files = sorted(DATASET_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATASET_DIR}")
        return

    print("=" * 70)
    print("CIC-IDS2017 DATA QUALITY CHECK")
    print("=" * 70)

    for csv_file in csv_files:
        print(f"\nFile: {csv_file.name}")

        try:
            header = pd.read_csv(csv_file, nrows=0)
            columns = list(header.columns)

            # Report repeated column names after trimming whitespace.
            normalized = [str(col).strip() for col in columns]
            repeated = sorted({
                col for col in normalized
                if normalized.count(col) > 1
            })

            print(f"Columns: {len(columns)}")
            print(f"Repeated column names after trimming: {repeated or 'None'}")

            missing_counts = Counter()
            infinite_counts = Counter()
            label_counts = Counter()
            total_rows = 0

            for chunk in pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                low_memory=True,
            ):
                total_rows += len(chunk)

                # Check missing values.
                for col, count in chunk.isna().sum().items():
                    if count:
                        missing_counts[str(col)] += int(count)

                # Check infinite values in numeric columns.
                numeric = chunk.select_dtypes(include=[np.number])
                if not numeric.empty:
                    values = numeric.to_numpy(dtype=np.float64)
                    infinite_mask = np.isinf(values)

                    if infinite_mask.any():
                        for index, col in enumerate(numeric.columns):
                            count = int(infinite_mask[:, index].sum())
                            if count:
                                infinite_counts[str(col)] += count

                # Count original labels without modifying them.
                label_col = next(
                    (
                        col for col in chunk.columns
                        if str(col).strip().lower() == "label"
                    ),
                    None,
                )

                if label_col is not None:
                    labels = chunk[label_col].astype("string").str.strip()
                    label_counts.update(
                        labels.fillna("<MISSING LABEL>").tolist()
                    )

            print(f"Rows scanned: {total_rows:,}")
            print(f"Columns with missing values: {dict(missing_counts) or 'None'}")
            print(f"Columns with infinite values: {dict(infinite_counts) or 'None'}")

            print("Labels found:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count:,}")

        except Exception as error:
            print(f"Could not complete the check: {error}")

    print("\nQuality check finished. Original CSV files were not changed.")


if __name__ == "__main__":
    main()
