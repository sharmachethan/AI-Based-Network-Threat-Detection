
from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "MachineLearningCSV"
CHUNK_SIZE = 100_000


def main():
    csv_files = sorted(DATASET_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {DATASET_DIR}")
        print("Check the dataset folder location.")
        return

    overall_counts = Counter()
    total_rows = 0

    print("=" * 65)
    print("CIC-IDS2017 LABEL SUMMARY")
    print("=" * 65)

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        file_counts = Counter()
        file_rows = 0

        try:
            # Read only the header first to find the actual Label column.
            header = pd.read_csv(csv_file, nrows=0)
            label_columns = [
                col for col in header.columns
                if str(col).strip().lower() == "label"
            ]

            if len(label_columns) != 1:
                print(f"Skipped: expected one Label column, found {label_columns}")
                continue

            label_column = label_columns[0]

            # Read only the label column in chunks.
            for chunk in pd.read_csv(
                csv_file,
                usecols=[label_column],
                chunksize=CHUNK_SIZE,
                low_memory=True,
            ):
                labels = chunk[label_column].astype("string").str.strip()
                labels = labels.fillna("<MISSING>")
                labels = labels.replace("", "<BLANK>")

                counts = labels.value_counts(dropna=False)
                file_counts.update(counts.to_dict())

                file_rows += len(chunk)

            print(f"Rows: {file_rows:,}")
            for label, count in sorted(file_counts.items()):
                print(f"  {label}: {count:,}")

            overall_counts.update(file_counts)
            total_rows += file_rows

        except Exception as error:
            print(f"Error processing file: {error}")

    print("\n" + "=" * 65)
    print(f"TOTAL ROWS PROCESSED: {total_rows:,}")
    print("COMBINED LABEL COUNTS:")
    for label, count in sorted(overall_counts.items()):
        print(f"  {label}: {count:,}")

    print("\nLabel inspection complete.")


if __name__ == "__main__":
    main()
