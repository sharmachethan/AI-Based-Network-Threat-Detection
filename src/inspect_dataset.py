
from pathlib import Path
import pandas as pd

# Project root: NetworkThreatDetection
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Search the extracted dataset folder and data folder.
SEARCH_FOLDERS = [
    PROJECT_ROOT / "MachineLearningCSV",
    PROJECT_ROOT / "data",
]

def main():
    csv_files = []

    for folder in SEARCH_FOLDERS:
        if folder.exists():
            csv_files.extend(folder.rglob("*.csv"))

    # Avoid duplicate paths and sort the results.
    csv_files = sorted(set(csv_files))

    print("=" * 65)
    print("NETWORK THREAT DETECTION - DATASET INSPECTION")
    print("=" * 65)

    if not csv_files:
        print("No CSV files found.")
        print("Check that the extracted MachineLearningCSV folder")
        print("is inside the NetworkThreatDetection project folder.")
        return

    print(f"\nCSV files found: {len(csv_files)}")

    for index, csv_file in enumerate(csv_files, start=1):
        size_mb = csv_file.stat().st_size / (1024 * 1024)

        print("\n" + "-" * 65)
        print(f"File {index}: {csv_file.name}")
        print(f"Size: {size_mb:.2f} MB")

        try:
            # Read only five rows; do not load the entire file.
            try:
                sample = pd.read_csv(csv_file, nrows=5)
            except UnicodeDecodeError:
                sample = pd.read_csv(
                    csv_file, nrows=5, encoding="latin1"
                )

            # Remove accidental whitespace from column names for display.
            columns = [str(col).strip() for col in sample.columns]

            print(f"Number of columns: {len(columns)}")
            print("First five column names:")
            for column in columns[:5]:
                print(f"  - {column}")

            if len(columns) > 5:
                print(f"  ... and {len(columns) - 5} more columns")

            label_columns = [
                col for col in columns
                if col.lower() in {"label", "class", "attack"}
            ]

            if label_columns:
                print(f"Possible target column(s): {label_columns}")
            else:
                print("Target column not identified automatically.")

        except Exception as error:
            print(f"Could not inspect this file: {error}")

    print("\n" + "=" * 65)
    print("Inspection complete.")
    print("=" * 65)

if __name__ == "__main__":
    main()
