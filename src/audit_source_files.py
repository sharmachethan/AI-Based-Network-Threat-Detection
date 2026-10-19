
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "MachineLearningCSV" / "MachineLearningCVE"

print("=" * 65)
print("SOURCE FILE AUDIT")
print("=" * 65)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in: {DATA_DIR}")
    raise SystemExit(1)

for file_path in csv_files:
    print(f"\nFile: {file_path.name}")

    total_rows = 0
    label_counts = {}

    for chunk in pd.read_csv(
        file_path,
        usecols=[" Label"],
        chunksize=100_000,
        encoding_errors="replace"
    ):
        labels = chunk.iloc[:, 0].astype(str).str.strip()
        total_rows += len(chunk)

        counts = labels.value_counts()
        for label, count in counts.items():
            label_counts[label] = (
                label_counts.get(label, 0) + int(count)
            )

    print(f"Rows: {total_rows:,}")
    print("Labels:")

    for label, count in sorted(
        label_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"  {label}: {count:,}")

print("\nSource file audit complete.")
