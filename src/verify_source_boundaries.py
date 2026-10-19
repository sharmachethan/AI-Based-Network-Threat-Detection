
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT / "MachineLearningCSV" / "MachineLearningCVE"
COMBINED_CSV = ROOT / "data" / "binary_network_flows.csv"

source_files = sorted(SOURCE_DIR.glob("*.csv"))

print("=" * 60)
print("SOURCE BOUNDARY VERIFICATION")
print("=" * 60)

expected_total = 0

for file_path in source_files:
    rows = sum(
        len(chunk)
        for chunk in pd.read_csv(
            file_path,
            usecols=[0],
            chunksize=100_000,
            encoding_errors="replace"
        )
    )

    expected_total += rows
    print(f"{file_path.name}: {rows:,} rows")

combined_rows = sum(
    len(chunk)
    for chunk in pd.read_csv(
        COMBINED_CSV,
        usecols=["Target"],
        chunksize=100_000
    )
)

print("-" * 60)
print(f"Total source rows:   {expected_total:,}")
print(f"Combined CSV rows:   {combined_rows:,}")
print(f"Row counts match:    {expected_total == combined_rows}")

if expected_total != combined_rows:
    print("WARNING: Investigate the row-count mismatch.")
else:
    print(
        "Row counts match. This verifies the total row count, "
        "but not the ordering of rows."
    )
