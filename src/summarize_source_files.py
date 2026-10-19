
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "MachineLearningCSV" / "MachineLearningCVE"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

files = sorted(SOURCE_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {SOURCE_DIR}")

records = []

for file_path in files:
    total = 0
    benign = 0
    attack = 0
    labels = set()

    for chunk in pd.read_csv(
        file_path,
        usecols=lambda column: column.strip() == "Label",
        chunksize=100_000,
        encoding_errors="replace"
    ):
        label_col = chunk.iloc[:, 0].astype(str).str.strip()

        total += len(label_col)
        benign += int((label_col == "BENIGN").sum())
        attack += int((label_col != "BENIGN").sum())
        labels.update(label_col.unique())

    records.append({
        "source_file": file_path.name,
        "total_rows": total,
        "benign_rows": benign,
        "attack_rows": attack,
        "number_of_labels": len(labels),
        "labels": " | ".join(sorted(labels)),
    })

summary = pd.DataFrame(records)

output_path = REPORT_DIR / "source_file_summary.csv"
summary.to_csv(output_path, index=False)

print(summary[[
    "source_file",
    "total_rows",
    "benign_rows",
    "attack_rows",
    "number_of_labels"
]].to_string(index=False))

print(f"\nTotal rows: {summary['total_rows'].sum():,}")
print(f"Summary saved to: {output_path}")
