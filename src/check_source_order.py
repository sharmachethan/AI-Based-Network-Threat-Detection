
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "binary_network_flows.csv"

files = [
    ("Friday DDoS", 225745),
    ("Friday PortScan", 286467),
    ("Friday Morning", 191033),
    ("Monday", 529918),
    ("Thursday Infiltration", 288602),
    ("Thursday Web Attacks", 170366),
    ("Tuesday", 445909),
    ("Wednesday", 692703),
]

boundaries = []
position = 0

for name, rows in files:
    boundaries.append((name, position, position + rows))
    position += rows

wanted = set()

for _, start, end in boundaries:
    for idx in range(max(0, start - 3), min(position, end + 3)):
        wanted.add(idx)

found = {}
row_position = 0

for chunk in pd.read_csv(
    CSV_PATH,
    usecols=["Original_Label"],
    chunksize=100_000
):
    end_position = row_position + len(chunk)

    relevant = [
        idx for idx in wanted
        if row_position <= idx < end_position
    ]

    for idx in relevant:
        found[idx] = str(
            chunk.iloc[idx - row_position]["Original_Label"]
        ).strip()

    row_position = end_position

print("=" * 65)
print("LABELS AROUND SOURCE-FILE BOUNDARIES")
print("=" * 65)

for name, start, end in boundaries:
    print(f"\n{name}: rows {start:,} to {end - 1:,}")

    for idx in range(max(0, start - 2), min(position, end + 2)):
        print(f"  Row {idx:,}: {found.get(idx, 'NOT FOUND')}")

print("\nCheck complete.")
