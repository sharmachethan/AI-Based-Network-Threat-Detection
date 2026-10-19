
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

train_path = DATA_DIR / "train_indices.npy"
test_path = DATA_DIR / "test_indices.npy"

train_indices = np.load(train_path)
test_indices = np.load(test_path)

print("=" * 55)
print("TRAIN/TEST SPLIT AUDIT")
print("=" * 55)

print(f"Training rows: {len(train_indices):,}")
print(f"Testing rows:  {len(test_indices):,}")

print(f"Training index range: {train_indices.min():,} to {train_indices.max():,}")
print(f"Testing index range:  {test_indices.min():,} to {test_indices.max():,}")

overlap = np.intersect1d(train_indices, test_indices)

print(f"Overlapping indices: {len(overlap):,}")
print(f"Total split rows: {len(train_indices) + len(test_indices):,}")

print("\nChecking duplicate indices...")

print(f"Duplicate training indices: "
      f"{len(train_indices) - len(np.unique(train_indices)):,}")

print(f"Duplicate testing indices: "
      f"{len(test_indices) - len(np.unique(test_indices)):,}")

print("\nChecking target distribution in each split...")

csv_path = DATA_DIR / "binary_network_flows.csv"

train_set = set(train_indices.tolist())
test_set = set(test_indices.tolist())

train_counts = {0: 0, 1: 0}
test_counts = {0: 0, 1: 0}

row_position = 0

for chunk in pd.read_csv(
    csv_path,
    usecols=["Target"],
    chunksize=100_000
):
    targets = chunk["Target"].to_numpy()

    for value in (0, 1):
        positions = np.arange(row_position, row_position + len(chunk))
        train_counts[value] += int(
            np.isin(positions, train_indices[train_indices < row_position + len(chunk)]).sum()
        ) if False else 0

    # Count targets using the saved indices that fall in this chunk.
    chunk_end = row_position + len(chunk)

    train_local = train_indices[
        (train_indices >= row_position) & (train_indices < chunk_end)
    ] - row_position

    test_local = test_indices[
        (test_indices >= row_position) & (test_indices < chunk_end)
    ] - row_position

    for value in (0, 1):
        train_counts[value] += int((targets[train_local] == value).sum())
        test_counts[value] += int((targets[test_local] == value).sum())

    row_position = chunk_end

print(f"\nRows scanned: {row_position:,}")
print(f"Training benign (0): {train_counts[0]:,}")
print(f"Training attack (1): {train_counts[1]:,}")
print(f"Testing benign (0):  {test_counts[0]:,}")
print(f"Testing attack (1):  {test_counts[1]:,}")

print("\nAudit complete.")
