
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned_network_flows.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "binary_network_flows.csv"

CHUNK_SIZE = 50_000


def main():
    if not INPUT_FILE.exists():
        print(f"Input dataset not found: {INPUT_FILE}")
        return

    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Remove or rename it manually before running again.")
        return

    total_rows = 0
    benign_rows = 0
    attack_rows = 0
    first_chunk = True

    try:
        for chunk in pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=True,
            encoding_errors="replace",
        ):
            # Standardize column names.
            chunk.columns = chunk.columns.astype(str).str.strip()

            if "Label" not in chunk.columns:
                raise ValueError("The Label column is missing.")

            # Preserve the original category.
            chunk["Original_Label"] = (
                chunk["Label"].astype("string").str.strip()
            )

            # Exclude rows with missing or empty labels.
            valid = (
                chunk["Original_Label"].notna()
                & chunk["Original_Label"].ne("")
            )
            chunk = chunk.loc[valid].copy()

            # Binary target: 0 = benign, 1 = attack.
            chunk["Target"] = (
                chunk["Original_Label"].str.upper() != "BENIGN"
            ).astype("int8")

            benign_rows += int((chunk["Target"] == 0).sum())
            attack_rows += int((chunk["Target"] == 1).sum())
            total_rows += len(chunk)

            # Remove the old label column; keep Original_Label
            # for later analysis and use Target for model training.
            chunk.drop(columns=["Label"], inplace=True)

            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False

        print("\n" + "=" * 60)
        print("BINARY DATASET CREATED")
        print("=" * 60)
        print(f"Output: {OUTPUT_FILE}")
        print(f"Total rows: {total_rows:,}")
        print(f"Benign rows (0): {benign_rows:,}")
        print(f"Attack rows (1): {attack_rows:,}")
        print("Original_Label preserves the original category.")
        print("Original datasets remain unchanged.")

    except Exception:
        # Avoid leaving a partial output that looks complete.
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        raise


if __name__ == "__main__":
    main()
