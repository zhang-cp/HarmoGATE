
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_random(frame, seed):
    train, remainder = train_test_split(frame, test_size=0.2, random_state=seed, shuffle=True)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed, shuffle=True)
    return train, validation, test


def split_by_entity(frame, column, seed):
    entities = np.asarray(sorted(frame[column].unique()))
    shuffled = np.random.default_rng(seed).permutation(entities)
    train_end = int(np.floor(0.8 * len(shuffled)))
    validation_end = train_end + int(np.floor(0.1 * len(shuffled)))
    groups = [
        set(shuffled[:train_end]),
        set(shuffled[train_end:validation_end]),
        set(shuffled[validation_end:]),
    ]
    return tuple(frame[frame[column].isin(group)] for group in groups)


def save(parts, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "test"), parts):
        part = part.reset_index(drop=True).copy()
        part.insert(0, "id", range(len(part)))
        part.to_csv(output / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--protein-column", default="protein_sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    columns = [args.smiles_column, args.protein_column, args.label_column]
    frame = pd.read_csv(args.input_csv, usecols=columns).dropna().copy()
    frame.columns = ["smiles", "protein_sequence", "label"]
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame["protein_sequence"] = frame["protein_sequence"].astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna().drop_duplicates(["smiles", "protein_sequence"]).reset_index(drop=True)

    save(split_random(frame, args.seed), args.output_root / "random")
    save(split_by_entity(frame, "smiles", args.seed), args.output_root / "cold_drug")
    save(split_by_entity(frame, "protein_sequence", args.seed), args.output_root / "cold_target")


if __name__ == "__main__":
    main()
