"""Validate the registered engine sensor dataset before downstream ML jobs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
    "Engine Condition",
]
FEATURE_COLUMNS = EXPECTED_COLUMNS[:-1]
TARGET = "Engine Condition"


def validate_dataset(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Registered dataset not found: {path}")

    data = pd.read_csv(path)
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names found.")

    missing = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    extra = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing={missing}; unexpected={extra}")
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    for column in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"Expected a numeric column: {column}")
    if not np.isfinite(data[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Infinite or non-finite sensor values found.")

    classes = set(data[TARGET].unique().tolist())
    if classes != {0, 1}:
        raise ValueError(f"Target must contain exactly classes 0 and 1; found {classes}")

    summary = {
        "file": str(path),
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "predictor_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "missing_cells": int(data[EXPECTED_COLUMNS].isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "target_counts": {
            str(label): int(count)
            for label, count in data[TARGET].value_counts().sort_index().items()
        },
    }
    print("DATA VALIDATION: PASSED\n" + json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    validate_dataset(project_root / "data" / "engine_data.csv")
