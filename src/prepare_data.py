"""Create stratified train/test CSVs and a portable workflow hand-off archive."""
from pathlib import Path
import json
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from validate_data import FEATURE_COLUMNS, TARGET, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "engine_data.csv"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def prepare() -> Path:
    validate_dataset(DATA)
    frame = pd.read_csv(DATA)

    train_frame, test_frame = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET],
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train_path = ARTIFACTS / "train.csv"
    test_path = ARTIFACTS / "test.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    metadata = {
        "source_file": "data/engine_data.csv",
        "rows_total": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "features": FEATURE_COLUMNS,
        "target": TARGET,
    }
    metadata_path = ARTIFACTS / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    bundle_path = ARTIFACTS / "train_test_workflow_artifact.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(train_path, arcname="train.csv")
        archive.write(test_path, arcname="test.csv")
        archive.write(metadata_path, arcname="split_metadata.json")

    # Verify the archive itself, not only the local CSVs.
    with zipfile.ZipFile(bundle_path) as archive:
        train_check = pd.read_csv(archive.open("train.csv"))
        test_check = pd.read_csv(archive.open("test.csv"))
        split_check = json.load(archive.open("split_metadata.json"))

    assert len(train_check) + len(test_check) == len(frame)
    assert len(train_check) == metadata["train_rows"]
    assert len(test_check) == metadata["test_rows"]
    assert split_check["stratified"] is True
    train_share = train_check[TARGET].value_counts(normalize=True).sort_index()
    test_share = test_check[TARGET].value_counts(normalize=True).sort_index()
    assert all(abs(float(train_share.loc[label]) - float(test_share.loc[label])) < 0.005
               for label in [0, 1])

    print(f"Train rows: {len(train_check):,}; test rows: {len(test_check):,}")
    print(f"Workflow artifact verified: {bundle_path}")
    return bundle_path


if __name__ == "__main__":
    prepare()
