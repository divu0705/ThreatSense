"""
download_data.py
Downloads the NSL-KDD train/test CSVs (no header row, 41 features + label)
from a public GitHub mirror and saves them locally with proper column names.

Run once:  python src/download_data.py
"""

import os
import pandas as pd
import requests

from constants import COLUMN_NAMES

RAW_BASE = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _download(filename: str) -> str:
    url = f"{RAW_BASE}/{filename}"
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest):
        print(f"Already have {filename}, skipping download.")
        return dest
    print(f"Downloading {filename} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {dest}")
    return dest


def load_dataset(filename: str) -> pd.DataFrame:
    path = _download(filename)
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df = load_dataset("NSL_KDD_Train.csv")
    test_df = load_dataset("NSL_KDD_Test.csv")
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"\nLabel distribution (train):\n{train_df['label'].value_counts().head(10)}")
