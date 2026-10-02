"""Shared configuration for the EDF capstone pipelines."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "entraineddropletfraction_VerticalFlow.csv"
DATA_PATH = Path(os.environ.get("EDF_DATA_PATH", DEFAULT_DATA))

FEATURE_COLS = ["D", "Usg", "UsL", "rhol", "rhog", "st", "MuL", "Mug"]
TARGET_COL = "e"
SEED = 42
N_SPLITS = 5


def load_dataset():
    import pandas as pd

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place the Aliyu et al. (2017) CSV there "
            "or set the EDF_DATA_PATH environment variable."
        )
    return pd.read_csv(DATA_PATH)
