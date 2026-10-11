"""Load and lightly clean the CICIDS2017 flow-feature CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import FEATURE_COLUMNS, RAW_LABEL_COL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv(filename: str) -> pd.DataFrame:
    """Load one CICIDS2017 CSV and normalize its column names.

    The published CSVs have inconsistent leading spaces in header names
    (e.g. " Flow Duration") — this strips them so column lookups are stable.
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the modeling columns + label, drop inf/NaN rows.

    Flow Bytes/s and Flow Packets/s can be inf when Flow Duration is 0
    (a zero-length flow) — CICFlowMeter's own known quirk. Dropping those
    rows rather than imputing them keeps the training signal honest.
    """
    cols = FEATURE_COLUMNS + [RAW_LABEL_COL]
    df = df[cols].copy()
    numeric_cols = [c for c in FEATURE_COLUMNS]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].mask(np.isinf(df[numeric_cols]))
    df = df.dropna()
    df[RAW_LABEL_COL] = df[RAW_LABEL_COL].str.strip()
    return df.reset_index(drop=True)


def load_all() -> dict:
    """Load the three source files as separate, cleaned DataFrames.

    Kept separate (not concatenated) because each file has its own
    arrival-order semantics used for windowing in src/features.py.
    """
    return {
        "monday_benign": clean(load_csv("monday_benign.csv")),
        "friday_ddos": clean(load_csv("friday_ddos.csv")),
        "friday_portscan": clean(load_csv("friday_portscan.csv")),
    }
