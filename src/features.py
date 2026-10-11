"""Rolling-window behavioral feature engineering over the flow stream.

Each window summarizes WINDOW_SIZE consecutive flows (in arrival order) into
one row of aggregate behavioral features (src/schema.py:WINDOW_FEATURE_COLUMNS).
A window is labeled "anomalous" if any flow inside it carries a non-BENIGN
label — a window-level weak label, which is how a real streaming detector
would actually be evaluated (it flags a time slice, not a single packet).
"""
import numpy as np
import pandas as pd

WINDOW_SIZE = 20
STEP = 10  # 50% overlap between consecutive windows


def make_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE, step: int = STEP) -> pd.DataFrame:
    n = len(df)
    rows = []
    for start in range(0, n - window_size + 1, step):
        chunk = df.iloc[start:start + window_size]
        rows.append(_summarize_window(chunk))
    return pd.DataFrame(rows)


def _summarize_window(chunk: pd.DataFrame) -> dict:
    syn = chunk["SYN Flag Count"]
    rst = chunk["RST Flag Count"]
    total_flags = (syn + rst + chunk["ACK Flag Count"] + chunk["FIN Flag Count"]).clip(lower=1)
    is_attack = (chunk["Label"] != "BENIGN").any()
    return {
        "flow_count": len(chunk),
        "bytes_per_sec_mean": chunk["Flow Bytes/s"].mean(),
        "packets_per_sec_mean": chunk["Flow Packets/s"].mean(),
        "flow_duration_mean": chunk["Flow Duration"].mean(),
        "iat_mean": chunk["Flow IAT Mean"].mean(),
        "iat_std": chunk["Flow IAT Std"].mean(),
        "syn_ratio": (syn / total_flags).mean(),
        "rst_ratio": (rst / total_flags).mean(),
        "unique_ports_touched": chunk["Destination Port"].nunique(),
        "avg_packet_size_mean": chunk["Average Packet Size"].mean(),
        "label": "ATTACK" if is_attack else "BENIGN",
    }


def to_sequence_tensor(df: pd.DataFrame, feature_cols: list, window_size: int = WINDOW_SIZE, step: int = STEP) -> np.ndarray:
    """Return raw (unaggregated) per-flow feature sequences, shape (n_windows, window_size, n_features).

    Used by the LSTM autoencoder, which learns from the flow-by-flow pattern
    within a window rather than a single aggregated summary row.
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    n = len(df)
    sequences = []
    for start in range(0, n - window_size + 1, step):
        sequences.append(values[start:start + window_size])
    return np.stack(sequences) if sequences else np.empty((0, window_size, len(feature_cols)))
