import numpy as np
import pandas as pd

from src.features import WINDOW_SIZE, make_windows, to_sequence_tensor
from src.schema import FEATURE_COLUMNS, WINDOW_FEATURE_COLUMNS


def _fake_flows(n: int, label: str = "BENIGN", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 100, size=n) for col in FEATURE_COLUMNS}
    data["Destination Port"] = rng.integers(1, 5, size=n)
    data["SYN Flag Count"] = rng.integers(0, 2, size=n)
    data["RST Flag Count"] = rng.integers(0, 2, size=n)
    data["ACK Flag Count"] = rng.integers(0, 2, size=n)
    data["FIN Flag Count"] = rng.integers(0, 2, size=n)
    df = pd.DataFrame(data)
    df["Label"] = label
    return df


def test_make_windows_shape_and_columns():
    df = _fake_flows(100)
    windows = make_windows(df, window_size=20, step=10)
    assert len(windows) == (100 - 20) // 10 + 1
    for col in WINDOW_FEATURE_COLUMNS:
        assert col in windows.columns


def test_make_windows_labels_attack_if_any_flow_is_attack():
    df = _fake_flows(20, label="BENIGN")
    df.loc[5, "Label"] = "DDoS"
    windows = make_windows(df, window_size=20, step=20)
    assert len(windows) == 1
    assert windows.iloc[0]["label"] == "ATTACK"


def test_make_windows_all_benign_labeled_benign():
    df = _fake_flows(20, label="BENIGN")
    windows = make_windows(df, window_size=20, step=20)
    assert windows.iloc[0]["label"] == "BENIGN"


def test_unique_ports_touched_reflects_scan_breadth():
    df = _fake_flows(20)
    df["Destination Port"] = 80  # all same port -> no scan signature
    windows = make_windows(df, window_size=20, step=20)
    assert windows.iloc[0]["unique_ports_touched"] == 1

    df2 = _fake_flows(20)
    df2["Destination Port"] = range(20)  # every flow a different port -> scan signature
    windows2 = make_windows(df2, window_size=20, step=20)
    assert windows2.iloc[0]["unique_ports_touched"] == 20


def test_to_sequence_tensor_shape():
    df = _fake_flows(50)
    tensor = to_sequence_tensor(df, FEATURE_COLUMNS, window_size=20, step=10)
    assert tensor.shape == (4, 20, len(FEATURE_COLUMNS))
