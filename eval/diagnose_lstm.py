"""Root-cause check: why did the LSTM autoencoder score below random (ROC-AUC
0.44) on the DDoS+PortScan eval set? Breaks reconstruction error down by
true label (BENIGN / DDoS / PortScan) to see which attack type is driving it.
"""
import numpy as np
import pandas as pd

from src.detectors import SequenceAutoencoderDetector
from src.features import make_windows, to_sequence_tensor
from src.ingest import clean, load_csv
from src.schema import FEATURE_COLUMNS


def main():
    benign = clean(load_csv("monday_benign.csv"))
    ddos = clean(load_csv("friday_ddos.csv"))
    portscan = clean(load_csv("friday_portscan.csv"))

    train_seq = to_sequence_tensor(benign, FEATURE_COLUMNS)
    lstm = SequenceAutoencoderDetector(n_features=len(FEATURE_COLUMNS), epochs=8)
    lstm.fit(train_seq)

    for name, df in [("DDoS_day", ddos), ("PortScan_day", portscan)]:
        windows = make_windows(df)
        seq = to_sequence_tensor(df, FEATURE_COLUMNS)
        labels = windows["label"].to_numpy()[: len(seq)]
        scores = lstm.score(seq)
        benign_mask = labels == "BENIGN"
        attack_mask = labels == "ATTACK"
        print(f"--- {name} ---")
        print(f"  benign windows: n={benign_mask.sum()}, mean recon error={scores[benign_mask].mean():.4f}")
        print(f"  attack windows: n={attack_mask.sum()}, mean recon error={scores[attack_mask].mean():.4f}")

    # Also compare raw flow-duration / SYN-flag characteristics by label to
    # see if attack flows are just structurally "simpler" (easier to reconstruct).
    for name, df in [("DDoS", ddos), ("PortScan", portscan)]:
        g = df.groupby("Label")[["Flow Duration", "SYN Flag Count", "Total Fwd Packets"]].mean()
        print(f"--- {name} raw flow stats by label ---")
        print(g)


if __name__ == "__main__":
    main()
