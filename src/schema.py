"""Column definitions for the CICIDS2017 flow-feature CSVs used by this project.

The CSVs (Monday-WorkingHours, Friday-Afternoon-DDos, Friday-Afternoon-PortScan)
are the pre-extracted, ML-ready CICFlowMeter output published by the Canadian
Institute for Cybersecurity. They contain per-flow statistical features only —
there is NO source/destination IP, no Flow ID, and no wall-clock timestamp in
this cleaned export. That shapes what "behavioral" features are honestly
buildable here: we treat row order within a file as flow-arrival order (the
original capture order, since no explicit timestamp survives in this cleaned
export) and compute rolling windows of consecutive flows over that stream,
rather than per-source-IP sessions. A window's "unique destination ports
touched" is the closest honest proxy for scan breadth available in this data.
"""

RAW_LABEL_COL = "Label"

# Numeric flow-feature columns used for modeling (subset of the ~78 CICFlowMeter
# columns; excludes the label and columns that are constant/near-constant in
# our sampled days).
FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

# Windowed/aggregate behavioral features computed per Destination-Port bucket
# (see src/features.py). These are what the detectors actually consume.
WINDOW_FEATURE_COLUMNS = [
    "flow_count",                 # request/connection volume in the window
    "bytes_per_sec_mean",         # traffic volume
    "packets_per_sec_mean",       # traffic volume
    "flow_duration_mean",
    "iat_mean",                   # temporal spacing between flows
    "iat_std",                    # temporal deviation
    "syn_ratio",                  # SYN flag share -> scan/flood signature
    "rst_ratio",                  # RST flag share -> scan/reset signature
    "unique_ports_touched",       # port-scan signature (breadth)
    "avg_packet_size_mean",
]
