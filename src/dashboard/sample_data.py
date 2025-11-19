"""
Embedded Sample Network Flows for Dashboard Testing

This module contains real network flow samples from the CICIDS2017 dataset,
embedded directly in the code to eliminate file dependencies in production.

Samples:
- SAMPLE_BENIGN_FLOW: Normal HTTP traffic (38 seconds duration)
- SAMPLE_DOS_ATTACK: Real DoS Slowloris attack (86+ minutes duration)

Source: CIC IDS 2017 Wednesday dataset
"""

# =============================================================================
# SAMPLE 1: BENIGN TRAFFIC
# =============================================================================
# Description: Normal HTTP connection
# Duration: 38.3 seconds
# Packets: 1 forward, 1 backward (minimal exchange)
# Characteristics: Low packet count, symmetrical, normal port 80

SAMPLE_BENIGN_FLOW = {
    " Source Port": 49459.0,
    " Destination Port": 80.0,
    " Protocol": 6.0,
    " Flow Duration": 38308.0,
    " Total Fwd Packets": 1.0,
    " Total Backward Packets": 1.0,
    "Total Length of Fwd Packets": 6.0,
    " Total Length of Bwd Packets": 6.0,
    " Fwd Packet Length Max": 6.0,
    " Fwd Packet Length Min": 6.0,
    " Fwd Packet Length Mean": 6.0,
    " Fwd Packet Length Std": 0.0,
    "Bwd Packet Length Max": 6.0,
    " Bwd Packet Length Min": 6.0,
    " Bwd Packet Length Mean": 6.0,
    " Bwd Packet Length Std": 0.0,
    "Flow Bytes/s": 313.250496,
    " Flow Packets/s": 52.208416,
    " Flow IAT Mean": 38308.0,
    " Flow IAT Std": 0.0,
    " Flow IAT Max": 38308.0,
    " Flow IAT Min": 38308.0,
    "Fwd IAT Total": 0.0,
    " Fwd IAT Mean": 0.0,
    " Fwd IAT Std": 0.0,
    " Fwd IAT Max": 0.0,
    " Fwd IAT Min": 0.0,
    "Bwd IAT Total": 0.0,
    " Bwd IAT Mean": 0.0,
    " Bwd IAT Std": 0.0,
    " Bwd IAT Max": 0.0,
    " Bwd IAT Min": 0.0,
    "Fwd PSH Flags": 0.0,
    " Bwd PSH Flags": 0.0,
    " Fwd URG Flags": 0.0,
    " Bwd URG Flags": 0.0,
    " Fwd Header Length": 20.0,
    " Bwd Header Length": 20.0,
    "Fwd Packets/s": 26.104208,
    " Bwd Packets/s": 26.104208,
    " Min Packet Length": 6.0,
    " Max Packet Length": 6.0,
    " Packet Length Mean": 6.0,
    " Packet Length Std": 0.0,
    " Packet Length Variance": 0.0,
    "FIN Flag Count": 0.0,
    " SYN Flag Count": 0.0,
    " RST Flag Count": 0.0,
    " PSH Flag Count": 0.0,
    " ACK Flag Count": 1.0,
    " URG Flag Count": 1.0,
    " CWE Flag Count": 0.0,
    " ECE Flag Count": 0.0,
    " Down/Up Ratio": 1.0,
    " Average Packet Size": 9.0,
    " Avg Fwd Segment Size": 6.0,
    " Avg Bwd Segment Size": 6.0,
    " Fwd Header Length.1": 20.0,
    "Fwd Avg Bytes/Bulk": 0.0,
    " Fwd Avg Packets/Bulk": 0.0,
    " Fwd Avg Bulk Rate": 0.0,
    " Bwd Avg Bytes/Bulk": 0.0,
    " Bwd Avg Packets/Bulk": 0.0,
    "Bwd Avg Bulk Rate": 0.0,
    "Subflow Fwd Packets": 1.0,
    " Subflow Fwd Bytes": 6.0,
    " Subflow Bwd Packets": 1.0,
    " Subflow Bwd Bytes": 6.0,
    "Init_Win_bytes_forward": 255.0,
    " Init_Win_bytes_backward": 946.0,
    " act_data_pkt_fwd": 0.0,
    " min_seg_size_forward": 20.0,
    "Active Mean": 0.0,
    " Active Std": 0.0,
    " Active Max": 0.0,
    " Active Min": 0.0,
    "Idle Mean": 0.0,
    " Idle Std": 0.0,
    " Idle Max": 0.0,
    " Idle Min": 0.0,
}

# =============================================================================
# SAMPLE 2: DOS SLOWLORIS ATTACK
# =============================================================================
# Description: Real DoS Slowloris slow HTTP attack
# Duration: 5,169.9 seconds (86+ minutes!)
# Packets: 8 forward, 6 backward
# Characteristics: Extremely long flow, variable packet sizes, anomalous timing
# Attack Type: Slow-read/slow-write targeting HTTP

SAMPLE_DOS_ATTACK = {
    " Source Port": 49631.0,
    " Destination Port": 80.0,
    " Protocol": 6.0,
    " Flow Duration": 5169956.0,
    " Total Fwd Packets": 8.0,
    " Total Backward Packets": 6.0,
    "Total Length of Fwd Packets": 1101.0,
    " Total Length of Bwd Packets": 4222.0,
    " Fwd Packet Length Max": 410.0,
    " Fwd Packet Length Min": 0.0,
    " Fwd Packet Length Mean": 137.625,
    " Fwd Packet Length Std": 185.75862790000002,
    "Bwd Packet Length Max": 3525.0,
    " Bwd Packet Length Min": 0.0,
    " Bwd Packet Length Mean": 703.6666667000001,
    " Bwd Packet Length Std": 1395.868284,
    "Flow Bytes/s": 1029.6025730000001,
    " Flow Packets/s": 2.707953414,
    " Flow IAT Mean": 397688.9231,
    " Flow IAT Std": 1368409.299,
    " Flow IAT Max": 4951173.0,
    " Flow IAT Min": 112.0,
    "Fwd IAT Total": 218783.0,
    " Fwd IAT Mean": 31254.714289999996,
    " Fwd IAT Std": 30686.293960000003,
    " Fwd IAT Max": 78311.0,
    " Fwd IAT Min": 219.0,
    "Bwd IAT Total": 5169833.0,
    " Bwd IAT Mean": 1033966.6,
    " Bwd IAT Std": 2218975.06,
    " Bwd IAT Max": 5002421.0,
    " Bwd IAT Min": 840.0,
    "Fwd PSH Flags": 0.0,
    " Bwd PSH Flags": 0.0,
    " Fwd URG Flags": 0.0,
    " Bwd URG Flags": 0.0,
    " Fwd Header Length": 172.0,
    " Bwd Header Length": 132.0,
    "Fwd Packets/s": 1.5474019509999999,
    " Bwd Packets/s": 1.160551463,
    " Min Packet Length": 0.0,
    " Max Packet Length": 3525.0,
    " Packet Length Mean": 354.8666667,
    " Packet Length Std": 895.1151456,
    " Packet Length Variance": 801231.1238,
    "FIN Flag Count": 0.0,
    " SYN Flag Count": 0.0,
    " RST Flag Count": 0.0,
    " PSH Flag Count": 1.0,
    " ACK Flag Count": 0.0,
    " URG Flag Count": 0.0,
    " CWE Flag Count": 0.0,
    " ECE Flag Count": 0.0,
    " Down/Up Ratio": 0.0,
    " Average Packet Size": 380.2142857,
    " Avg Fwd Segment Size": 137.625,
    " Avg Bwd Segment Size": 703.6666667000001,
    " Fwd Header Length.1": 172.0,
    "Fwd Avg Bytes/Bulk": 0.0,
    " Fwd Avg Packets/Bulk": 0.0,
    " Fwd Avg Bulk Rate": 0.0,
    " Bwd Avg Bytes/Bulk": 0.0,
    " Bwd Avg Packets/Bulk": 0.0,
    "Bwd Avg Bulk Rate": 0.0,
    "Subflow Fwd Packets": 8.0,
    " Subflow Fwd Bytes": 1101.0,
    " Subflow Bwd Packets": 6.0,
    " Subflow Bwd Bytes": 4222.0,
    "Init_Win_bytes_forward": 8192.0,
    " Init_Win_bytes_backward": 254.0,
    " act_data_pkt_fwd": 7.0,
    " min_seg_size_forward": 20.0,
    "Active Mean": 0.0,
    " Active Std": 0.0,
    " Active Max": 0.0,
    " Active Min": 0.0,
    "Idle Mean": 0.0,
    " Idle Std": 0.0,
    " Idle Max": 0.0,
    " Idle Min": 0.0,
}

# =============================================================================
# FEATURE NAMES LIST (80 features in correct order)
# =============================================================================
FEATURE_NAMES = [
    " Source Port",
    " Destination Port",
    " Protocol",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Total",
    " Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Total",
    " Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    "Fwd PSH Flags",
    " Bwd PSH Flags",
    " Fwd URG Flags",
    " Bwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    "Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    "FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Packets",
    " Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " act_data_pkt_fwd",
    " min_seg_size_forward",
    "Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
    "Idle Mean",
    " Idle Std",
    " Idle Max",
    " Idle Min",
]

# =============================================================================
# SAMPLE DESCRIPTIONS
# =============================================================================
BENIGN_DESCRIPTION = "Normal HTTP traffic - minimal packet exchange, short duration (38 seconds)"
DOS_DESCRIPTION = "DoS Slowloris attack - 86+ minute flow, anomalous timing patterns (avg 17-minute gaps between packets)"

# =============================================================================
# KEY DIFFERENCES FOR DETECTION
# =============================================================================
"""
Comparison showing why the DoS attack is detectable:

| Metric                | Benign        | DoS Attack     | Ratio     |
|-----------------------|---------------|----------------|-----------|
| Flow Duration         | 38,308 ms     | 5,169,956 ms   | 135x      |
| Total Packets         | 2             | 14             | 7x        |
| Bwd IAT Mean          | 0 ms          | 1,033,966 ms   | infinite  |
| Packet Variance       | 0             | 801,231        | infinite  |
| Flow IAT Max          | 38,308 ms     | 4,951,173 ms   | 129x      |

The DoS Slowloris attack is characterized by:
- Extremely long connection duration (86 minutes vs typical 38 seconds)
- Huge inter-arrival time gaps (17+ minutes between backward packets)
- High packet length variance (some 0 bytes, some 3525 bytes)
- Slow data transmission rate (1029 bytes/s vs typical traffic)
"""
