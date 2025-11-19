# CICIDS2017 Dataset

## Overview

The CICIDS2017 dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs). It includes network traffic analysis using CICFlowMeter with labeled flows based on the time stamp, source and destination IPs, source and destination ports, protocols and attack.

## Dataset Information

- **Source**: Canadian Institute for Cybersecurity (CIC)
- **Year**: 2017
- **Format**: CSV files with extracted flow features
- **Features**: 80+ network flow features
- **Duration**: 5 days of network traffic (Monday - Friday)

## Attack Types Included

- **Brute Force**: FTP-Patator, SSH-Patator
- **DoS/DDoS**: DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, DDoS
- **Web Attacks**: SQL Injection, XSS
- **Infiltration**: Dropbox download, Cool disk
- **Botnet**: Ares
- **Port Scan**: Port scanning activities

## Download Instructions

### Option 1: Direct Download from Official Source

1. Visit the official dataset page:
   https://www.unb.ca/cic/datasets/ids-2017.html

2. Download the CSV files for the days you need:
   - **Monday**: Benign traffic only (recommended for initial training)
   - **Tuesday**: Brute Force attacks
   - **Wednesday**: DoS/DDoS attacks
   - **Thursday**: Web attacks and Infiltration
   - **Friday**: Botnet and Port Scan

3. Extract the CSV files to the `data/raw/` directory

### Option 2: Using wget (if direct links are available)

```bash
# Example - replace with actual download links
cd data/raw/
wget <URL_TO_MONDAY_CSV>
wget <URL_TO_TUESDAY_CSV>
# ... download other days as needed
```

### Option 3: Kaggle

The dataset is also available on Kaggle:
https://www.kaggle.com/datasets/cicdataset/cicids2017

```bash
# Using Kaggle CLI
kaggle datasets download -d cicdataset/cicids2017
unzip cicids2017.zip -d data/raw/
```

## Directory Structure

After downloading, your data directory should look like:

```
data/
├── raw/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── processed/
│   └── (processed files will be saved here)
└── README.md (this file)
```

## Initial Setup - Start with Monday Data

For initial model development and prototyping, we recommend starting with Monday's data:
- Contains only benign traffic
- Useful for training anomaly detection models (normal behavior baseline)
- Smaller file size for faster iteration

## Feature Information

The dataset contains 80+ features including:

**Flow-based features:**
- Duration, packets, bytes
- Packet length statistics (mean, std, min, max)
- Flow IAT (Inter-Arrival Time) statistics
- Flags count (FIN, SYN, RST, PSH, ACK, URG)

**Advanced features:**
- Active/Idle time statistics
- Subflow information
- Header lengths
- Fwd/Bwd segment sizes

## Data Preprocessing Notes

Common preprocessing steps:

1. **Handle missing values**: Some features may contain NaN or Inf values
2. **Remove duplicates**: Check for duplicate flows
3. **Handle infinite values**: Replace Inf/-Inf with appropriate values
4. **Normalize features**: Most ML algorithms benefit from normalized features
5. **Balance classes**: For supervised approaches, consider class imbalance
6. **Feature selection**: 80+ features may include redundant information

## Label Information

- **Benign**: Normal network traffic
- **Attack labels**: Specific attack type names (e.g., "DDoS", "PortScan", "Bot")

For anomaly detection:
- Benign = Normal (label 0)
- All attacks = Anomaly (label 1)

## File Size Estimates

- Monday (Benign): ~470 MB
- Tuesday - Friday (with attacks): 500 MB - 2 GB each
- Total dataset: ~7-8 GB

## Citation

If you use this dataset, please cite:

```
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani,
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization",
4th International Conference on Information Systems Security and Privacy (ICISSP),
Portugal, January 2018
```

## Additional Resources

- Dataset paper: https://www.scitepress.org/Papers/2018/66398/66398.pdf
- CICFlowMeter: https://github.com/ahlashkari/CICFlowMeter
- Dataset statistics: See notebook `01_eda.ipynb` after download

## Troubleshooting

**Issue**: File too large to load in memory
**Solution**: Use chunked reading with pandas or process day-by-day

**Issue**: Column encoding errors
**Solution**: Use `encoding='utf-8'` or `encoding='latin1'` when reading CSV

**Issue**: Infinite values in features
**Solution**: Replace with `np.nan` then handle missing values appropriately
