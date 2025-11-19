# CICIDS2017 Dataset Download Instructions

## Quick Start - Recommended Approach

The CICIDS2017 dataset requires manual download due to size and authentication requirements.

### Option 1: Kaggle (Recommended - Fastest)

1. **Create Kaggle Account** (if you don't have one):
   - Go to https://www.kaggle.com and sign up

2. **Download the Dataset**:

   **Latest Upload (December 2024):**
   - Visit: https://www.kaggle.com/datasets/naveengill/cicids2017-dataset
   - Click "Download" button (requires login)
   - Extract the ZIP file

   **Alternative - Preprocessed Version (January 2025):**
   - Visit: https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
   - Already cleaned and ready for ML
   - Click "Download" and extract

3. **Place Files in Project**:
   ```bash
   # After downloading and extracting, copy the Monday file:
   cp ~/Downloads/Monday-WorkingHours.pcap_ISCX.csv \
      /Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/data/raw/
   ```

### Option 2: Kaggle CLI (For Automated Downloads)

1. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

2. **Set up API Credentials**:
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token" (downloads kaggle.json)
   - Move to ~/.kaggle/:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download Dataset**:
   ```bash
   cd /Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/data/raw

   # Download full dataset
   kaggle datasets download -d naveengill/cicids2017-dataset
   unzip cicids2017-dataset.zip

   # Or download preprocessed version
   kaggle datasets download -d ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
   unzip cicids2017-cleaned-and-preprocessed.zip
   ```

### Option 3: Official UNB Source

1. Visit: http://205.174.165.80/CICDataset/CIC-IDS-2017/
2. Navigate to Dataset folder
3. Download MachineLearningCSV.zip (224 MB)
4. Extract to data/raw/ directory

Note: The official server paths have changed over time. If link doesn't work, use Kaggle instead.

---

## Dataset File Structure

After downloading, you should have:

```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv          # Benign traffic only (~470 MB, ~530K rows)
├── Tuesday-WorkingHours.pcap_ISCX.csv         # Brute Force attacks
├── Wednesday-workingHours.pcap_ISCX.csv       # DoS/DDoS attacks
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## Start with Monday Data

For initial development and prototyping:
- Download **only** Monday-WorkingHours.pcap_ISCX.csv first
- Contains ~530,000 rows of benign network traffic
- File size: ~470 MB
- Perfect for training anomaly detection baseline

---

## Verification

After downloading, verify the file:

```bash
# Check file exists and size
ls -lh /Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/data/raw/Monday-WorkingHours.pcap_ISCX.csv

# Count rows (should be 530K+)
wc -l /Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/data/raw/Monday-WorkingHours.pcap_ISCX.csv

# Preview first few lines
head /Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/data/raw/Monday-WorkingHours.pcap_ISCX.csv
```

Expected output:
- File size: ~400-500 MB
- Row count: ~530,000 rows
- Columns: 79 features + 1 label column

---

## Troubleshooting

**Issue**: Kaggle download fails with authentication error
- **Solution**: Ensure kaggle.json is in ~/.kaggle/ with correct permissions (600)

**Issue**: File too large to download on slow connection
- **Solution**: Use preprocessed version (smaller) or download overnight

**Issue**: CSV encoding errors when loading
- **Solution**: Use `encoding='utf-8'` or `encoding='latin1'` in pandas.read_csv()

**Issue**: Out of memory errors when loading
- **Solution**: Use chunked reading or sample the dataset (see EDA notebook)

---

## Quick Test with Sample Data

If you want to start development before downloading the full dataset, the EDA notebook includes a sample data generator that creates synthetic network traffic features for testing.

---

## Next Steps

Once Monday-WorkingHours.pcap_ISCX.csv is in data/raw/:
1. Run the EDA notebook: `jupyter notebook notebooks/01_eda.ipynb`
2. Analyze dataset characteristics
3. Build preprocessing pipeline
4. Begin model training

---

## Dataset Citation

If you use this dataset, please cite:

```
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani,
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization",
4th International Conference on Information Systems Security and Privacy (ICISSP),
Portugal, January 2018
```

---

## Additional Resources

- Official Dataset Page: https://www.unb.ca/cic/datasets/ids-2017.html
- Dataset Paper: https://www.scitepress.org/Papers/2018/66398/66398.pdf
- CICFlowMeter Tool: https://github.com/ahlashkari/CICFlowMeter
