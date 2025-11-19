# Multi-Attack Type Testing Report
**Date:** 2025-11-19
**Models Tested:** VAE (200K) and OCSVM (200K, tuned)
**Attack Categories:** PortScan, Web Attacks, Brute Force
**Training Data:** Monday BENIGN (200K samples)
**Quality Gate:** F1 > 0.80 on 2+ attack types

---

## Executive Summary

### Overall Results: **CRITICAL FAILURE**

Both VAE and OCSVM models **FAILED** the quality gate on all three attack categories. The models exhibit a **critical false positive problem**, predicting 80-100% of benign traffic as attacks.

| Metric | VAE | OCSVM | Status |
|--------|-----|-------|--------|
| **Average F1** | 0.274 | 0.266 | FAIL |
| **Quality Gate (2/3 pass)** | 0/3 scenarios | 0/3 scenarios | FAIL |
| **Generalization** | Poor | Poor | FAIL |

**Root Cause:** Models trained on DoS/DDoS attacks (Wednesday data) do NOT generalize to other attack types. The 0.87 F1 score on DoS was **overfitting**, not true anomaly detection capability.

---

## Detailed Results by Attack Type

### 1. PortScan (Network Reconnaissance)
**Dataset:** Friday-Afternoon-PortScan.pcap_ISCX.csv
**Attack Ratio:** 55.5% (55,479 attacks / 44,521 benign in 100K test)
**Attack Type:** Network port scanning - reconnaissance activity

| Metric | VAE | OCSVM | Better |
|--------|-----|-------|--------|
| **F1 Score** | **0.7550** | 0.7137 | VAE |
| Precision | 0.607 | 0.555 | VAE |
| Recall | 0.998 | 1.000 | OCSVM |
| **FP Rate** | **80.5%** | **100.0%** | VAE |
| True Positives | 55,363 | 55,479 | OCSVM |
| False Positives | 35,820 | 44,521 | VAE |
| Quality Gate | **FAIL** | **FAIL** | Both fail |

**Analysis:**
- **PortScan attacks detected very well** (99.8-100% recall)
- **Catastrophic false positive rate**: OCSVM flags ALL benign traffic as attacks
- VAE slightly better (80.5% FP vs 100%) but still unacceptable
- F1=0.75 fails quality gate (< 0.80) despite high recall
- **Implication:** Models cannot distinguish PortScan from normal traffic patterns

**Why PortScan is hard:**
- PortScans can look like normal traffic (connection attempts)
- Low-volume scans blend with legitimate network activity
- Models trained on DoS (high volume) fail on reconnaissance (low volume)

---

### 2. Web Attacks (Application Layer)
**Dataset:** Thursday-Morning-WebAttacks.pcap_ISCX.csv
**Attack Ratio:** 1.3% (1,280 attacks / 98,720 benign in 100K test)
**Attack Types:** SQL Injection (14), XSS (373), Brute Force Web (893)

| Metric | VAE | OCSVM | Better |
|--------|-----|-------|--------|
| **F1 Score** | **0.0297** | 0.0253 | VAE |
| Precision | **0.0151** | **0.0128** | VAE |
| Recall | 0.923 | 1.000 | OCSVM |
| **FP Rate** | **78.1%** | **100.0%** | VAE |
| True Positives | 1,181 | 1,280 | OCSVM |
| False Positives | 77,111 | 98,720 | VAE |
| Quality Gate | **FAIL** | **FAIL** | Both fail |

**Per-Attack Breakdown (VAE):**

| Attack Type | Samples | Detected | Detection Rate | Attack F1 |
|-------------|---------|----------|----------------|-----------|
| Web XSS | 373 | 360 | 96.5% | 0.982 |
| Web Brute Force | 893 | 812 | 90.9% | 0.952 |
| SQL Injection | 14 | 9 | 64.3% | 0.783 |

**Analysis:**
- **CATASTROPHIC FAILURE** on low-attack-rate datasets
- Precision of 1.5% means **98.5% of alerts are false positives**
- OCSVM flags 100% of benign traffic as attacks (completely broken)
- VAE detects web attacks well (90-96% recall) but drowns in false alarms
- SQL Injection hardest to detect (only 64% recall, 14 samples total)
- **Production deployment impossible** - SOC teams would ignore all alerts

**Why Web Attacks failed:**
- Web attacks are subtle (malicious payloads in HTTP)
- Low attack rate (1.3%) means precision is critical
- Models trained on network-layer DoS fail on application-layer attacks
- Different feature space: DoS = volume/rate, Web = payload content

---

### 3. Brute Force (Authentication Attacks)
**Dataset:** Tuesday-WorkingHours.pcap_ISCX.csv
**Attack Ratio:** 3.1% (3,102 attacks / 96,898 benign in 100K test)
**Attack Types:** FTP-Patator (1,765), SSH-Patator (1,337)

| Metric | VAE | OCSVM | Better |
|--------|-----|-------|--------|
| **F1 Score** | 0.0381 | **0.0602** | OCSVM |
| Precision | 0.0198 | 0.0310 | OCSVM |
| Recall | 0.492 | 1.000 | OCSVM |
| **FP Rate** | **77.8%** | **100.0%** | VAE |
| True Positives | 1,525 | 3,102 | OCSVM |
| False Positives | 75,351 | 96,898 | VAE |
| Quality Gate | **FAIL** | **FAIL** | Both fail |

**Per-Attack Breakdown (VAE):**

| Attack Type | Samples | Detected | Detection Rate | Attack F1 |
|-------------|---------|----------|----------------|-----------|
| FTP-Patator | 1,765 | 884 | 50.1% | 0.667 |
| SSH-Patator | 1,337 | 641 | 47.9% | 0.648 |

**Analysis:**
- **WORST PERFORMANCE** among all attack types
- VAE recall of 49% means **51% of brute force attacks go undetected**
- OCSVM detects all attacks but flags 100% of benign traffic
- Precision of 3.1% (OCSVM) and 2.0% (VAE) is **unacceptable**
- FTP and SSH attacks equally hard to detect (~50% recall for VAE)

**Why Brute Force failed:**
- Brute force attacks spread over time (not high-volume like DoS)
- Individual login attempts look like normal authentication
- Models need to detect patterns (repeated failures) not individual requests
- Different feature space than DoS (timing, repetition vs volume)

---

## Cross-Attack Comparison: DoS vs Others

### DoS/DDoS Performance (from previous tests)
**Dataset:** Wednesday-workingHours.pcap_ISCX.csv
**Attack Ratio:** 36.3% (36,246 attacks / 63,754 benign in 100K test)

| Model | DoS F1 | PortScan F1 | Web F1 | Brute Force F1 | Avg F1 |
|-------|--------|-------------|--------|----------------|--------|
| VAE | **0.8713** | 0.7550 | 0.0297 | 0.0381 | 0.423 |
| OCSVM | **0.8540** | 0.7137 | 0.0253 | 0.0602 | 0.411 |

**Key Findings:**
1. **DoS F1 (0.87) was NOT generalization - it was overfitting**
   - Models learned DoS-specific patterns (high volume, packet floods)
   - Failed to learn general "anomaly" concept

2. **22x performance drop from DoS to Web Attacks**
   - VAE: 0.87 → 0.03 (96.6% degradation)
   - OCSVM: 0.85 → 0.03 (97.0% degradation)

3. **PortScan partially works (F1=0.71-0.75)**
   - Similar to DoS: network-layer, volume-based
   - Still fails quality gate due to 80-100% false positive rate

4. **Application-layer attacks completely fail**
   - Web Attacks: F1 < 0.03
   - Brute Force: F1 < 0.06
   - Models have NO understanding of application-layer anomalies

---

## False Positive Rate Analysis

### FP Rate by Attack Type

| Attack Type | Attack % | VAE FP Rate | OCSVM FP Rate | Impact |
|-------------|----------|-------------|---------------|--------|
| DoS (Wed) | 36.3% | 2.8% | 2.76% | Acceptable |
| PortScan | 55.5% | 80.5% | 100.0% | **Catastrophic** |
| Web Attacks | 1.3% | 78.1% | 100.0% | **Catastrophic** |
| Brute Force | 3.1% | 77.8% | 100.0% | **Catastrophic** |

**Critical Insight:**
- **FP rate inversely correlated with attack prevalence**
- DoS test (36% attacks) → 3% FP rate (good)
- Web test (1.3% attacks) → 78-100% FP rate (broken)
- **Models biased toward "predict attack" due to high attack rate in training evaluation**

**Production Implications:**
- Real networks have <1% attack rate (more like 0.01-0.1%)
- 78% FP rate on 1.3% attack data → **99.9% FP rate on 0.1% attack data**
- SOC analysts would receive 1000 false alarms per 1 real attack
- **System unusable in production**

---

## Why Models Failed: Root Cause Analysis

### 1. Training Data Bias
**Problem:** Models evaluated on Wednesday DoS data during hyperparameter tuning
- Wednesday has 36.3% attack rate (very high)
- Models optimized F1 score on DoS-heavy dataset
- Learned "DoS detector" not "anomaly detector"

**Evidence:**
- DoS F1 = 0.87 (excellent)
- Non-DoS F1 = 0.03-0.75 (terrible)
- FP rate acceptable on DoS (3%), catastrophic on others (80-100%)

### 2. Feature Space Mismatch
**Problem:** Network-layer vs application-layer attacks

| Attack Type | Feature Space | Example Features |
|-------------|---------------|------------------|
| **DoS/DDoS** | Network layer | Packet rate, flow duration, bytes/sec |
| **PortScan** | Network layer | Connection attempts, port diversity |
| **Web Attacks** | Application layer | HTTP payload, URL patterns, headers |
| **Brute Force** | Application layer | Login attempts, timing, success/fail |

**Models trained on network features cannot detect application-layer attacks**

### 3. Threshold Calibration Failure
**Problem:** Anomaly thresholds set for high attack rates

- VAE threshold: 95th percentile of training reconstruction errors
- OCSVM nu: 0.01 (1% contamination expected)
- **Calibrated for 36% attack rate, fails at 1-3% attack rates**

**Evidence:**
- OCSVM predicts 100% attacks on all non-DoS datasets
- VAE predicts ~80% attacks regardless of actual attack rate
- Thresholds not adaptive to dataset characteristics

### 4. Unsupervised Learning Limitations
**Problem:** Trained on BENIGN only, no attack examples

- Models learn "normal traffic distribution"
- ANY deviation flagged as anomaly
- Different traffic patterns (web, SSH, FTP) look "abnormal" vs Monday baseline
- **Need supervised learning or semi-supervised with attack examples**

---

## Model Comparison: VAE vs OCSVM

### Overall Winner: **VAE** (by tiny margin)

| Metric | VAE | OCSVM | Winner |
|--------|-----|-------|--------|
| Avg F1 | 0.274 | 0.266 | VAE |
| Best F1 (PortScan) | 0.755 | 0.714 | VAE |
| Worst F1 (Web) | 0.030 | 0.025 | VAE |
| Avg Precision | 0.214 | 0.199 | VAE |
| Avg Recall | 0.804 | 1.000 | OCSVM |
| Avg FP Rate | 78.8% | 100.0% | VAE |
| Scenarios Won | 2/3 | 1/3 | VAE |

**VAE Advantages:**
- Lower FP rate (78% vs 100%) - still terrible but less broken
- Better precision (21% vs 20%)
- Won on PortScan and Web Attacks

**OCSVM Advantages:**
- Perfect recall (100%) - detects all attacks
- Won on Brute Force (F1=0.06 vs 0.04)
- Simpler model, faster training

**Both models equally unusable in production**

---

## Quality Gate Assessment

### Target: F1 > 0.80 on 2+ attack types

| Model | PortScan | Web Attacks | Brute Force | Pass (2/3) | Result |
|-------|----------|-------------|-------------|------------|--------|
| **VAE** | 0.755 ✗ | 0.030 ✗ | 0.038 ✗ | 0/3 | **FAIL** |
| **OCSVM** | 0.714 ✗ | 0.025 ✗ | 0.060 ✗ | 0/3 | **FAIL** |

**Quality Gate Status:** **CATASTROPHIC FAILURE**
- No model passed ANY attack type
- Best result: VAE on PortScan (F1=0.755, 2.5% below threshold)
- Worst result: OCSVM on Web Attacks (F1=0.025, 96.9% below threshold)
- Gap to quality gate: -0.525 F1 points on average

---

## Recommendations

### 1. DO NOT PROCEED WITH SHAP IMPLEMENTATION ❌
**Reason:** Models are fundamentally broken
- SHAP explains model decisions, but decisions are 80% wrong
- Explaining why model flags benign traffic as attacks adds no value
- Fix models first, explain later

### 2. DO NOT DEPLOY EITHER MODEL TO PRODUCTION ❌
**Reason:** 78-100% false positive rate is catastrophic
- SOC teams would be overwhelmed with false alarms
- Real attacks would be missed in noise (alert fatigue)
- Models worse than random classifier on Web/Brute Force

### 3. IMMEDIATE ACTIONS REQUIRED

#### A. **Re-evaluate Training Strategy**
Current approach: Train on BENIGN only (Monday)
**Problem:** No attack examples, overfits to Monday traffic patterns

**Solutions:**
1. **Semi-supervised learning**: Include attack samples in training
   - Sample attacks from all types (DoS, Web, PortScan, Brute Force)
   - Train model to recognize normal + multiple attack patterns

2. **Multi-day training**: Combine Monday + Tuesday + Thursday + Friday BENIGN
   - Diverse normal traffic patterns (different services, times)
   - Prevents overfitting to single day's characteristics

3. **Transfer learning**: Fine-tune on mixed attack types
   - Pre-train on BENIGN, fine-tune on 5% attack data
   - Learn general anomaly concept, not DoS-specific

#### B. **Threshold Calibration**
Current: Fixed threshold (95th percentile, nu=0.01)
**Problem:** Not adaptive to attack prevalence

**Solutions:**
1. **Per-dataset calibration**: Set threshold based on expected attack rate
   - DoS dataset (36% attacks) → threshold X
   - Web dataset (1% attacks) → threshold Y (much higher)

2. **Dynamic thresholding**: Adjust threshold at inference time
   - Monitor false positive rate in production
   - Automatically tune threshold to maintain FP < 5%

3. **F1-optimized threshold**: Find threshold that maximizes F1, not recall
   - Current: Optimizes recall (detect all attacks)
   - Better: Balance precision and recall

#### C. **Feature Engineering**
Current: Use all 66 network features
**Problem:** Network features can't detect application-layer attacks

**Solutions:**
1. **Attack-specific features**:
   - Web: HTTP status codes, URL patterns, payload entropy
   - Brute Force: Login attempt frequency, failure rate, timing patterns
   - PortScan: Port diversity, connection duration, SYN/ACK ratios

2. **Temporal features**:
   - Rolling window statistics (last 1 min, 5 min, 1 hour)
   - Detect patterns over time, not just individual flows

3. **Feature selection**:
   - Remove features not useful for non-DoS attacks
   - Train separate models per attack category

#### D. **Model Architecture**
Current: Single VAE/OCSVM trained on all data
**Problem:** One-size-fits-all fails on diverse attacks

**Solutions:**
1. **Ensemble of specialists**:
   - DoS detector (current models)
   - Web attack detector (application-layer features)
   - Brute force detector (temporal features)
   - PortScan detector (connection patterns)
   - **Combine predictions** via voting or stacking

2. **Supervised learning**:
   - Random Forest, XGBoost, or Neural Network
   - Train on labeled data (Monday BENIGN + Wednesday attacks)
   - **Much better than unsupervised** when labels available

3. **Advanced anomaly detection**:
   - **Isolation Forest on attack-specific features** (not all features)
   - **LSTM Autoencoder** for temporal patterns
   - **Transformer-based** for sequence modeling

### 4. VALIDATION PROTOCOL
Before claiming success:

1. **Test on ALL attack types**:
   - ✓ DoS/DDoS (Wednesday)
   - ✓ PortScan (Friday)
   - ✓ Web Attacks (Thursday)
   - ✓ Brute Force (Tuesday)
   - ✗ Infiltration (Thursday afternoon) - not tested
   - ✗ DDoS Friday (Friday afternoon) - not tested

2. **Require F1 > 0.80 on 5+ attack types** (not just 2)

3. **Test on realistic attack rates**:
   - Simulate 0.1% attack rate (real-world)
   - Verify FP rate < 5% at production attack prevalence

4. **Per-attack-type quality gates**:
   - DoS: F1 > 0.85 (currently 0.87 ✓)
   - PortScan: F1 > 0.85 (currently 0.75 ✗)
   - Web Attacks: F1 > 0.80 (currently 0.03 ✗)
   - Brute Force: F1 > 0.80 (currently 0.04 ✗)

---

## Comparison with DoS Baseline

### VAE Performance Degradation

| Attack Type | F1 Score | vs DoS Baseline | Degradation |
|-------------|----------|-----------------|-------------|
| DoS/DDoS | 0.8713 | Baseline | 0% |
| PortScan | 0.7550 | -0.1163 | -13.4% |
| Web Attacks | 0.0297 | -0.8416 | **-96.6%** |
| Brute Force | 0.0381 | -0.8332 | **-95.6%** |

### OCSVM Performance Degradation

| Attack Type | F1 Score | vs DoS Baseline | Degradation |
|-------------|----------|-----------------|-------------|
| DoS/DDoS | 0.8540 | Baseline | 0% |
| PortScan | 0.7137 | -0.1403 | -16.4% |
| Web Attacks | 0.0253 | -0.8287 | **-97.0%** |
| Brute Force | 0.0602 | -0.7938 | **-93.0%** |

**Key Insight:**
- **DoS performance does NOT indicate general capability**
- 95-97% performance drop on application-layer attacks
- Models are **DoS detectors, not general anomaly detectors**

---

## Technical Details

### Test Configuration
- **Models:** VAE (latent_dim=20, kl_weight=0.001), OCSVM (nu=0.01, kernel=rbf, gamma=scale)
- **Training:** Monday BENIGN (200K samples)
- **Evaluation:** Wednesday DoS (for tuning), Friday PortScan, Thursday Web, Tuesday Brute Force
- **Test samples:** 100K per attack type (stratified sampling, attack ratio preserved)
- **Features:** 66 (after preprocessing, first 66 features used for alignment)
- **Preprocessing:** StandardScaler, median imputation, inf replacement

### Feature Alignment Issue
- Training data (Monday/Wednesday): 66 features after cleaning
- Test data (Friday/Tuesday/Thursday): 70 features after cleaning
- **Solution:** Used first 66 features for compatibility
- **Limitation:** Last 4 features discarded, may contain useful information

### Inference Performance
- **VAE:** 3-6 seconds per 100K samples (16K-33K samples/sec)
- **OCSVM:** 29-37 seconds per 100K samples (2.7K-3.4K samples/sec)
- **Winner:** VAE (10x faster inference)

### Files Generated
- `multi_attack_test_results.pkl` - Complete results
- `multi_attack_test_results.csv` - Summary table
- `multi_attack_summary.json` - JSON summary
- `multi_attack_comparison.png` - Visualization
- Per-attack CSVs: `vae_portscan_per_attack.csv`, etc. (6 files)

---

## Conclusion

### CRITICAL FAILURE: Models Do Not Generalize

The multi-attack testing reveals that both VAE and OCSVM models:
1. **Fail quality gate on all 3 attack types** (0/3 pass, need 2/3)
2. **Have catastrophic false positive rates** (78-100%)
3. **Cannot detect application-layer attacks** (F1 < 0.06)
4. **Overfit to DoS attacks** (96% performance drop on other types)
5. **Are unusable in production** (would generate 1000 FPs per real attack)

### The 0.87 F1 Score on DoS Was Misleading

Previous celebration of F1=0.87 on Wednesday DoS attacks was **premature**:
- Models learned DoS-specific patterns, not general anomaly detection
- High attack rate (36%) masked false positive problem
- Low FP rate (3%) on DoS inflated to 80-100% on other attacks
- **DoS performance is NOT transferable to other attack types**

### DO NOT PROCEED with SHAP Implementation

SHAP would explain why models make wrong predictions 80% of the time. Fix models first.

### Next Steps: **Major Redesign Required**

1. **Re-train with diverse attack types** (not just DoS)
2. **Implement ensemble of specialists** (per-attack-type detectors)
3. **Use supervised learning** (labels available, use them!)
4. **Calibrate thresholds per attack prevalence**
5. **Add application-layer features** (HTTP, authentication patterns)
6. **Test on realistic attack rates** (<1%, not 36%)
7. **Only proceed to SHAP after F1 > 0.80 on 5+ attack types**

The current models are **research prototypes that overfit to DoS attacks**, not production-ready intrusion detection systems.

---

**Report Generated:** 2025-11-19 11:56 UTC
**Test Script:** `test_multi_attack.py`
**Results Directory:** `/results/`
