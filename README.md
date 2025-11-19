# Network Intrusion Detection System

Production-ready DoS/DDoS attack detection using One-Class SVM with SHAP explainability.

[![Status](https://img.shields.io/badge/status-deployed-success)](https://nids-dashboard.onrender.com)
[![F1 Score](https://img.shields.io/badge/F1--Score-0.8540-blue)](https://github.com/byessilyurt/network-intrusion-detection)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://nids-dashboard.onrender.com)

## 🚀 Live Demo

Try the deployed system (no installation required):

- **Dashboard**: [nids-dashboard.onrender.com](https://nids-dashboard.onrender.com)
- **API**: [nids-api-6pus.onrender.com](https://nids-api-6pus.onrender.com)
- **API Docs**: [nids-api-6pus.onrender.com/docs](https://nids-api-6pus.onrender.com/docs)

> **Note**: Free tier services may take 30-60s to wake up after inactivity.

## Quick Start

**Docker (Recommended)**

```bash
docker-compose up -d
```

Access services at:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

**Local Development**

```bash
pip install -r requirements.txt
uvicorn src.api.app:app --reload          # API on :8000
streamlit run src/dashboard/app.py        # Dashboard on :8501
```

## Features

- **Real-time Detection**: Identifies volumetric DoS/DDoS attacks from network flow data
- **Explainable AI**: SHAP feature importance for every prediction
- **REST API**: FastAPI with `/predict`, `/health`, `/model/info` endpoints
- **Interactive Dashboard**: Streamlit UI for single/batch flow analysis
- **Production Ready**: Docker deployment, health monitoring, comprehensive logging

## Performance

**Model**: One-Class SVM (nu=0.02, RBF kernel)
**Training Data**: CICIDS2017 dataset (200K benign flows)

| Metric | Value |
|--------|-------|
| F1 Score | 0.8540 |
| Precision | 92.4% |
| Recall | 79.4% |

**Attack Detection Rates**:
- DoS Hulk: 81.2% F1
- DoS GoldenEye: 81.9% F1
- DoS Slowhttptest: 86.2% F1
- Heartbleed: 100% F1

## Tech Stack

- **ML/AI**: scikit-learn, SHAP, pandas, numpy
- **API**: FastAPI, Pydantic, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Deployment**: Docker, Render.com
- **Dataset**: CICIDS2017

## Project Structure

```
.
├── src/
│   ├── api/           # FastAPI REST API
│   ├── dashboard/     # Streamlit web UI
│   ├── models/        # ML model implementations
│   ├── data/          # Data preprocessing
│   └── evaluation/    # Metrics & explainability
├── models/            # Trained model files
├── notebooks/         # Training notebooks
├── Dockerfile.api     # API container
├── Dockerfile.dashboard  # Dashboard container
└── docker-compose.yml # Multi-container setup
```

## API Usage

**Predict Network Flow**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Destination Port": 80,
      "Flow Duration": 120000,
      "Total Fwd Packets": 10,
      ...
    }
  }'
```

**Response**

```json
{
  "prediction": 1,
  "prediction_label": "ATTACK",
  "anomaly_score": -0.234,
  "confidence": 87.5,
  "top_features": [
    {"feature": "Flow Bytes/s", "shap_value": 0.12},
    {"feature": "Fwd Packets/s", "shap_value": 0.09}
  ],
  "explanation": "🚨 ATTACK DETECTED: Anomalous behavior detected..."
}
```

## Deployment

**Render.com (Free Tier)**

1. Fork this repository
2. Connect to [Render.com](https://render.com)
3. Deploy using `render.yaml` (auto-detected)

**Docker**

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Development

**Training New Models**

```bash
jupyter notebook notebooks/02_train_all_models.ipynb
```

**Running Tests**

```bash
pytest tests/
```

**Code Structure**

- `src/api/app.py` - FastAPI application
- `src/models/one_class_svm.py` - OCSVM detector
- `src/dashboard/app.py` - Streamlit dashboard
- `src/evaluation/explainability.py` - SHAP integration

## Limitations

**In Scope**: DoS/DDoS volumetric attacks (Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed)

**Out of Scope**: Application-layer attacks (SQL injection, XSS), authentication attacks (brute force), port scanning, malware C2

For comprehensive security, deploy alongside WAF and HIDS.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

Dataset: Canadian Institute for Cybersecurity. (2017). Intrusion Detection Evaluation Dataset (CICIDS2017).

## Contact

Created by [Yusuf Yesilyurt](https://github.com/byessilyurt)

---

**Live System**: [Dashboard](https://nids-dashboard.onrender.com) | [API](https://nids-api-6pus.onrender.com)
