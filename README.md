# AI-Based Fraud Detection & Risk Monitoring System for Mobile Payments
### The Case of Ghana

> A stacked ensemble machine learning system combining Logistic Regression, Random Forest, XGBoost/LightGBM, and LSTM for real-time mobile money fraud detection, with a live browser dashboard, REST API, and WebSocket stream.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Folder Structure](#folder-structure)
4. [Datasets](#datasets)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Running the Training Pipeline](#running-the-training-pipeline)
8. [Running the Real-Time System](#running-the-real-time-system)
9. [API Endpoints](#api-endpoints)
10. [Dashboard](#dashboard)
11. [Model Architecture](#model-architecture)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Cloud Deployment](#cloud-deployment)
14. [File Reference](#file-reference)
15. [Troubleshooting](#troubleshooting)
16. [References](#references)

---

## Prerequisites

Before you begin, make sure you have the following installed on your system:

- **Python 3.8 or higher** - Download from [python.org](https://www.python.org/downloads/)
- **Git** - Download from [git-scm.com](https://git-scm.com/downloads)
- **A code editor** (optional but recommended) - We suggest [Visual Studio Code](https://code.visualstudio.com/)

### System Requirements
- **RAM**: At least 8GB (16GB recommended for full PaySim training)
- **Storage**: 5GB free space for datasets and models
- **Operating System**: Windows, macOS, or Linux

### Quick Check
Open a terminal/command prompt and run these commands to verify your setup:

```bash
# Check Python version
python --version
# Should show: Python 3.8.x or higher

# Check pip (Python package installer)
pip --version
# Should show pip version information
```

If Python is not installed, download and install it from python.org. Make sure to check "Add Python to PATH" during installation on Windows.

---

## Project Overview

This system was developed as part of a thesis studying AI-based fraud detection for mobile money payments in Ghana. It addresses three research objectives:

1. Design and implement a stacked ensemble that combines four complementary base learners into a single fraud score.
2. Evaluate the system across three public fraud datasets with different fraud environments and imbalance ratios.
3. Build a deployable real-time fraud monitoring pipeline with a live browser dashboard and explainable alerts.

**Key results on PaySim (primary dataset):**

| Model | Precision | Recall | F1-Score | AUC-ROC | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.834 | 0.701 | 0.763 | 0.931 | 0.682 |
| Random Forest | 0.869 | 0.812 | 0.840 | 0.957 | 0.771 |
| XGBoost | 0.882 | 0.841 | 0.861 | 0.964 | 0.798 |
| LightGBM | 0.876 | 0.835 | 0.855 | 0.961 | 0.792 |
| LSTM | 0.851 | 0.796 | 0.823 | 0.948 | 0.752 |
| **Stacked Ensemble** | **0.903** | **0.872** | **0.887** | **0.978** | **0.831** |

---

## Folder Structure

```
fraud_project/
├── data/                             ← place all CSV files here (see Datasets)
│   ├── creditcard.csv
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── PS_20174392719_1491204439457_log.csv
│
├── outputs/                          ← auto-created by train_model.py
│   ├── models/
│   │   ├── CreditCard/
│   │   │   ├── meta_xgboost.joblib
│   │   │   ├── base_lr.joblib
│   │   │   ├── base_rf.joblib
│   │   │   ├── base_xgb.joblib
│   │   │   └── base_lgb.joblib
│   │   ├── IEEE-CIS/
│   │   │   └── (same files as CreditCard/)
│   │   └── PaySim/
│   │       ├── meta_xgboost.joblib
│   │       ├── base_lr.joblib
│   │       ├── base_rf.joblib
│   │       ├── base_xgb.joblib
│   │       ├── base_lgb.joblib
│   │       └── scaler.joblib         ← used by the real-time API
│   ├── cm_*.png                      ← confusion matrix plots
│   └── shap_meta_*.png               ← SHAP feature importance plots
│
├── fraud_realtime/                   ← standalone real-time system
│   ├── app/
│   │   └── main.py                   ← FastAPI backend
│   ├── static/
│   │   └── dashboard.html            ← live browser dashboard
│   ├── models/                       ← copy PaySim artefacts here for the API
│   │   ├── stacked_ensemble.joblib
│   │   └── scaler.joblib
│   ├── train_and_save.py             ← trains and saves API-ready model
│   ├── stream_simulator.py           ← simulates a live transaction stream
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── config.py                         ← all paths, seeds, and hyperparameters
├── data_loader.py                    ← dataset loaders + PaySim feature engineering
├── models.py                         ← model constructors (LR, RF, XGB, LGB, LSTM, meta)
├── stacking.py                       ← OOF stacking matrix + meta-model trainer
├── evaluate.py                       ← metrics, CV, confusion matrix, SHAP, latency
├── train_model.py                    ← main entry point — run this to train everything
└── requirements.txt
```

---

## Datasets

All three datasets must be downloaded manually from Kaggle and placed in the `data/` folder. The filenames must match exactly as shown.

### 1. Credit Card Fraud Detection
- **Source:** [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Size:** 284,807 rows | 31 features | 0.173% fraud rate
- **Purpose:** Benchmark dataset for testing class imbalance handling under extreme rarity conditions. V1–V28 features are PCA-anonymised and used as-is.

### 2. IEEE-CIS Fraud Detection
- **Source:** [kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv)
- **Files:** `train_transaction.csv` + `train_identity.csv` (both required — joined on `TransactionID`)
- **Size:** ~590,540 rows after merge | 3.5% fraud rate
- **Purpose:** Extended feature engineering dataset with device, identity, and behavioural signals. High-cardinality fields use target encoding (Micci-Barreca 2001).

### 3. PaySim Mobile Money Simulation ← Primary Dataset
- **Source:** [kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv)
- **File:** `PS_20174392719_1491204439457_log.csv`
- **Size:** 1,048,575 rows | 1.29% fraud rate
- **Purpose:** Primary mobile money dataset (Lopez-Rojas et al. 2016). The only dataset where LSTM sequences are built and the five engineered features are applied. The trained PaySim model powers the real-time API and dashboard.

> **Note:** IEEE-CIS requires a Kaggle account and agreement to competition rules before downloading. Credit Card and PaySim are freely available datasets.

---

## Installation

### Requirements

- Python 3.10 or higher
- Node.js 18+ (only needed if regenerating the presentation)
- 8 GB RAM minimum (16 GB recommended for PaySim + LSTM)

### Install Python dependencies

```bash
# From the project root
pip install -r requirements.txt
```

`requirements.txt` covers:

```
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.3.0
tensorflow>=2.16.0
imbalanced-learn>=0.12.0
category_encoders>=2.6.0
shap>=0.45.0
joblib>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
websockets>=12.0
python-multipart>=0.0.9
aiohttp>=3.9.0
pydantic>=2.0.0
```

---

## Quick Start

**Follow these steps to get the fraud detection system running in under 10 minutes:**

### Step 1: Download and Setup
```bash
# Clone or download this repository
git clone <repository-url>
cd fraud_project

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Train a Quick Model (Optional but Recommended)
For fastest testing, train on a small sample of the PaySim data:

**Recommended sample sizes:**
- **Quick test**: `--sample 0.01` (10,000 rows, ~1-2 minutes)
- **Development**: `--sample 0.05` (50,000 rows, ~10-15 minutes) 
- **Production**: `--sample 1.0` (full dataset, ~45-90 minutes)

```bash
cd fraud_realtime
python train_and_save.py --data ../data/PS_20174392719_1491204439457_log.csv --sample 0.01
cd ..
```

**Note**: Minimum sample size is `--sample 0.005` (5,000 rows) to ensure SMOTE-ENN has enough fraud examples to work.

### Step 3: Start the API Server
```bash
cd fraud_realtime
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Keep this terminal running. The API is now live at `http://localhost:8000`

### Step 4: Open the Dashboard
Open your web browser and go to: `http://localhost:8000/dashboard`

### Step 5: Start the Transaction Stream
Open a second terminal and run:
```bash
cd fraud_realtime
python stream_simulator.py --demo --rate 5
```

**What you'll see:**
- Live dashboard with fraud scores updating every few seconds
- Real-time charts showing transaction risk levels
- Transaction feed with color-coded risk bands (LOW/MEDIUM/HIGH)

**Need help?** Check the [Troubleshooting](#troubleshooting) section below.

---

## Running the Training Pipeline

### Step 1 — Configure paths

Open `config.py` and verify the paths under `DATA` point to your CSV files. Everything else (seeds, hyperparameters, thresholds) is already set to match the thesis values.

```python
DATA = {
    "creditcard": ROOT / "data" / "creditcard.csv",
    "ieee_trans":  ROOT / "data" / "train_transaction.csv",
    "ieee_id":     ROOT / "data" / "train_identity.csv",
    "paysim":      ROOT / "data" / "PS_20174392719_1491204439457_log.csv",
}
```

### Step 2 — Run the training script

```bash
python train_model.py
```

For faster PaySim prototyping, run:

```bash
# Quick test (10,000 rows, ~5 minutes)
python train_model.py --sample 0.01

# Development iteration (50,000 rows, ~15-20 minutes)
python train_model.py --sample 0.05

# Full production training (1M+ rows, ~45-90 minutes)
python train_model.py --sample 1.0
```

**Sample size guidelines:**
- `--sample 0.01`: Fast prototyping, basic functionality testing
- `--sample 0.05`: Development work, meaningful evaluation metrics
- `--sample 1.0`: Production-quality training (recommended for final models)

This runs the full pipeline for all three datasets in sequence:

1. Loads and cleans each dataset
2. Applies SMOTE-ENN on the training split only
3. Builds LSTM sequences for PaySim
4. Runs 5-fold CV on the logistic regression baseline
5. Generates the 5-column OOF stacking matrix Z
6. Evaluates each base learner individually
7. Trains the XGBoost meta-model on Z
8. Computes SHAP values on the meta-model
9. Measures per-transaction latency
10. Saves all model artefacts to `outputs/models/`
11. Prints the cross-dataset comparison table (Table 12)

**Expected runtimes on a standard laptop (no GPU):**

| Dataset | Approximate Time |
|---|---|
| Credit Card | 5 – 10 minutes |
| IEEE-CIS | 15 – 25 minutes |
| PaySim (with LSTM) | 45 – 90 minutes |

### Step 3 — PaySim only (optional shortcut)

If you want to test the system end-to-end before running all three datasets, comment out the Credit Card and IEEE-CIS blocks in `train_model.py`:

```python
def main():
    # X, y, _ = load_creditcard()
    # all_results["CreditCard"] = run_pipeline("CreditCard", X, y)

    # X, y, _ = load_ieee()
    # all_results["IEEE-CIS"] = run_pipeline("IEEE-CIS", X, y)

    X, y, _, scaler = load_paysim()
    all_results["PaySim"] = run_pipeline("PaySim", X, y, use_lstm=True)
```

The real-time API works fully with just the PaySim model.

### What to check in the outputs

After training completes, verify these in the `outputs/` folder:

| File | What it confirms |
|---|---|
| `cm_PaySim_StackedEnsemble.png` | Confusion matrix — false negatives (missed fraud) should be minimal |
| `shap_meta_PaySim.png` | SHAP bar chart — XGBoost/LGB should dominate, LSTM adds distinct signal |
| `models/PaySim/meta_xgboost.joblib` | Stacked ensemble model saved successfully |
| Terminal output: stacked F1 > all base models | Ensemble improvement confirmed |
| Terminal output: AUC-ROC > 0.95 for PaySim | Discrimination quality confirmed |
| Terminal output: latency mean < 50 ms | Deployment-ready confirmed |

---

## Running the Real-Time System

### Step 1 — Copy PaySim model artefacts to the API folder

```bash
cp outputs/models/PaySim/scaler.joblib        fraud_realtime/models/scaler.joblib
cp outputs/models/PaySim/meta_xgboost.joblib  fraud_realtime/models/stacked_ensemble.joblib
cp outputs/models/PaySim/base_lr.joblib       fraud_realtime/models/base_lr.joblib
cp outputs/models/PaySim/base_rf.joblib       fraud_realtime/models/base_rf.joblib
cp outputs/models/PaySim/base_xgb.joblib      fraud_realtime/models/base_xgb.joblib
cp outputs/models/PaySim/base_lgb.joblib      fraud_realtime/models/base_lgb.joblib
```

Alternatively, run `train_and_save.py` inside `fraud_realtime/` which retrains and saves all required files directly:

```bash
cd fraud_realtime
python train_and_save.py --data ../data/PS_20174392719_1491204439457_log.csv --sample 0.1
```

Use `--sample` when the PaySim CSV is too large to process full size during development.

```bash
cd fraud_realtime
python train_and_save.py --data ../data/PS_20174392719_1491204439457_log.csv
```

### Step 2 — Start the API (Terminal 1)

```bash
cd fraud_realtime
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3 — Start the simulator (Terminal 2)

```bash
# Synthetic demo transactions at 5 per second
python stream_simulator.py --demo --rate 5

# Replay real PaySim CSV at 20 per second
python stream_simulator.py --data ../data/PS_20174392719_1491204439457_log.csv --rate 20

# Both REST and WebSocket simultaneously
python stream_simulator.py --demo --mode both --rate 10
```

### Step 4 — Open the dashboard

```
http://localhost:8000/dashboard
```

### Step 5 — Test the REST API manually

```bash
# Single high-risk transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 950000,
    "oldbalanceOrg": 950000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 950000
  }'

# Batch CSV upload
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@data/sample.csv"

# System metrics
curl http://localhost:8000/metrics

# Recent high-risk alerts
curl http://localhost:8000/alerts
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — confirms model is loaded |
| `GET` | `/metrics` | Fraud rate, risk counts, latency stats |
| `GET` | `/alerts?limit=20` | Recent HIGH risk alerts with SHAP |
| `POST` | `/predict` | Score a single transaction (~18 ms) |
| `POST` | `/predict/batch` | Upload and score a CSV file |
| `WS` | `/ws/stream` | WebSocket — server pushes every scored transaction |
| `GET` | `/dashboard` | Live monitoring dashboard HTML |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Risk bands

| Band | P(Fraud) | Action |
|---|---|---|
| LOW | < 0.30 | PASS — transaction clears automatically |
| MEDIUM | 0.30 – 0.60 | REVIEW — queued for analyst review |
| HIGH | > 0.60 | BLOCK — immediate alert with SHAP explanation |

### Example response from `/predict`

```json
{
  "transaction_id": "a3f92b1c-...",
  "fraud_probability": 0.8741,
  "risk_band": "HIGH",
  "decision": "BLOCK",
  "shap_top5": {
    "amount": 0.312,
    "type_TRANSFER": 0.201,
    "balance_drop_flag": 0.188,
    "transaction_velocity": 0.094,
    "counterparty_spread": 0.061
  },
  "latency_ms": 17.4,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

---

## Dashboard

The live dashboard at `http://localhost:8000/dashboard` shows:

- **KPI bar** — total scored, fraud rate, LOW/MEDIUM/HIGH counts, average latency
- **Live probability chart** — rolling fraud probability for the last 60 transactions
- **Risk distribution doughnut** — proportion of transactions in each risk band
- **Latency chart** — per-transaction scoring time in milliseconds
- **Transaction feed** — scrolling table of every scored transaction with colour-coded bands
- **Alert panel** — real-time HIGH risk alerts with SHAP feature contributions

The dashboard connects via WebSocket automatically. Every transaction scored through `/predict` is pushed to all connected browser tabs within the same scoring cycle.

---

## Model Architecture

```
Raw transaction features
        │
        ▼
┌────────────────────────────────────────────────────┐
│              Level-0 Base Learners                  │
│                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐  │
│  │Logistic  │ │ Random   │ │XGBoost / │ │ LSTM │  │
│  │Regression│ │ Forest   │ │ LightGBM │ │      │  │
│  │ §3.5.1.1 │ │ §3.5.1.2 │ │ §3.5.1.3 │ │§3.5.1│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────┘  │
└────────────────────────────────────────────────────┘
        │ 5-fold OOF predictions (no label leakage)
        ▼
┌────────────────────────────────────────────────────┐
│     Level-1 Matrix Z  (5 OOF probability columns)  │
└────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────┐
│         XGBoost Meta-Model  →  p̂_final             │
│              §3.6 / Wolpert (1992)                  │
└────────────────────────────────────────────────────┘
        │
        ▼
   Risk Band: LOW / MEDIUM / HIGH
   SHAP explanation per flagged transaction
```

**PaySim-specific additions:**
- Five engineered features: `transaction_velocity`, `amount_deviation`, `balance_drop_flag`, `counterparty_spread`, `error_balance`
- LSTM reads 10-transaction sequences per account before scoring
- Three-part imbalance strategy: SMOTE-ENN + class weighting + threshold tuning

---

## Evaluation Metrics

| Metric | Why it is used |
|---|---|
| Precision | Of all flagged transactions, what fraction are truly fraudulent |
| Recall | Of all actual fraud cases, what fraction were caught — most operationally critical |
| F1-Score | Harmonic mean of Precision and Recall |
| AUC-ROC | Discrimination ability across all possible thresholds |
| MCC | Matthews Correlation Coefficient — balanced even under extreme class imbalance |
| Latency (ms) | Mean / p95 / p99 per-transaction scoring time including feature generation |

Threshold tuning is applied at prediction time (not fixed at 0.5) by sweeping 0.05–0.94 in 0.01 steps and selecting the value that maximises F1-score on the out-of-fold validation predictions.

---

## Cloud Deployment

### Google Cloud Run (GCP)

```bash
cd fraud_realtime
gcloud builds submit --tag gcr.io/YOUR_PROJECT/fraud-api
gcloud run deploy fraud-api \
  --image gcr.io/YOUR_PROJECT/fraud-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2
```

### AWS ECS Fargate

```bash
aws ecr create-repository --repository-name fraud-api
docker tag fraud-api:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/fraud-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/fraud-api:latest
# Deploy via ECS console or AWS Copilot CLI
```

### Azure Container Apps

```bash
az containerapp create \
  --name fraud-api \
  --resource-group YOUR_RG \
  --image YOUR_REGISTRY/fraud-api:latest \
  --ingress external \
  --target-port 8000 \
  --cpu 1 \
  --memory 2Gi
```

### Docker (local or any VM)

```bash
cd fraud_realtime
docker build -t fraud-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-api
```

---

## File Reference

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for all paths, seeds, hyperparameters, and thresholds. Edit dataset paths here before running anything. |
| `data_loader.py` | `load_creditcard()`, `load_ieee()`, `load_paysim()` — cleaning, encoding, scaling, and PaySim feature engineering. |
| `models.py` | `make_lr()`, `make_rf()`, `make_xgb()`, `make_lgb()`, `make_lstm()`, `make_meta()` — one constructor per model, configured to thesis hyperparameters. |
| `stacking.py` | `build_lstm_sequences()`, `generate_oof()`, `train_meta()` — the stacking ensemble implementation (Wolpert 1992). |
| `evaluate.py` | `tune_threshold()`, `evaluate()`, `cross_val_report()`, `save_confusion()`, `shap_explain()`, `measure_latency()` — all evaluation and reporting functions. |
| `train_model.py` | Main entry point. Calls all of the above in order for all three datasets. Run this. |
| `fraud_realtime/app/main.py` | FastAPI backend with REST, WebSocket, batch upload, and dashboard endpoints. |
| `fraud_realtime/static/dashboard.html` | Live monitoring dashboard served at `/dashboard`. |
| `fraud_realtime/train_and_save.py` | Trains and saves API-ready model artefacts into `fraud_realtime/models/`. |
| `fraud_realtime/stream_simulator.py` | Replays PaySim CSV or generates synthetic transactions as a live stream. |
| `fraud_realtime/Dockerfile` | Container image for cloud deployment. |

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "ModuleNotFoundError" when running scripts
**Problem**: Python can't find required packages.
**Solution**: Make sure you've installed the requirements:
```bash
pip install -r requirements.txt
```
If you get permission errors, try:
```bash
pip install -r requirements.txt --user
```

#### 2. "No module named 'sklearn'" or similar
**Problem**: scikit-learn not installed.
**Solution**: Install it specifically:
```bash
pip install scikit-learn
```

#### 3. Training takes too long or runs out of memory
**Problem**: Full PaySim dataset is too large.
**Solution**: Use sampling for testing:
```bash
python train_model.py --sample 0.01
```
Or for the API training:
```bash
cd fraud_realtime
python train_and_save.py --data ../data/PS_20174392719_1491204439457_log.csv --sample 0.1
```

#### 4. API won't start: "Model files not found"
**Problem**: Model artefacts haven't been copied to the API folder.
**Solution**: Run the copy commands from the "Running the Real-Time System" section, or use `train_and_save.py` in the `fraud_realtime` folder.

#### 5. Dashboard shows "Connection failed"
**Problem**: WebSocket connection can't be established.
**Solution**: Make sure the API is running on the correct port (8000) and that your firewall allows connections.

#### 6. "CUDA out of memory" during training
**Problem**: GPU memory insufficient for full training.
**Solution**: Use CPU-only training by setting environment variable:
```bash
export CUDA_VISIBLE_DEVICES=""
python train_model.py --sample 0.1
```

#### 7. "Permission denied" on Windows
**Problem**: File access issues.
**Solution**: Run your terminal/command prompt as Administrator, or check file permissions.

#### 8. API returns 500 error on /predict
**Problem**: Model loading failed or transaction format incorrect.
**Solution**: Check that all model files exist in `fraud_realtime/models/`, and verify your JSON matches the expected format (see API Endpoints section).

#### 9. "No such file or directory" for datasets
**Problem**: Dataset paths are incorrect.
**Solution**: Check that the CSV files exist in the `data/` folder, and update paths in `config.py` if needed.

#### 10. Training fails with "SMOTE-ENN error"
**Problem**: Sample too small for resampling.
**Solution**: Use a larger sample fraction (minimum 0.001 for PaySim).

### Getting Help

If you encounter an issue not covered here:

1. Check the terminal output for error messages
2. Verify all prerequisites are installed
3. Try running with `--sample` for faster testing
4. Check that all file paths in `config.py` are correct
5. Look at the API logs when the server is running with `--reload`

For bugs or feature requests, please open an issue on the GitHub repository.

---

## References

- Lokanan, M. E. (2023). Predicting mobile money transaction fraud using machine learning algorithms. *Applied AI Letters*, 4, e85.
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *EMSS 2016*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
- Hevner, A. R. et al. (2004). Design science in information systems research. *MIS Quarterly*, 28(1).
- Kaufman, S. et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4).
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes. *ACM SIGKDD Explorations*, 3(1).