# FraudSentinel — Real-Time Fraud Detection System

AI-powered mobile money fraud scoring with a live browser dashboard,
REST API, WebSocket stream, and CSV batch processing.

```
fraud_realtime/
├── app/
│   └── main.py              ← FastAPI backend (REST + WebSocket + dashboard)
├── static/
│   └── dashboard.html       ← Live monitoring dashboard (served at /dashboard)
├── models/                  ← Saved model artifacts (created by train_and_save.py)
├── train_and_save.py        ← Train stacked ensemble and save artifacts
├── stream_simulator.py      ← Live transaction stream simulator
├── Dockerfile               ← Container image
└── requirements.txt
```

---

## 1. Install

```bash
pip install -r requirements.txt
```

---

## 2. Train the model

```bash
python train_and_save.py --data /path/to/PS_20174392719_log.csv
```

Saves three files to `models/`:
- `stacked_ensemble.joblib`  — XGBoost meta-model
- `scaler.joblib`            — fitted MinMaxScaler
- `base_models.joblib`       — all four base learners

> **No model yet?** The API runs in **demo mode** (mock scores) until you train.

---

## 3. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000/dashboard` or `http://127.0.0.1:8000/dashboard` | Live browser dashboard |
| `http://localhost:8000/docs` or `http://127.0.0.1:8000/docs`      | Interactive API docs (Swagger) |
| `http://localhost:8000/predict` or `http://127.0.0.1:8000/predict`   | POST a single transaction |
| `http://localhost:8000/predict/batch` | POST a CSV file |
| `ws://localhost:8000/ws/stream`   | WebSocket live feed |
| `http://localhost:8000/metrics`   | Aggregate fraud stats |
| `http://localhost:8000/alerts`    | Recent high-risk alerts |
| `http://localhost:8000/health`    | Health check |

---

## 4. Send transactions

**Single transaction (REST):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 500000,
    "oldbalanceOrg": 500000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 500000
  }'
```

**Response:**
```json
{
  "transaction_id": "abc123...",
  "fraud_probability": 0.847,
  "risk_band": "HIGH",
  "decision": "BLOCK",
  "shap_top5": {
    "amount": 0.312,
    "type_TRANSFER": 0.201,
    "balance_drop_flag": 0.188
  },
  "latency_ms": 14.3,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**CSV batch upload:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@transactions.csv"
```

**Live stream simulator:**
```bash
# Demo mode (synthetic random transactions at 5 tx/s)
python stream_simulator.py --demo --rate 5

# Replay a PaySim CSV at 20 tx/s
python stream_simulator.py --data data/PS_log.csv --rate 20

# REST + WebSocket simultaneously
python stream_simulator.py --demo --mode both --rate 10
```

---

## 5. Risk bands

| Band | P(Fraud) | Action |
|------|----------|--------|
| LOW | < 0.30 | PASS — transaction clears automatically |
| MEDIUM | 0.30 – 0.60 | REVIEW — queued for analyst review |
| HIGH | > 0.60 | BLOCK — immediate alert raised |

---

## 6. Cloud deployment

### Google Cloud Run (GCP)
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/fraud-api
gcloud run deploy fraud-api \
  --image gcr.io/YOUR_PROJECT/fraud-api \
  --platform managed --region us-central1 \
  --allow-unauthenticated --port 8000 \
  --memory 2Gi --cpu 2
```

### AWS ECS Fargate
```bash
aws ecr create-repository --repository-name fraud-api
docker tag fraud-api:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/fraud-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/fraud-api:latest
# Then deploy via ECS console or Copilot CLI
```

### Azure Container Apps
```bash
az containerapp create \
  --name fraud-api \
  --resource-group YOUR_RG \
  --image YOUR_REGISTRY/fraud-api:latest \
  --ingress external --target-port 8000 \
  --cpu 1 --memory 2Gi
```

### Docker (local / any VM)
```bash
docker build -t fraud-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-api
```

---

## 7. WebSocket dashboard connection

The dashboard at `/dashboard` connects automatically via WebSocket.
Every scored transaction is pushed to all connected browsers in real time.

To embed the WebSocket feed in your own app:
```javascript
const ws = new WebSocket('wss://YOUR_API_HOST/ws/stream');
ws.onmessage = (event) => {
  const txn = JSON.parse(event.data);
  console.log(txn.risk_band, txn.fraud_probability);
};
```
