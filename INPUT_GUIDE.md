# How to Feed Data Into the Fraud Detection System

This guide covers all ways to input data for predictions, evaluations, and dashboard monitoring.

---

## **System Architecture**

```
Your Data
    ↓
┌─────────────────────────────────────────────────────┐
│  FastAPI Real-Time API (fraud_realtime/app/main.py) │
├─────────────────────────────────────────────────────┤
│  • Single transaction REST endpoint (/predict)      │
│  • Batch CSV upload endpoint (/predict/batch)       │
│  • WebSocket live stream (/ws/stream)               │
│  • Dashboard metrics & alerts (/metrics, /alerts)   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Trained Model (fraud_realtime/models/)             │
│  • stacked_ensemble.joblib (XGBoost meta-model)     │
│  • scaler.joblib (feature normalization)            │
│  • base_models.joblib (4 base learners)             │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Outputs: Predictions, Risk Scores, Alerts          │
└─────────────────────────────────────────────────────┘
```

---

## **1. START THE SERVER**

The server is already running on `http://127.0.0.1:8000` (or `http://localhost:8000`).

If you need to restart it:
```bash
cd fraud_project
.venv\Scripts\activate.bat
uvicorn fraud_realtime.app.main:app --host 0.0.0.0 --port 8000
```

---

## **2. ACCESS THE DASHBOARD**

Open your browser and navigate to:
```
http://localhost:8000/dashboard
```

The dashboard shows:
- **Live transaction feed** (last 500 transactions)
- **Real-time metrics**: fraud rate, latency, risk distribution
- **Recent high-risk alerts** (last 100 flagged transactions)
- **WebSocket live stream**: Updates as new transactions arrive

---

## **3. SEND SINGLE TRANSACTIONS (REST)**

### **Option A: cURL (command line)**

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

### **Option B: Python**

```python
import requests
import json

url = "http://localhost:8000/predict"
transaction = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 500000,
    "oldbalanceOrg": 500000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 500000,
    "transaction_velocity": 0.0,
    "amount_deviation": 0.0,
    "balance_drop_flag": 0,
    "counterparty_spread": 0.0,
    "error_balance": 0.0
}

response = requests.post(url, json=transaction)
result = response.json()

print(f"Transaction ID: {result['transaction_id']}")
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
print(f"Risk Band: {result['risk_band']}")
print(f"Decision: {result['decision']}")
print(f"Top 5 Contributing Features:")
for feature, importance in result['shap_top5'].items():
    print(f"  {feature}: {importance:.4f}")
```

### **Option C: JavaScript (Web)**

```javascript
const transaction = {
  step: 1,
  type: "TRANSFER",
  amount: 500000,
  oldbalanceOrg: 500000,
  newbalanceOrig: 0,
  oldbalanceDest: 0,
  newbalanceDest: 500000
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(transaction)
})
  .then(res => res.json())
  .then(data => {
    console.log(`Fraud Probability: ${data.fraud_probability}`);
    console.log(`Risk Band: ${data.risk_band}`);
    console.log(`Decision: ${data.decision}`);
  });
```

### **Response Format**

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "fraud_probability": 0.847,
  "risk_band": "HIGH",
  "decision": "BLOCK",
  "shap_top5": {
    "amount": 0.312,
    "type_TRANSFER": 0.201,
    "balance_drop_flag": 0.188,
    "counterparty_spread": 0.101,
    "transaction_velocity": 0.045
  },
  "latency_ms": 14.3,
  "timestamp": "2025-04-27T12:00:00Z"
}
```

---

## **4. BATCH UPLOAD (CSV)**

### **Create a CSV file** (`transactions.csv`)

```csv
step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,transaction_velocity,amount_deviation,balance_drop_flag,counterparty_spread,error_balance
1,TRANSFER,500000,500000,0,0,500000,0.0,0.0,0,0.0,0.0
2,CASH_OUT,100000,200000,100000,0,0,1.5,0.5,1,0.2,0.0
3,PAYMENT,50000,100000,50000,0,0,0.5,0.1,0,0.1,0.0
4,DEBIT,75000,150000,75000,0,0,0.8,0.3,0,0.15,0.0
5,CASH_IN,250000,0,250000,0,0,2.0,1.0,0,0.3,0.0
```

### **Upload via cURL**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@transactions.csv"
```

### **Upload via Python**

```python
import requests

url = "http://localhost:8000/predict/batch"
with open("transactions.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    result = response.json()
    
print(f"Rows scored: {result['rows_scored']}")
print(f"Fraud flagged: {result['fraud_flagged']}")
print(f"Preview (first 5):")
for pred in result['preview']:
    print(f"  {pred['transaction_id']}: {pred['risk_band']} (prob={pred['fraud_probability']:.4f})")
```

### **Response Format**

```json
{
  "rows_scored": 5,
  "fraud_flagged": 1,
  "preview": [
    {
      "transaction_id": "txn_001",
      "fraud_probability": 0.847,
      "risk_band": "HIGH",
      "decision": "BLOCK",
      "shap_top5": {...},
      "latency_ms": 12.5,
      "timestamp": "2025-04-27T12:00:01Z"
    },
    ...
  ]
}
```

---

## **5. LIVE WEBSOCKET STREAM**

Stream real-time predictions to your app or dashboard.

### **JavaScript Client**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
  console.log('Connected to fraud detection stream');
  // Optional: Send ping to keep alive
  setInterval(() => {
    ws.send(JSON.stringify({ action: 'ping' }));
  }, 25000);
};

ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  
  if (prediction.action === 'heartbeat') {
    console.log('Heartbeat from server');
  } else {
    console.log(`Transaction ${prediction.transaction_id}:`);
    console.log(`  Risk: ${prediction.risk_band}`);
    console.log(`  Probability: ${prediction.fraud_probability}`);
    
    // Update dashboard UI
    updateDashboard(prediction);
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from stream');
};
```

### **Python Client**

```python
import asyncio
import json
import websockets

async def stream_listener():
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as ws:
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                
                if 'transaction_id' in data:
                    print(f"Transaction: {data['transaction_id']}")
                    print(f"  Risk Band: {data['risk_band']}")
                    print(f"  Probability: {data['fraud_probability']:.4f}")
                    print(f"  Decision: {data['decision']}")
            except Exception as e:
                print(f"Error: {e}")

asyncio.run(stream_listener())
```

---

## **6. STREAM SIMULATOR (for testing)**

Pre-made tool to generate realistic test transactions:

```bash
cd fraud_realtime
python stream_simulator.py --duration 60 --rate 10
```

This sends 10 transactions/second for 60 seconds to the API.

---

## **7. QUERY METRICS & ALERTS**

### **Overall Metrics** (fraud rate, latency, risk distribution)

```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "total_scored": 1250,
  "flagged": 23,
  "fraud_rate_pct": 1.84,
  "risk_counts": {
    "low": 1150,
    "medium": 77,
    "high": 23
  },
  "latency_ms": {
    "mean": 14.2,
    "p95": 28.5,
    "p99": 45.1
  }
}
```

### **High-Risk Alerts** (last 20)

```bash
curl "http://localhost:8000/alerts?limit=20"
```

---

## **8. INTERACTIVE API DOCUMENTATION**

Swagger UI with live testing:

```
http://localhost:8000/docs
```

- **Try out** each endpoint directly in the browser
- **View schemas** for all request/response objects
- **Copy example code** in multiple languages

---

## **9. TRAINING & MODEL EVALUATION**

### **Train on PaySim Data**

```bash
cd fraud_project
python fraud_realtime/train_and_save.py --data data/PS_20174392719_1491204439457_log.csv
```

Saves trained models to `fraud_realtime/models/`.

### **Evaluate Trained Models** (on test set)

```bash
cd fraud_project
python evaluate.py  # Loads trained models and shows metrics
```

Or run the full training pipeline:

```bash
python train_model.py  # Trains all datasets and generates metrics
```

---

## **10. INPUT DATA REQUIREMENTS**

### **Required Fields**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `step` | int | Simulation hour/time step | 1 |
| `type` | str | Transaction type: CASH_IN, CASH_OUT, TRANSFER, PAYMENT, DEBIT | "TRANSFER" |
| `amount` | float | Transaction amount (must be > 0) | 500000 |
| `oldbalanceOrg` | float | Sender's balance before transaction | 500000 |
| `newbalanceOrig` | float | Sender's balance after transaction | 0 |
| `oldbalanceDest` | float | Receiver's balance before transaction | 0 |
| `newbalanceDest` | float | Receiver's balance after transaction | 500000 |

### **Optional Fields** (auto-computed if missing)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transaction_velocity` | float | 0.0 | Frequency of transactions |
| `amount_deviation` | float | 0.0 | Deviation from user's avg amount |
| `balance_drop_flag` | int | 0 | Whether balance dropped significantly |
| `counterparty_spread` | float | 0.0 | Uniqueness of counterparty |
| `error_balance` | float | 0.0 | Balance calculation error |

---

## **11. RISK BANDS & DECISIONS**

| Fraud Probability | Risk Band | Decision | Action |
|-------------------|-----------|----------|--------|
| < 0.30 | LOW | PASS | Allow transaction |
| 0.30 – 0.60 | MEDIUM | REVIEW | Manual review recommended |
| ≥ 0.60 | HIGH | BLOCK | Reject transaction |

---

## **12. TROUBLESHOOTING**

### **Server not responding?**
```bash
curl http://localhost:8000/health
```

### **Models not loaded (demo mode)?**
- Check if `fraud_realtime/models/stacked_ensemble.joblib` exists
- If not, run: `python fraud_realtime/train_and_save.py --data data/PS_...csv`

### **Dashboard not loading?**
- Verify `fraud_realtime/static/dashboard.html` exists
- Check browser console for errors
- Try: `http://localhost:8000/docs` (API docs should load)

### **CSV upload failing?**
- Ensure CSV has the required columns
- Check for missing values or invalid data types
- Use the required field names exactly (case-sensitive)

---

## **QUICK START EXAMPLE**

```python
# 1. Send a single high-risk transaction
import requests

response = requests.post("http://localhost:8000/predict", json={
    "step": 1,
    "type": "TRANSFER",
    "amount": 1000000,  # Large amount
    "oldbalanceOrg": 1000000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 1000000
})

result = response.json()
print(f"Decision: {result['decision']} (probability: {result['fraud_probability']:.4f})")

# 2. Check overall metrics
metrics = requests.get("http://localhost:8000/metrics").json()
print(f"Fraud rate: {metrics['fraud_rate_pct']}%")
print(f"Mean latency: {metrics['latency_ms']['mean']} ms")

# 3. View recent alerts
alerts = requests.get("http://localhost:8000/alerts?limit=5").json()
for alert in alerts['alerts']:
    print(f"Alert: {alert['transaction_id']} - {alert['risk_band']}")
```

---

**For more details, see [fraud_realtime/README.md](fraud_realtime/README.md)**
