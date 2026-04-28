"""
=============================================================================
Real-Time Fraud Detection API
=============================================================================
Endpoints:
  POST   /predict          — Score a single transaction (REST)
  POST   /predict/batch    — Score a CSV upload (batch)
  WS     /ws/stream        — WebSocket live transaction stream
  GET    /dashboard        — Serve the monitoring dashboard HTML
  GET    /health           — Health check
  GET    /metrics          — Aggregate stats (fraud rate, latency, counts)
  GET    /alerts           — Recent high-risk alerts log

Cloud deployment:
  Docker → Cloud Run (GCP) / ECS Fargate (AWS) / Container Apps (Azure)
  See Dockerfile and deploy/ folder for cloud configs.
=============================================================================
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "stacked_ensemble.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
BASE_MODELS = [BASE_DIR / "models" / name for name in [
    "base_lr.joblib", "base_rf.joblib", "base_xgb.joblib", "base_lgb.joblib"
]]
BASE_MODELS_BUNDLE = BASE_DIR / "models" / "base_models.joblib"
STATIC_DIR  = BASE_DIR / "static"

# ── Risk band thresholds ──────────────────────────────────────────────────────
THRESHOLD_LOW  = 0.30
THRESHOLD_HIGH = 0.60

# ── In-memory stores ─────────────────────────────────────────────────────────
# Ring buffer: last 500 transactions for the dashboard
RECENT_TXN: deque[dict] = deque(maxlen=500)
# Ring buffer: last 100 high-risk alerts
ALERTS: deque[dict] = deque(maxlen=100)
# Running aggregate stats
STATS: dict[str, Any] = {
    "total": 0, "fraud": 0, "low": 0, "medium": 0, "high": 0,
    "latency_ms": deque(maxlen=200),
    "started_at": datetime.now(timezone.utc).isoformat(),
}

# ── Active WebSocket connections ──────────────────────────────────────────────
WS_CLIENTS: set[WebSocket] = set()


# =============================================================================
# Model loader
# =============================================================================

class FraudScorer:
    """Thin wrapper around the trained stacked ensemble + scaler."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.base_models: list[Any] = []
        self.explainer = None
        self.base_model_names = ["base_lr", "base_rf", "base_xgb", "base_lgb"]
        self._load()

    def _load(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists() and all(p.exists() for p in BASE_MODELS):
            log.info("Loading trained stacked ensemble and base models from disk …")
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.base_models = [joblib.load(p) for p in BASE_MODELS]
            self.explainer = shap.TreeExplainer(self.model)
            log.info("Model and base learners loaded ✓")
            return

        if MODEL_PATH.exists() and SCALER_PATH.exists() and BASE_MODELS_BUNDLE.exists():
            log.info("Loading stacked ensemble and bundled base models from disk …")
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            bundle = joblib.load(BASE_MODELS_BUNDLE)
            if isinstance(bundle, dict):
                self.base_models = list(bundle.values())
            else:
                self.base_models = list(bundle)
            self.explainer = shap.TreeExplainer(self.model)
            log.info("Model and bundled base learners loaded ✓")
            return

        missing = [str(p.name) for p in [MODEL_PATH, SCALER_PATH, *BASE_MODELS, BASE_MODELS_BUNDLE] if not p.exists()]
        log.warning(
            "Incomplete model artefacts (%s) — using mock scorer. "
            "Run train_fast.py or train_and_save.py to create the full stack.",
            ", ".join(missing)
        )

    def _mock_score(self, features: np.ndarray) -> tuple[float, dict]:
        """Return a random fraud probability when no model is available (demo mode)."""
        np.random.seed(int(time.time() * 1000) % 2**31)
        p = float(np.random.beta(1.5, 8))          # skewed toward 0 (mostly legit)
        return p, {"amount": 0.40, "type_TRANSFER": 0.28, "balance_drop": 0.18,
                   "velocity": 0.09, "counterparty": 0.05}

    def score(self, features: np.ndarray) -> tuple[float, dict]:
        """
        Returns (fraud_probability, shap_top5_dict).
        features shape: (1, n_features)
        """
        if self.model is None or not self.base_models:
            return self._mock_score(features)

        scaled = self.scaler.transform(features)
        base_probs = np.column_stack([
            m.predict_proba(scaled)[:, 1] for m in self.base_models
        ])
        prob = float(self.model.predict_proba(base_probs)[0, 1])

        shap_vals = self.explainer.shap_values(base_probs)[0]
        top5_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        feature_names = self.base_model_names
        shap_dict = {feature_names[i]: round(float(shap_vals[i]), 4) for i in top5_idx}

        return prob, shap_dict


SCORER = FraudScorer()


# =============================================================================
# Pydantic schemas
# =============================================================================

class Transaction(BaseModel):
    """A single mobile money transaction."""
    transaction_id: str            = Field(default_factory=lambda: str(uuid.uuid4()))
    step:           int            = Field(...,  description="Simulation hour step")
    type:           str            = Field(...,  description="CASH_IN|CASH_OUT|TRANSFER|PAYMENT|DEBIT")
    amount:         float          = Field(...,  gt=0)
    oldbalanceOrg:  float          = Field(0.0)
    newbalanceOrig: float          = Field(0.0)
    oldbalanceDest: float          = Field(0.0)
    newbalanceDest: float          = Field(0.0)
    # Optional engineered features (computed server-side if absent)
    transaction_velocity: float    = Field(0.0)
    amount_deviation:     float    = Field(0.0)
    balance_drop_flag:    int      = Field(0)
    counterparty_spread:  float    = Field(0.0)
    error_balance:        float    = Field(0.0)


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_band:         str          # LOW | MEDIUM | HIGH
    decision:          str          # PASS | REVIEW | BLOCK
    shap_top5:         dict
    latency_ms:        float
    demo_mode:         bool
    timestamp:         str


# =============================================================================
# Helper functions
# =============================================================================

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def txn_to_features(txn: Transaction) -> np.ndarray:
    """Convert a Transaction object into a numpy feature vector."""
    type_ohe = {f"type_{t}": 0.0 for t in TRANSACTION_TYPES}
    key = f"type_{txn.type.upper()}"
    if key in type_ohe:
        type_ohe[key] = 1.0

    vec = [
        txn.step,
        txn.amount,
        txn.oldbalanceOrg,
        txn.newbalanceOrig,
        txn.oldbalanceDest,
        txn.newbalanceDest,
        txn.transaction_velocity,
        txn.amount_deviation,
        txn.balance_drop_flag,
        txn.counterparty_spread,
        txn.error_balance,
        *type_ohe.values()
    ]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def risk_band(prob: float) -> tuple[str, str]:
    if prob >= THRESHOLD_HIGH:
        return "HIGH", "BLOCK"
    if prob >= THRESHOLD_LOW:
        return "MEDIUM", "REVIEW"
    return "LOW", "PASS"


def build_response(txn_id: str, prob: float, shap5: dict, latency: float, demo_mode: bool) -> dict:
    band, decision = risk_band(prob)
    return {
        "transaction_id":   txn_id,
        "fraud_probability": round(prob, 6),
        "risk_band":         band,
        "decision":          decision,
        "shap_top5":         shap5,
        "latency_ms":        round(latency, 2),
        "demo_mode":         demo_mode,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


async def broadcast(message: dict):
    """Push a scored transaction to all connected WebSocket clients."""
    global WS_CLIENTS
    if not WS_CLIENTS:
        return
    payload = json.dumps(message)
    dead = set()
    for ws in WS_CLIENTS:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    WS_CLIENTS -= dead


def record(response: dict, txn: dict | None = None):
    """Update in-memory stats and alert log."""
    STATS["total"] += 1
    band = response["risk_band"]
    STATS[band.lower()] += 1
    if response["fraud_probability"] >= THRESHOLD_LOW:
        STATS["fraud"] += 1
    STATS["latency_ms"].append(response["latency_ms"])

    entry = {**response, "raw": txn or {}}
    RECENT_TXN.append(entry)

    if band == "HIGH":
        ALERTS.append(entry)


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="AI-powered mobile money fraud scoring with live dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (dashboard HTML is generated separately)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# REST endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": SCORER.model is not None,
        "demo_mode": SCORER.model is None,
        "uptime_since": STATS["started_at"],
    }


@app.get("/metrics")
async def metrics():
    lats = list(STATS["latency_ms"])
    return {
        "total_scored":   STATS["total"],
        "flagged":        STATS["fraud"],
        "fraud_rate_pct": round(STATS["fraud"] / max(STATS["total"], 1) * 100, 3),
        "risk_counts":    {"low": STATS["low"], "medium": STATS["medium"], "high": STATS["high"]},
        "latency_ms": {
            "mean":  round(np.mean(lats), 2) if lats else 0,
            "p95":   round(np.percentile(lats, 95), 2) if lats else 0,
            "p99":   round(np.percentile(lats, 99), 2) if lats else 0,
        },
        "demo_mode": SCORER.model is None,
    }


@app.get("/alerts")
async def get_alerts(limit: int = 20):
    return {"alerts": list(ALERTS)[-limit:][::-1]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(txn: Transaction, background_tasks: BackgroundTasks):
    """Score a single transaction and return a risk decision instantly."""
    t0 = time.perf_counter()
    features = txn_to_features(txn)
    prob, shap5 = SCORER.score(features)
    latency = (time.perf_counter() - t0) * 1000
    demo_mode = SCORER.model is None

    response = build_response(txn.transaction_id, prob, shap5, latency, demo_mode)
    background_tasks.add_task(record, response, txn.dict())
    background_tasks.add_task(broadcast, response)

    return response


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accept a CSV file upload, score every row, and return a results CSV.
    Expected CSV columns: same fields as the Transaction schema.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    results = []
    for _, row in df.iterrows():
        try:
            txn = Transaction(**row.to_dict())
        except Exception as e:
            results.append({"transaction_id": row.get("transaction_id", "?"),
                             "error": str(e)})
            continue

        t0 = time.perf_counter()
        prob, shap5 = SCORER.score(txn_to_features(txn))
        latency = (time.perf_counter() - t0) * 1000
        demo_mode = SCORER.model is None
        resp = build_response(txn.transaction_id, prob, shap5, latency, demo_mode)
        record(resp, row.to_dict())
        results.append(resp)

    out_df = pd.DataFrame(results)
    csv_bytes = out_df.to_csv(index=False).encode()
    return JSONResponse({
        "rows_scored": len(results),
        "fraud_flagged": int((out_df.get("risk_band", pd.Series()) == "HIGH").sum()),
        "preview": results[:5],
    })


# =============================================================================
# WebSocket stream endpoint
# =============================================================================

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Bidirectional WebSocket.
    - Server pushes every scored transaction as JSON.
    - Client can send {"action": "ping"} to keep the connection alive.
    """
    await ws.accept()
    WS_CLIENTS.add(ws)
    log.info(f"WebSocket connected — total clients: {len(WS_CLIENTS)}")
    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                msg  = json.loads(data)
                if msg.get("action") == "ping":
                    await ws.send_text(json.dumps({"action": "pong"}))
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await ws.send_text(json.dumps({"action": "heartbeat",
                                               "ts": datetime.now(timezone.utc).isoformat()}))
    except WebSocketDisconnect:
        WS_CLIENTS.discard(ws)
        log.info(f"WebSocket disconnected — clients remaining: {len(WS_CLIENTS)}")


# =============================================================================
# Dashboard HTML endpoint
# =============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the live monitoring dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        log.error("Dashboard file missing: %s", html_path)
        return HTMLResponse("<h2>Dashboard not found — place dashboard.html in fraud_realtime/static</h2>", status_code=404)

    try:
        return FileResponse(html_path, media_type="text/html")
    except Exception as exc:
        log.exception("Failed to serve dashboard HTML")
        return HTMLResponse(
            "<h2>Internal server error while loading dashboard.</h2>"
            "<p>Check the server logs for details.</p>",
            status_code=500,
        )
