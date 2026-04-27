"""
config.py — Central configuration for the fraud detection project.
All paths, hyperparameters, and thresholds live here.
Edit DATA paths before running any script.
"""
from pathlib import Path

# ── Project root (wherever config.py lives) ───────────────────────────────────
ROOT = Path(__file__).parent

# ── Dataset paths ─────────────────────────────────────────────────────────────
# Download from Kaggle and place CSV files inside the data/ folder.
DATA = {
    "creditcard": ROOT / "data" / "creditcard.csv",
    "ieee_trans":  ROOT / "data" / "train_transaction.csv",
    "ieee_id":     ROOT / "data" / "train_identity.csv",
    "paysim":      ROOT / "data" / "PS_20174392719_1491204439457_log.csv",
}

# ── Output directories ────────────────────────────────────────────────────────
OUT_DIR   = ROOT / "outputs"          # plots, CSVs
MODEL_DIR = ROOT / "outputs" / "models"  # saved .joblib artefacts
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Train / test split (§3.8 time-based, no shuffle) ─────────────────────────
TRAIN_RATIO = 0.60

# ── Cross-validation folds (§3.8 / §5.4) ────────────────────────────────────
N_FOLDS = 5

# ── IEEE-CIS: drop columns with more than this fraction missing (§4.3) ────────
MISSING_THRESH = 0.80

# ── LSTM sequence window length — PaySim only (§4.6.4) ───────────────────────
LSTM_WINDOW = 10

# ── Risk-band thresholds for dashboard (§4.8) ────────────────────────────────
RISK_LOW  = 0.30   # below → LOW / PASS
RISK_HIGH = 0.60   # above → HIGH / BLOCK

# ── PaySim transaction type columns ──────────────────────────────────────────
PAYSIM_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# ── PaySim fixed feature order (must match API feature vector) ────────────────
PAYSIM_FEATURE_COLS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "transaction_velocity", "amount_deviation",
    "balance_drop_flag", "counterparty_spread", "error_balance",
    *[f"type_{t}" for t in PAYSIM_TYPES],
]

# ── Model hyperparameters (§4.6) ──────────────────────────────────────────────
LR_PARAMS = dict(
    Cs=[0.01, 0.1, 1.0, 10.0], cv=3, penalty="l2",
    solver="lbfgs", class_weight="balanced",
    max_iter=1000, scoring="roc_auc",
)
RF_PARAMS = dict(
    n_estimators=200, criterion="entropy", max_depth=10,
    class_weight="balanced_subsample", n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    eval_metric="auc", early_stopping_rounds=30,
    use_label_encoder=False, verbosity=0,
)
LGB_PARAMS = dict(
    n_estimators=500, num_leaves=500, learning_rate=0.05,
    min_child_samples=50, verbose=-1,
)
LSTM_PARAMS = dict(units=64, dropout=0.3, lr=0.001,
                   epochs=30, batch_size=512)

# ── Meta-model hyperparameters (§4.7) ─────────────────────────────────────────
META_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    eval_metric="auc", early_stopping_rounds=20,
    use_label_encoder=False, verbosity=0,
)
