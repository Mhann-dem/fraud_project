"""
data_loader.py — Dataset loaders for all three fraud datasets.

Chapters 3 and 4 reference:
  §3.3  Three data sources
  §4.3  Cleaning, leakage removal, encoding, scaling
  §4.5  Five PaySim engineered features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder

from config import (DATA, MISSING_THRESH, PAYSIM_TYPES,
                    PAYSIM_FEATURE_COLS, SEED)


# =============================================================================
# Dataset 1 — Credit Card  (§3.3 / §4.3)
# =============================================================================

def load_creditcard():
    """
    Kaggle: mlg-ulb/creditcardfraud
    284,807 rows | 492 fraud | 0.173 % fraud rate

    §4.3 decisions:
    - V1–V28 already PCA-scaled — used as-is (no further transform)
    - Time → elapsed hours (temporal position without calendar anchor)
    - Amount → MinMaxScaler
    - Class → target y
    """
    print("\n[1/3] Credit Card Fraud Detection")
    df = pd.read_csv(DATA["creditcard"])
    df.drop_duplicates(inplace=True)

    df["Hour"] = df["Time"] / 3600.0
    df.drop(columns=["Time"], inplace=True)

    df["Amount"] = MinMaxScaler().fit_transform(df[["Amount"]])

    y = df.pop("Class").values
    X = df.values
    cols = df.columns.tolist()

    print(f"  Rows: {len(df):,} | Features: {X.shape[1]} "
          f"| Fraud rate: {y.mean()*100:.3f}%")
    return X, y, cols


# =============================================================================
# Dataset 2 — IEEE-CIS  (§3.3 / §4.3)
# =============================================================================

def load_ieee():
    """
    Kaggle: ieee-fraud-detection
    ~590,540 rows after merge | 3.5 % fraud rate

    §4.3 decisions:
    - Merge transaction + identity on TransactionID
    - Drop columns with > MISSING_THRESH NaN fraction
    - Numeric: median imputation
    - Categorical: 'MISSING' sentinel for low-cardinality (≤ 20 unique)
    - High-cardinality categoricals: target encoding (Micci-Barreca 2001)
    - Hour + DayOfWeek from TransactionDT
    - MinMaxScaler on all columns
    """
    print("\n[2/3] IEEE-CIS Fraud Detection")
    trans = pd.read_csv(DATA["ieee_trans"])
    ident = pd.read_csv(DATA["ieee_id"])
    df = trans.merge(ident, on="TransactionID", how="left")
    df.drop(columns=["TransactionID"], inplace=True)
    df.drop_duplicates(inplace=True)

    # Drop high-missing columns (§4.3)
    df = df.loc[:, df.isnull().mean() < MISSING_THRESH]

    # Time features before dropping TransactionDT
    if "TransactionDT" in df.columns:
        df["Hour"]      = (df["TransactionDT"] // 3600) % 24
        df["DayOfWeek"] = (df["TransactionDT"] // 86400) % 7
        df.drop(columns=["TransactionDT"], inplace=True)

    y = df.pop("isFraud").values

    cat_cols = df.select_dtypes("object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    # Impute
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING")

    # Encode: one-hot (low cardinality) vs target encode (high cardinality)
    low_card  = [c for c in cat_cols if df[c].nunique() <= 20]
    high_card = [c for c in cat_cols if df[c].nunique() >  20]
    df = pd.get_dummies(df, columns=low_card, drop_first=True)
    if high_card:
        te = TargetEncoder(cols=high_card, smoothing=10)
        df[high_card] = te.fit_transform(df[high_card], y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df)
    cols = df.columns.tolist()

    print(f"  Rows: {len(df):,} | Features: {X.shape[1]} "
          f"| Fraud rate: {y.mean()*100:.3f}%")
    return X, y, cols


# =============================================================================
# Dataset 3 — PaySim  (§3.3 / §4.3 / §4.5)
# =============================================================================

def _engineer_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    §4.5 — Five behavioural features built before identifier columns are dropped.
    Uses rolling window instead of expanding for computational efficiency on large datasets.

    (1) transaction_velocity  — rolling count (last 100) of transfers per origin account
    (2) amount_deviation      — z-score of amount vs account rolling history (last 100)
    (3) balance_drop_flag     — 1 when origin balance collapses to near-zero
    (4) counterparty_spread   — rolling distinct destinations per origin (last 100)
    (5) error_balance         — |oldOrig + amount − newOrig| recording gap
    """
    df = df.sort_values(["nameOrig", "step"]).copy()

    # (1) Rolling transaction velocity (count in last 100 transactions)
    df["transaction_velocity"] = (
        df.groupby("nameOrig")["step"]
          .transform(lambda s: s.rolling(window=100, min_periods=1).count() - 1)
          .fillna(0)
    )
    
    # (2) Amount deviation: z-score vs rolling mean/std (last 100 transactions)
    rm = df.groupby("nameOrig")["amount"].transform(
             lambda s: s.rolling(window=100, min_periods=1).mean().shift(1))
    rs = df.groupby("nameOrig")["amount"].transform(
             lambda s: s.rolling(window=100, min_periods=1).std().shift(1).fillna(1))
    df["amount_deviation"] = (df["amount"] - rm) / rs.clip(lower=1e-6)
    df["amount_deviation"] = df["amount_deviation"].fillna(0)

    # (3) Balance drop flag
    df["balance_drop_flag"] = (
        (df["newbalanceOrig"] < 1.0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)

    # (4) Counterparty spread: distinct destinations in last 100 transactions
    df["nameDest_code"] = df["nameDest"].astype("category").cat.codes
    df["counterparty_spread"] = (
        df.groupby("nameOrig")["nameDest_code"]
          .transform(lambda s: s.rolling(window=100, min_periods=1)
                               .apply(lambda x: len(np.unique(x)), raw=True)
                               .shift(1))
          .fillna(0)
    )
    df.drop(columns=["nameDest_code"], inplace=True)

    # (5) Error balance: absolute difference
    df["error_balance"] = (
        df["oldbalanceOrg"] + df["amount"] - df["newbalanceOrig"]
    ).abs()

    return df


def load_paysim(sample_fraction: float = 1.0):
    """
    Kaggle / Lopez-Rojas et al. (2016) — PaySim mobile money simulation.
    ~1.05 M rows | ~1.3 % fraud rate

    §4.3 leakage removal:
    - isFlaggedFraud  → near-zero variance, target-adjacent (Kaufman et al. 2012)
    - nameOrig / nameDest → account ID keys, not predictive features

    Args:
        sample_fraction: float in (0.005, 1.0]. Use < 1.0 for faster prototyping.
                        Recommended values:
                        - 0.01: Quick testing (10k rows, ~5 min training)
                        - 0.05: Development work (50k rows, ~15-20 min training)
                        - 1.0: Production training (1M+ rows, ~45-90 min training)
                        Minimum 0.005 required for SMOTE-ENN compatibility.

    Returns scaler alongside arrays so the API can reuse it at inference.
    """
    print(f"\n[3/3] PaySim Mobile Money Simulation")
    df = pd.read_csv(DATA["paysim"])
    df.columns = [c.strip() for c in df.columns]
    
    # For large datasets, skip drop_duplicates on read (time-intensive)
    # Just sample directly if needed
    if sample_fraction < 1.0:
        print(f"  Sampling {sample_fraction*100:.0f}% of data for fast prototyping...")
        sample_size = int(len(df) * sample_fraction)
        df = df.sample(n=sample_size, random_state=SEED)
        print(f"  Sample size: {len(df):,} rows")
    else:
        # Only drop duplicates if using full dataset
        print(f"  Removing duplicates from {len(df):,} rows...")
        df.drop_duplicates(inplace=True)
        print(f"  After dedup: {len(df):,} rows")

    # Leakage removal
    df.drop(columns=["isFlaggedFraud"], errors="ignore", inplace=True)

    # Feature engineering (must happen before dropping name cols)
    print(f"  Engineering features...")
    df = _engineer_paysim_features(df)
    df.drop(columns=["nameOrig", "nameDest"], inplace=True)

    y = df.pop("isFraud").values

    # One-hot encode type (low cardinality)
    df = pd.get_dummies(df, columns=["type"], drop_first=False)
    for t in PAYSIM_TYPES:
        if f"type_{t}" not in df.columns:
            df[f"type_{t}"] = 0

    # Enforce fixed column order for API compatibility
    df = df[PAYSIM_FEATURE_COLS]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df)

    print(f"  Rows: {len(df):,} | Features: {X.shape[1]} "
          f"| Fraud rate: {y.mean()*100:.3f}%")
    return X, y, PAYSIM_FEATURE_COLS, scaler
