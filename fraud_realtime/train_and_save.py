"""
=============================================================================
train_and_save.py
=============================================================================
Trains the stacked ensemble on PaySim and saves:
  models/stacked_ensemble.joblib  — trained XGBoost meta-model
  models/scaler.joblib            — fitted MinMaxScaler
  models/base_models.joblib       — all four base learners (for retraining)
  models/base_lr.joblib           — logistic regression base learner
  models/base_rf.joblib           — random forest base learner
  models/base_xgb.joblib          — XGBoost base learner
  models/base_lgb.joblib          — LightGBM base learner

Run once before starting the API:
    python train_and_save.py --data data/PS_20174392719_1491204439457_log.csv
=============================================================================
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SEED     = 42
N_FOLDS  = 5
OUT_DIR  = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


# =============================================================================
# Feature engineering (mirrors training pipeline)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["nameOrig", "step"]).copy()

    # 1. Transaction velocity (rolling count of prior steps)
    df["transaction_velocity"] = (
        df.groupby("nameOrig")["step"]
        .transform(lambda s: s.rolling(window=100, min_periods=1).count() - 1)
        .fillna(0)
    )

    # 2. Amount deviation (rolling z-score)
    roll_mean = df.groupby("nameOrig")["amount"].transform(
        lambda s: s.rolling(window=100, min_periods=1).mean().shift(1))
    roll_std = df.groupby("nameOrig")["amount"].transform(
        lambda s: s.rolling(window=100, min_periods=1).std().shift(1).fillna(1))
    df["amount_deviation"] = (df["amount"] - roll_mean) / roll_std.clip(lower=1e-6)
    df["amount_deviation"] = df["amount_deviation"].fillna(0)

    # 3. Balance drop flag
    df["balance_drop_flag"] = (
        (df["newbalanceOrig"] < 1.0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)

    # 4. Counterparty spread (rolling distinct destinations per origin)
    df["nameDest_code"] = df["nameDest"].astype("category").cat.codes
    df["counterparty_spread"] = (
        df.groupby("nameOrig")["nameDest_code"]
        .transform(lambda s: s.rolling(window=100, min_periods=1)
                              .apply(lambda x: len(np.unique(x)), raw=True)
                              .shift(1))
        .fillna(0)
    )
    df.drop(columns=["nameDest_code"], inplace=True)

    # 5. Error balance
    df["error_balance"] = (df["oldbalanceOrg"] + df["amount"] - df["newbalanceOrig"]).abs()

    return df


def load_and_prepare(csv_path: str, sample_fraction: float = 1.0):
    log.info(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if sample_fraction < 1.0:
        sample_size = int(len(df) * sample_fraction)
        log.info(f"Sampling {sample_fraction:.2f} fraction → {sample_size:,} rows for prototyping …")
        df = df.sample(frac=sample_fraction, random_state=SEED)

    # Drop leakage
    df.drop(columns=["isFlaggedFraud"], errors="ignore", inplace=True)

    df = engineer_features(df)
    df.drop(columns=["nameOrig", "nameDest"], inplace=True)

    y = df.pop("isFraud").values

    # One-hot encode type
    df = pd.get_dummies(df, columns=["type"], drop_first=False)
    # Ensure all type columns exist even if missing in this slice
    for t in TRANSACTION_TYPES:
        col = f"type_{t}"
        if col not in df.columns:
            df[col] = 0

    # Ensure consistent column order
    fixed_cols = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "transaction_velocity", "amount_deviation",
        "balance_drop_flag", "counterparty_spread", "error_balance",
        *[f"type_{t}" for t in TRANSACTION_TYPES],
    ]
    df = df[fixed_cols]

    log.info(f"  Rows: {len(df):,}  |  Features: {df.shape[1]}  |  Fraud rate: {y.mean()*100:.2f}%")
    return df, y


# =============================================================================
# Model builders
# =============================================================================

def build_models(scale_pos: float):
    return {
        "LogReg": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=1000, random_state=SEED),
        "RandForest": RandomForestClassifier(
            n_estimators=200, criterion="entropy", max_depth=10,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric="auc",
            early_stopping_rounds=30, random_state=SEED, verbosity=0,
            use_label_encoder=False),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, num_leaves=500, learning_rate=0.05,
            min_child_samples=50, scale_pos_weight=scale_pos,
            random_state=SEED, verbose=-1),
    }


# =============================================================================
# Stacking
# =============================================================================

def generate_oof(models: dict, X_train, y_train, X_test):
    n_models = len(models)
    oof  = np.zeros((len(X_train), n_models))
    test_preds = np.zeros((len(X_test), n_models))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for col, (name, model) in enumerate(models.items()):
        log.info(f"  OOF [{col+1}/{n_models}]: {name}")
        fold_test = np.zeros((len(X_test), N_FOLDS))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]

            if name == "XGBoost":
                model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            elif name == "LightGBM":
                model.fit(Xtr, ytr, eval_set=[(Xval, yval)])
            else:
                model.fit(Xtr, ytr)

            oof[val_idx, col]    = model.predict_proba(Xval)[:, 1]
            fold_test[:, fold]   = model.predict_proba(X_test)[:, 1]

        test_preds[:, col] = fold_test.mean(axis=1)

        # Refit on full training data for the saved base model
        if name == "XGBoost":
            # Need a small eval set for early stopping
            Xtr2, Xv2, ytr2, yv2 = train_test_split(
                X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
            model.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)], verbose=False)
        elif name == "LightGBM":
            Xtr2, Xv2, ytr2, yv2 = train_test_split(
                X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
            model.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)])
        else:
            model.fit(X_train, y_train)

    return oof, test_preds, models


def train_meta(oof, y_train, test_preds, y_test, scale_pos):
    log.info("  Training meta-model (XGBoost) …")
    Zm_tr, Zm_val, ym_tr, ym_val = train_test_split(
        oof, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    meta = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="auc",
        early_stopping_rounds=20, random_state=SEED, verbosity=0,
        use_label_encoder=False)
    meta.fit(Zm_tr, ym_tr, eval_set=[(Zm_val, ym_val)], verbose=False)

    prob = meta.predict_proba(test_preds)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    log.info(f"  Meta-model test AUC: {auc:.4f}")
    log.info("\n" + classification_report(y_test, (prob >= 0.5).astype(int),
             target_names=["Legit", "Fraud"]))
    return meta


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to PaySim CSV")
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of PaySim rows to load for faster prototyping (0 < sample <= 1.0)"
    )
    args = parser.parse_args()

    if not (0.0 < args.sample <= 1.0):
        parser.error("--sample must be greater than 0 and at most 1.0")

    df, y = load_and_prepare(args.data, sample_fraction=args.sample)

    # Time-based split (no shuffle to prevent future leakage)
    split     = int(len(df) * 0.60)
    X_train_df = df.iloc[:split]
    X_test_df  = df.iloc[split:]
    y_train    = y[:split]
    y_test     = y[split:]

    minority_count = int((y_train == 1).sum())
    if minority_count < 6:
        raise ValueError(
            f"Sample too small for SMOTE-ENN: training data has only {minority_count} fraud "
            "examples. Increase --sample or use the full dataset."
        )

    # Fit scaler on train only
    scaler = MinMaxScaler()
    scaler.feature_names_in_ = np.array(df.columns.tolist())
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    # SMOTE-ENN
    log.info("Applying SMOTE-ENN …")
    smote_enn = SMOTEENN(random_state=SEED)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    log.info(f"  Resampled: {len(X_res):,} rows")

    neg, pos = (y_res == 0).sum(), (y_res == 1).sum()
    scale_pos = neg / max(pos, 1)

    models = build_models(scale_pos)

    log.info("\nGenerating OOF predictions …")
    oof, test_preds, fitted_models = generate_oof(models, X_res, y_res, X_test)

    log.info("\nTraining stacking meta-model …")
    meta = train_meta(oof, y_res, test_preds, y_test, scale_pos)

    # Save artifacts
    joblib.dump(meta,          OUT_DIR / "stacked_ensemble.joblib")
    joblib.dump(scaler,        OUT_DIR / "scaler.joblib")
    joblib.dump(fitted_models, OUT_DIR / "base_models.joblib")
    joblib.dump(fitted_models["LogReg"],   OUT_DIR / "base_lr.joblib")
    joblib.dump(fitted_models["RandForest"], OUT_DIR / "base_rf.joblib")
    joblib.dump(fitted_models["XGBoost"],   OUT_DIR / "base_xgb.joblib")
    joblib.dump(fitted_models["LightGBM"],  OUT_DIR / "base_lgb.joblib")

    log.info(f"\nArtifacts saved to {OUT_DIR}/")
    log.info("  stacked_ensemble.joblib")
    log.info("  scaler.joblib")
    log.info("  base_models.joblib")
    log.info("  base_lr.joblib")
    log.info("  base_rf.joblib")
    log.info("  base_xgb.joblib")
    log.info("  base_lgb.joblib")


if __name__ == "__main__":
    main()
