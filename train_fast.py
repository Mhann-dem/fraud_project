#!/usr/bin/env python
"""
train_fast.py — Quick training on a small dataset sample for rapid testing.

This trains on 10% of PaySim data to quickly generate models for the dashboard.

Usage:
    python train_fast.py

Outputs:
    outputs/models/PaySim/   — trained models
    fraud_realtime/models/   — scaler and ensemble for the API
"""

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
import lightgbm as lgb

from config import SEED, N_FOLDS, MODEL_DIR, LSTM_WINDOW
from data_loader import load_paysim
from models import make_lr, make_rf, make_xgb, make_lgb, make_meta
from stacking import generate_oof, train_meta
from evaluate import tune_threshold, evaluate, measure_latency, print_summary

np.random.seed(SEED)

LEARNER_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


def main():
    print("\n" + "=" * 70)
    print("FAST TRAINING — PaySim 10% Sample")
    print("=" * 70)
    
    # Load data (10% sample for fast testing)
    print("\nLoading PaySim data (10% sample)...")
    X, y, _, scaler = load_paysim(sample_fraction=0.1)
    
    # Time-based split
    split = int(len(X) * 0.6)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    print(f"  Train: {len(X_tr):,} | Test: {len(X_te):,}")
    
    # Handle imbalance
    print("\nHandling class imbalance...")
    smote_enn = SMOTEENN(random_state=SEED)
    X_tr, y_tr = smote_enn.fit_resample(X_tr, y_tr)
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"  After SMOTE-ENN: {len(X_tr):,} rows")
    
    # Train base models (no OOF for speed, just direct)
    print("\nTraining base models...")
    all_metrics = {}
    
    lr_f = make_lr();  lr_f.fit(X_tr, y_tr)
    rf_f = make_rf();  rf_f.fit(X_tr, y_tr)
    
    Xtr2, Xv2, ytr2, yv2 = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
    xgb_f = make_xgb(scale_pos)
    xgb_f.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)], verbose=False)
    lgb_f = make_lgb(scale_pos)
    lgb_f.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
    
    print("  ✓ Base models trained")
    
    # Create stacking matrix from base model predictions
    print("\nCreating stacking matrix...")
    oof_probs = np.column_stack([
        lr_f.predict_proba(X_tr)[:, 1],
        rf_f.predict_proba(X_tr)[:, 1],
        xgb_f.predict_proba(X_tr)[:, 1],
        lgb_f.predict_proba(X_tr)[:, 1],
    ])
    test_probs = np.column_stack([
        lr_f.predict_proba(X_te)[:, 1],
        rf_f.predict_proba(X_te)[:, 1],
        xgb_f.predict_proba(X_te)[:, 1],
        lgb_f.predict_proba(X_te)[:, 1],
    ])
    
    # Evaluate base models
    print("\nEvaluating base models...")
    for idx, name in enumerate(LEARNER_NAMES):
        best_t = tune_threshold(y_tr, oof_probs[:, idx])
        m = evaluate(y_te, test_probs[:, idx], threshold=best_t,
                     label=f"{name} [PaySim]")
        all_metrics[name] = m
        print(f"  {name}: AUC={m['AUC-ROC']:.4f}, F1={m['F1-Score']:.4f}")
    
    # Train meta-model
    print("\nTraining meta-model...")
    meta = make_meta(scale_pos)
    Zm_tr, Zm_val, ym_tr, ym_val = train_test_split(
        oof_probs, y_tr, test_size=0.15, stratify=y_tr, random_state=SEED)
    meta.fit(Zm_tr, ym_tr, eval_set=[(Zm_val, ym_val)], verbose=False)
    final_prob = meta.predict_proba(test_probs)[:, 1]
    
    # Evaluate stacked ensemble
    best_t_meta = tune_threshold(y_tr, oof_probs[:, 0])
    m_stack = evaluate(y_te, final_prob, threshold=best_t_meta,
                       label="Stacked Ensemble [PaySim]")
    all_metrics["Stacked Ensemble"] = m_stack
    print(f"  ✓ Meta-model: AUC={m_stack['AUC-ROC']:.4f}, F1={m_stack['F1-Score']:.4f}")
    
    # Latency measurement
    print("\nMeasuring latency...")
    latency = measure_latency(meta, [lr_f, rf_f, xgb_f, lgb_f], X_te)
    all_metrics["latency"] = latency
    
    # Save models
    print("\nSaving models...")
    mdir = MODEL_DIR / "PaySim"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta,  mdir / "meta_xgboost.joblib")
    joblib.dump(lr_f,  mdir / "base_lr.joblib")
    joblib.dump(rf_f,  mdir / "base_rf.joblib")
    joblib.dump(xgb_f, mdir / "base_xgb.joblib")
    joblib.dump(lgb_f, mdir / "base_lgb.joblib")
    joblib.dump(scaler, mdir / "scaler.joblib")
    
    # Also copy to fraud_realtime for the API
    realtime_models = mdir.parent.parent / "fraud_realtime" / "models"
    realtime_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta, realtime_models / "stacked_ensemble.joblib")
    joblib.dump(scaler, realtime_models / "scaler.joblib")
    
    print(f"  ✓ Saved to: {mdir}/")
    print(f"  ✓ API models: {realtime_models}/")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS — PaySim (10% sample)")
    print("=" * 70)
    for name, m in all_metrics.items():
        if name == "latency":
            print(f"\n{name}:")
            print(f"  Mean: {m['mean_ms']:.2f} ms")
            print(f"  P95:  {m['p95_ms']:.2f} ms")
            print(f"  P99:  {m['p99_ms']:.2f} ms")
        else:
            print(f"\n{name}:")
            print(f"  AUC-ROC: {m['AUC-ROC']:.4f}")
            print(f"  F1-Score: {m['F1-Score']:.4f}")
            print(f"  Precision: {m['Precision']:.4f}")
            print(f"  Recall: {m['Recall']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Fast training complete!")
    print("=" * 70)
    print("\nNext: Start the API with the trained model")
    print("  uvicorn fraud_realtime.app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    wall = time.time()
    main()
    print(f"\nTotal time: {(time.time() - wall) / 60:.1f} minutes\n")
