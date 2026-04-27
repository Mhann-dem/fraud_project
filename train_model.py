"""
train_model.py — Main training orchestrator.

Usage:
    python train_model.py

Runs the full pipeline for all three datasets in sequence:
  Credit Card → IEEE-CIS → PaySim

Outputs saved to outputs/:
  cm_*.png              Confusion matrix plots
  shap_meta_*.png       SHAP bar plots
  models/               Saved .joblib artefacts per dataset
"""

import time
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import lightgbm as lgb

from config import SEED, TRAIN_RATIO, MODEL_DIR, RISK_LOW, RISK_HIGH
from data_loader import load_creditcard, load_ieee, load_paysim
from models import make_lr, make_rf, make_xgb, make_lgb
from stacking import generate_oof, train_meta, build_lstm_sequences
from evaluate import (tune_threshold, evaluate, cross_val_report,
                      save_confusion, shap_explain, measure_latency,
                      print_summary)

np.random.seed(SEED)
tf.random.set_seed(SEED)

LEARNER_NAMES = ["Logistic Regression", "Random Forest",
                 "XGBoost", "LightGBM", "LSTM"]


# =============================================================================
# Imbalance handling  (§3.4 / §4.4)
# =============================================================================

def handle_imbalance(X: np.ndarray,
                     y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Three-part strategy (§4.4):
      1. SMOTE-ENN on training set only.
      2. class_weight / scale_pos_weight inside model constructors.
      3. Threshold tuning at prediction time (tune_threshold).
    """
    neg, pos = (y == 0).sum(), (y == 1).sum()
    print(f"  Before SMOTE-ENN: Legit={neg:,}  Fraud={pos:,}  ratio 1:{neg//max(pos,1)}")
    smote_enn = SMOTEENN(random_state=SEED)
    X_res, y_res = smote_enn.fit_resample(X, y)
    neg_r, pos_r = (y_res == 0).sum(), (y_res == 1).sum()
    scale_pos = neg_r / max(pos_r, 1)
    print(f"  After  SMOTE-ENN: Legit={neg_r:,}  Fraud={pos_r:,}")
    return X_res, y_res, scale_pos


# =============================================================================
# Risk band preview
# =============================================================================

def risk_band(prob: float) -> str:
    if prob >= RISK_HIGH:
        return "HIGH"
    if prob >= RISK_LOW:
        return "MEDIUM"
    return "LOW"


def print_risk_preview(y_te, final_prob, tag):
    print(f"\n  Dashboard risk-band preview — {tag} (first 10 test rows)")
    print(f"  {'#':>3}  {'P(Fraud)':>10}  {'Band':>8}  {'Actual':>8}")
    print(f"  {'─'*38}")
    for i in range(min(10, len(final_prob))):
        actual = "FRAUD" if y_te[i] == 1 else "Legit"
        print(f"  {i+1:>3}  {final_prob[i]:>10.6f}  "
              f"{risk_band(final_prob[i]):>8}  {actual:>8}")


# =============================================================================
# Per-dataset pipeline
# =============================================================================

def run_pipeline(tag: str, X: np.ndarray, y: np.ndarray,
                 use_lstm: bool = False) -> dict:
    """
    End-to-end pipeline for one dataset (§3.8):
      1. Time-based 60/40 split — no shuffle to prevent future leakage.
      2. SMOTE-ENN on training split only.
      3. Optionally build LSTM sequences (PaySim only).
      4. Cross-validation report on linear baseline.
      5. Generate 5-column OOF stacking matrix Z.
      6. Evaluate each base learner column on test set.
      7. Train meta-model; evaluate stacked ensemble.
      8. SHAP explainability on meta-model.
      9. Latency measurement.
     10. Save model artefacts.
    """
    print(f"\n{'═'*60}\n  PIPELINE — {tag}\n{'═'*60}")

    # 1. Time-based split
    split = int(len(X) * TRAIN_RATIO)
    X_tr_raw, X_te = X[:split], X[split:]
    y_tr_raw, y_te = y[:split], y[split:]
    print(f"  Train: {len(X_tr_raw):,}  |  Test: {len(X_te):,}")

    # 2. Imbalance handling
    X_tr, y_tr, scale_pos = handle_imbalance(X_tr_raw, y_tr_raw)

    # 3. LSTM sequences (PaySim only)
    X_tr_seq = X_te_seq = None
    if use_lstm:
        print("  Building LSTM sequences …")
        X_tr_seq, _ = build_lstm_sequences(X_tr, y_tr)
        X_te_seq,  _ = build_lstm_sequences(X_te, y_te)

    # 4. CV on baseline
    cross_val_report(make_lr(), X_tr, y_tr, label="Logistic Regression")

    # 5. OOF stacking matrix
    print("\n  Generating OOF predictions for stacking matrix Z …")
    oof_mat, test_mat = generate_oof(
        X_tr, y_tr, X_te, scale_pos, X_tr_seq, X_te_seq
    )

    # 6. Individual base-learner evaluation
    all_metrics = {}
    for idx, name in enumerate(LEARNER_NAMES):
        if name == "LSTM" and not use_lstm:
            continue
        oof_prob  = oof_mat[:, idx]
        test_prob = test_mat[:, idx]
        best_t    = tune_threshold(y_tr, oof_prob)
        m = evaluate(y_te, test_prob, threshold=best_t,
                     label=f"{name} [{tag}]")
        all_metrics[name] = m
        save_confusion(y_te, test_prob, best_t,
                       title=f"{name} — {tag}",
                       fname=f"cm_{tag}_{name.replace(' ','_')}.png")

    # 7. Stacked ensemble
    meta, final_prob = train_meta(oof_mat, y_tr, test_mat, scale_pos)
    meta_oof_prob    = meta.predict_proba(oof_mat)[:, 1]
    best_t_meta      = tune_threshold(y_tr, meta_oof_prob)
    m_stack = evaluate(y_te, final_prob, threshold=best_t_meta,
                       label=f"Stacked Ensemble [{tag}]")
    all_metrics["Stacked Ensemble"] = m_stack
    save_confusion(y_te, final_prob, best_t_meta,
                   title=f"Stacked Ensemble — {tag}",
                   fname=f"cm_{tag}_StackedEnsemble.png")

    # 8. SHAP
    shap_explain(meta, test_mat, LEARNER_NAMES, tag)

    # 9. Latency — refit base tabular models on full training set
    lr_f  = make_lr();  lr_f.fit(X_tr, y_tr)
    rf_f  = make_rf();  rf_f.fit(X_tr, y_tr)
    Xtr2, Xv2, ytr2, yv2 = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
    xgb_f = make_xgb(scale_pos)
    xgb_f.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)], verbose=False)
    lgb_f = make_lgb(scale_pos)
    lgb_f.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
    latency = measure_latency(meta, [lr_f, rf_f, xgb_f, lgb_f], X_te)
    all_metrics["latency"] = latency

    # 10. Save artefacts
    mdir = MODEL_DIR / tag
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta,  mdir / "meta_xgboost.joblib")
    joblib.dump(lr_f,  mdir / "base_lr.joblib")
    joblib.dump(rf_f,  mdir / "base_rf.joblib")
    joblib.dump(xgb_f, mdir / "base_xgb.joblib")
    joblib.dump(lgb_f, mdir / "base_lgb.joblib")
    print(f"  → Artefacts saved to: {mdir}/")

    print_summary(all_metrics, tag)
    print_risk_preview(y_te, final_prob, tag)
    return all_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    wall = time.time()
    all_results = {}

    # Dataset 1: Credit Card
    X, y, _ = load_creditcard()
    all_results["CreditCard"] = run_pipeline("CreditCard", X, y, use_lstm=False)

    # Dataset 2: IEEE-CIS
    X, y, _ = load_ieee()
    all_results["IEEE-CIS"] = run_pipeline("IEEE-CIS", X, y, use_lstm=False)

    # Dataset 3: PaySim (primary — LSTM active)
    X, y, _, scaler = load_paysim()
    all_results["PaySim"] = run_pipeline("PaySim", X, y, use_lstm=True)
    # Save PaySim scaler for the real-time API
    joblib.dump(scaler, MODEL_DIR / "PaySim" / "scaler.joblib")
    print(f"  → PaySim scaler saved.")

    # Cross-dataset comparison (§5.6 Table 12)
    print(f"\n{'═'*60}")
    print("  §5.6  CROSS-DATASET — Stacked Ensemble (Table 12)")
    print(f"{'═'*60}")
    print(f"  {'Dataset':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'AUC':>6} {'MCC':>6}")
    print(f"  {'─'*50}")
    for ds, results in all_results.items():
        m = results.get("Stacked Ensemble", {})
        if not m:
            continue
        print(f"  {ds:<15} "
              f"{m['Precision']:>6.4f} {m['Recall']:>6.4f} "
              f"{m['F1-Score']:>6.4f} {m['AUC-ROC']:>6.4f} {m['MCC']:>6.4f}")

    print(f"\n  Total runtime: {(time.time()-wall)/60:.1f} min")
    print(f"  All outputs → outputs/")


if __name__ == "__main__":
    main()
