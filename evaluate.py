"""
evaluate.py — Evaluation helpers faithful to §3.8 / §5.3–5.7.

Functions:
  tune_threshold   — F1-maximising threshold selection
  evaluate         — full metric suite (Table 9)
  cross_val_report — 5-fold CV mean ± SD (Table 10)
  save_confusion   — confusion matrix PNG
  measure_latency  — per-transaction scoring latency (§5.7)
  shap_explain     — SHAP bar plot on meta-model (§4.8 / §5.5)
  print_summary    — formatted result table for terminal
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import N_FOLDS, SEED, OUT_DIR


# =============================================================================
# §3.8  Threshold tuning
# =============================================================================

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Sweep thresholds 0.05–0.94 in 0.01 steps.
    Return the value that maximises F1-score on the supplied arrays.
    Applied to OOF predictions so no test-set information leaks.
    """
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


# =============================================================================
# §3.8 / §5.3  Full metric suite  (Table 9)
# =============================================================================

def evaluate(y_true: np.ndarray, y_prob: np.ndarray,
             threshold: float = 0.50, label: str = "") -> dict:
    """
    Returns a dict with Accuracy, Precision, Recall, F1-Score,
    AUC-ROC, MCC, and the Threshold used.
    Prints a formatted block to stdout.
    """
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-Score":  round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "AUC-ROC":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "MCC":       round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "Threshold": threshold,
    }
    print(f"\n{'─'*56}\n  {label}\n{'─'*56}")
    for k, v in m.items():
        print(f"  {k:<12}: {v}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Legit','Fraud'])}")
    return m


# =============================================================================
# §3.8 / §5.4  Cross-validation report  (Table 10)
# =============================================================================

def cross_val_report(estimator, X: np.ndarray, y: np.ndarray,
                     label: str) -> dict:
    """
    Stratified 5-fold CV — reports mean ± SD for precision, recall, F1, AUC.
    Uses n_jobs=-1 for parallel fold execution.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}
    results = cross_validate(estimator, X, y,
                             cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n  Cross-Validation ({N_FOLDS}-fold) — {label}")
    out = {}
    for m in ["precision", "recall", "f1", "roc_auc"]:
        vals = results[f"test_{m}"]
        out[m] = {"mean": round(vals.mean(), 4), "std": round(vals.std(), 4)}
        print(f"    {m:<12}: {vals.mean():.4f} ± {vals.std():.4f}")
    return out


# =============================================================================
# §3.8  Confusion matrix plot
# =============================================================================

def save_confusion(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float, title: str, fname: str):
    """Save a heatmap confusion matrix PNG to outputs/."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Confusion matrix: {path}")


# =============================================================================
# §4.8 / §5.5  SHAP explainability  (Lundberg & Lee 2017)
# =============================================================================

def shap_explain(meta_model, test_matrix: np.ndarray,
                 learner_names: list, tag: str):
    """
    SHAP TreeExplainer on the XGBoost meta-model.
    Inputs are the 5 base-learner OOF probability columns.
    Saves a summary bar plot to outputs/.
    """
    print("\n  Computing SHAP values …")
    explainer = shap.TreeExplainer(meta_model)
    sv = explainer.shap_values(test_matrix)

    mean_abs = np.abs(sv).mean(axis=0)
    print(f"  {'Learner':<25} {'Mean |SHAP|':>12}")
    print(f"  {'─'*38}")
    for name, score in zip(learner_names, mean_abs):
        print(f"  {name:<25} {score:>12.4f}")

    plt.figure(figsize=(6, 3))
    shap.summary_plot(sv, test_matrix,
                      feature_names=learner_names,
                      plot_type="bar", show=False)
    path = OUT_DIR / f"shap_meta_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → SHAP plot: {path}")
    return sv


# =============================================================================
# §5.7  Latency measurement
# =============================================================================

def measure_latency(meta_model, base_models: list,
                    X_sample: np.ndarray, n_runs: int = 200) -> dict:
    """
    §5.7 — Full scoring latency per transaction:
    base-model inference for each learner + meta-model inference.
    Reported as mean, p95, p99 in milliseconds.
    """
    lats = []
    for i in range(n_runs):
        row = X_sample[i % len(X_sample)].reshape(1, -1)
        t0 = time.perf_counter()
        base_probs = np.array(
            [m.predict_proba(row)[0, 1] for m in base_models]
        ).reshape(1, -1)
        _ = meta_model.predict_proba(base_probs)[0, 1]
        lats.append((time.perf_counter() - t0) * 1000)

    result = {
        "mean_ms": round(float(np.mean(lats)), 2),
        "p95_ms":  round(float(np.percentile(lats, 95)), 2),
        "p99_ms":  round(float(np.percentile(lats, 99)), 2),
    }
    print(f"\n  Latency: mean={result['mean_ms']} ms  "
          f"p95={result['p95_ms']} ms  p99={result['p99_ms']} ms")
    return result


# =============================================================================
# Terminal summary table
# =============================================================================

def print_summary(all_metrics: dict, tag: str):
    print(f"\n  ── Result Summary: {tag} ──")
    hdr = f"  {'Model':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'MCC':>6}"
    print(hdr)
    print("  " + "─" * 54)
    for name, m in all_metrics.items():
        if name == "latency":
            continue
        print(f"  {name:<22} "
              f"{m['Precision']:>6.4f} "
              f"{m['Recall']:>6.4f} "
              f"{m['F1-Score']:>6.4f} "
              f"{m['AUC-ROC']:>6.4f} "
              f"{m['MCC']:>6.4f}")
