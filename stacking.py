"""
stacking.py — Stacking ensemble implementation (§3.6 / §4.7 / Wolpert 1992).

generate_oof  — produces the 5-column level-1 matrix Z via K-fold OOF
train_meta    — fits XGBoost meta-model on Z; returns model + test probabilities
build_lstm_sequences — reshapes flat array into (N, window, F) for LSTM
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm as lgb

from config import N_FOLDS, SEED, LSTM_WINDOW, LSTM_PARAMS
from models import make_lr, make_rf, make_xgb, make_lgb, make_lstm, make_meta


# =============================================================================
# §4.6.4  LSTM sequence builder
# =============================================================================

def build_lstm_sequences(X: np.ndarray, y: np.ndarray,
                          window: int = LSTM_WINDOW
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape (N, F) → (N, window, F).
    Rows are assumed sorted by account then step (done in load_paysim).
    Short histories are zero-padded on the left (right-aligned window).
    """
    n, f = X.shape
    X_seq = np.zeros((n, window, f), dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = X[start: i + 1]
        X_seq[i, window - len(chunk):] = chunk
    return X_seq, y


# =============================================================================
# §3.6 / §4.7  OOF stacking matrix
# =============================================================================

def generate_oof(X_train: np.ndarray, y_train: np.ndarray,
                 X_test:  np.ndarray,
                 scale_pos: float,
                 X_train_seq: np.ndarray | None = None,
                 X_test_seq:  np.ndarray | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    §3.6 — Generate the level-1 matrix Z (Wolpert 1992).

    For each of five base learners:
      1. Train on K-1 folds → predict held-out fold (OOF).
      2. Average test-set predictions across all K folds.

    OOF matrix: shape (n_train, 5) — used to train the meta-model.
    Test matrix: shape (n_test,  5) — used for final evaluation.

    No true label leaks into the meta-model because every OOF prediction
    is made on data the base learner was NOT trained on.

    Columns: 0=LR, 1=RF, 2=XGB, 3=LGB, 4=LSTM
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_tr = X_train.shape[0]
    n_te = X_test.shape[0]
    oof  = np.zeros((n_tr, 5))
    test = np.zeros((n_te, 5))

    p = LSTM_PARAMS
    learners = [
        ("Logistic Regression", 0, False),
        ("Random Forest",       1, False),
        ("XGBoost",             2, False),
        ("LightGBM",            3, False),
        ("LSTM",                4, True),
    ]

    for name, col, is_seq in learners:
        print(f"\n  ── OOF [{col+1}/5]: {name}")

        if is_seq and X_train_seq is None:
            print("    Skipped (no sequences — non-PaySim dataset)")
            oof[:, col]  = 0.5
            test[:, col] = 0.5
            continue

        fold_test = np.zeros((n_te, N_FOLDS))

        for fold_idx, (tr_idx, val_idx) in enumerate(
                skf.split(X_train, y_train)):

            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]

            if is_seq:
                Xtr_s  = X_train_seq[tr_idx]
                Xval_s = X_train_seq[val_idx]
                m = make_lstm(Xtr_s.shape[2])
                cw = {0: 1.0, 1: float(scale_pos)}
                es = EarlyStopping(monitor="val_loss", patience=5,
                                   restore_best_weights=True)
                m.fit(Xtr_s, ytr,
                      validation_data=(Xval_s, yval),
                      epochs=p["epochs"], batch_size=p["batch_size"],
                      class_weight=cw, callbacks=[es], verbose=0)
                oof[val_idx, col]      = m.predict(Xval_s, verbose=0).ravel()
                fold_test[:, fold_idx] = m.predict(X_test_seq, verbose=0).ravel()

            elif name == "Logistic Regression":
                m = make_lr(); m.fit(Xtr, ytr)
                oof[val_idx, col]      = m.predict_proba(Xval)[:, 1]
                fold_test[:, fold_idx] = m.predict_proba(X_test)[:, 1]

            elif name == "Random Forest":
                m = make_rf(); m.fit(Xtr, ytr)
                oof[val_idx, col]      = m.predict_proba(Xval)[:, 1]
                fold_test[:, fold_idx] = m.predict_proba(X_test)[:, 1]

            elif name == "XGBoost":
                m = make_xgb(scale_pos)
                m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
                oof[val_idx, col]      = m.predict_proba(Xval)[:, 1]
                fold_test[:, fold_idx] = m.predict_proba(X_test)[:, 1]

            else:  # LightGBM
                m = make_lgb(scale_pos)
                m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(-1)])
                oof[val_idx, col]      = m.predict_proba(Xval)[:, 1]
                fold_test[:, fold_idx] = m.predict_proba(X_test)[:, 1]

        test[:, col] = fold_test.mean(axis=1)

    return oof, test


# =============================================================================
# §3.6 / §4.7  Meta-model training
# =============================================================================

def train_meta(oof:   np.ndarray, y_train: np.ndarray,
               test:  np.ndarray,
               scale_pos: float):
    """
    Train XGBoost meta-model on the level-1 matrix Z.
    A 15 % holdout from the OOF matrix is used for early stopping.
    Returns the fitted meta-model and final test-set fraud probabilities.
    """
    print("\n  ── Meta-model: XGBoost (§3.6 / §4.7)")
    Zm_tr, Zm_val, ym_tr, ym_val = train_test_split(
        oof, y_train,
        test_size=0.15, stratify=y_train, random_state=SEED
    )
    meta = make_meta(scale_pos)
    meta.fit(Zm_tr, ym_tr,
             eval_set=[(Zm_val, ym_val)], verbose=False)
    final_prob = meta.predict_proba(test)[:, 1]
    return meta, final_prob
