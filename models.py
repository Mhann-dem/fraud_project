"""
models.py — Base model constructors faithful to §3.5 / §4.6.

Each function returns a fresh, unfitted estimator configured with the
hyperparameters documented in Chapter 4.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from config import (SEED, LR_PARAMS, RF_PARAMS, XGB_PARAMS,
                    LGB_PARAMS, LSTM_PARAMS, META_PARAMS, LSTM_WINDOW)


# =============================================================================
# §3.5.1.1 / §4.6.1  Logistic Regression
# =============================================================================

def make_lr():
    """
    LogisticRegressionCV auto-selects C ∈ {0.01, 0.1, 1.0, 10.0}.
    L2 penalty; class_weight='balanced' for imbalance handling.
    Sigmoid output kept as soft probability for the stacking layer.
    """
    return LogisticRegressionCV(**LR_PARAMS, random_state=SEED, n_jobs=-1)


# =============================================================================
# §3.5.1.2 / §4.6.2  Random Forest
# =============================================================================

def make_rf():
    """
    200 trees, entropy criterion, max_depth=10.
    balanced_subsample re-weights each bootstrap independently.
    """
    return RandomForestClassifier(**RF_PARAMS, random_state=SEED)


# =============================================================================
# §3.5.1.3 / §4.6.3  XGBoost
# =============================================================================

def make_xgb(scale_pos: float):
    """
    500 rounds, depth 6, lr 0.05, early stopping on AUC.
    scale_pos_weight = neg/pos to handle class imbalance inside the
    boosting objective. Also used as the stacking meta-model (§3.6 / §4.7).
    """
    return xgb.XGBClassifier(**XGB_PARAMS,
                              scale_pos_weight=scale_pos,
                              random_state=SEED)


# =============================================================================
# §3.5.1.3 / §4.6.3  LightGBM
# =============================================================================

def make_lgb(scale_pos: float):
    """
    500 leaves, lr 0.05, min_child_samples=50.
    Gradient-based one-side sampling + exclusive feature bundling
    (Ke et al. 2017) — faster on large datasets than XGBoost.
    """
    return lgb.LGBMClassifier(**LGB_PARAMS,
                               scale_pos_weight=scale_pos,
                               random_state=SEED)


# =============================================================================
# §3.5.1.4 / §4.6.4  LSTM
# =============================================================================

def make_lstm(n_features: int, window: int = LSTM_WINDOW) -> tf.keras.Model:
    """
    Architecture: LSTM(64) → Dropout(0.3) → Dense(1, sigmoid).
    Adam lr=0.001, binary cross-entropy loss.

    Gate equations (implemented by TF internally):
        f_t = σ(W_f·[h_{t-1}, x_t] + b_f)      forget gate
        i_t = σ(W_i·[h_{t-1}, x_t] + b_i)      input gate
        c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)   candidate cell
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        cell state
        o_t = σ(W_o·[h_{t-1}, x_t] + b_o)      output gate
        h_t = o_t ⊙ tanh(c_t)                   hidden state
    """
    p = LSTM_PARAMS
    model = Sequential([
        LSTM(p["units"], input_shape=(window, n_features)),
        Dropout(p["dropout"]),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p["lr"]),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


# =============================================================================
# §3.6 / §4.7  XGBoost meta-model
# =============================================================================

def make_meta(scale_pos: float):
    """
    Level-2 stacking meta-model.
    Shallower than the base XGBoost (depth 4 vs 6) since its input
    is only 5 OOF probability columns, not raw features.
    """
    return xgb.XGBClassifier(**META_PARAMS,
                              scale_pos_weight=scale_pos,
                              random_state=SEED)
