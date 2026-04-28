#!/usr/bin/env python
"""
bootstrap_models.py — Create minimal dummy models for rapid dashboard testing.

This creates placeholder models so you can test the API and dashboard immediately
while the real training runs in the background.

Run:
    python bootstrap_models.py
"""

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from pathlib import Path

# Create models directory
api_models_dir = Path("fraud_realtime/models")
api_models_dir.mkdir(parents=True, exist_ok=True)

print("\nBootstrapping minimal models for testing...")

# 1. Create a simple scaler
print("  Creating MinMaxScaler...")
scaler = MinMaxScaler()
# Fit on dummy range data
dummy_data = np.random.rand(1000, 16)  # 16 features for PaySim
scaler.fit(dummy_data)
joblib.dump(scaler, api_models_dir / "scaler.joblib")
print(f"  Saved: {api_models_dir / 'scaler.joblib'}")

# 2. Create a simple meta-model (trained on dummy stacking matrix)
print("  Creating XGBoost meta-model...")
X_dummy = np.random.rand(200, 4)  # 4 base model probabilities
y_dummy = np.random.randint(0, 2, 200)

meta = xgb.XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    use_label_encoder=False, verbosity=0, random_state=42
)
meta.fit(X_dummy, y_dummy)
joblib.dump(meta, api_models_dir / "stacked_ensemble.joblib")
print(f"  Saved: {api_models_dir / 'stacked_ensemble.joblib'}")

# 3. Also save to outputs/models/PaySim for consistency
output_models_dir = Path("outputs/models/PaySim")
output_models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(scaler, output_models_dir / "scaler.joblib")
joblib.dump(meta, output_models_dir / "meta_xgboost.joblib")

# Create base models
for name, filename in [
    ("Logistic Regression", "base_lr.joblib"),
    ("Random Forest", "base_rf.joblib"),
    ("XGBoost", "base_xgb.joblib"),
    ("LightGBM", "base_lgb.joblib"),
]:
    lr = LogisticRegression(random_state=42, max_iter=100)
    lr.fit(X_dummy[:, :3], y_dummy)  # Reduced features for simple models
    joblib.dump(lr, output_models_dir / filename)
    joblib.dump(lr, api_models_dir / filename)
    print(f"  Created: {filename}")

print("\n" + "=" * 70)
print("Bootstrap complete!")
print("=" * 70)
print(f"\nModels saved to:")
print(f"  API:     {api_models_dir}")
print(f"  Outputs: {output_models_dir}")
print(f"\nYou can now:")
print(f"  1. Start the API:")
print(f"     uvicorn fraud_realtime.app.main:app --host 0.0.0.0 --port 8000")
print(f"  2. Test the dashboard:")
print(f"     http://localhost:8000/dashboard")
print(f"  3. Run example_usage.py:")
print(f"     python example_usage.py")
print(f"\nThese are dummy models for testing. For production:")
print(f"  python train_fast.py     # 10% sample (5-10 minutes)")
print(f"  python train_model.py    # Full training (30+ minutes)")
print("\n")
