"""Fast loader test — no SHAP, just verifies model loading and a single forward pass."""
import sys, os
sys.path.append('.')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.ml_stub import ENSEMBLE_MODELS, _predict_one
import numpy as np

print(f"\n{'='*50}")
print(f"Models loaded: {len(ENSEMBLE_MODELS)}")
print(f"{'='*50}")

x5 = np.array([[55000.0, 80000.0, 25000.0, 1000.0, 56000.0]])

for m in ENSEMBLE_MODELS:
    try:
        p = _predict_one(m, x5)
        print(f"[OK]  {m['name']:45s}  p={p:.4f}  ({round(p*100,2)}%)")
    except Exception as e:
        print(f"[ERR] {m['name']:45s}  {e}")

print(f"\nEnsemble would average over {len(ENSEMBLE_MODELS)} models.")
print("Done.")
