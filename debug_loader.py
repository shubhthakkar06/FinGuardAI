import sys
sys.path.append('.')
from utils.ml_stub import load_models
import traceback

print("Starting to scan models...")
models = load_models('/home/adarsh/Documents/finguard/FinGuardAI/models')
print(f"Found {len(models)} models.")
for m in models:
    print("-", m['name'], m['type'])

