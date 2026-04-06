import sys
sys.path.append('.')
from utils.ml_stub import get_fraud_prediction
import json

dummy_tx = {
    'amount': 55000,
    'nameOrig': 'Alice',
    'oldbalanceOrg': 80000,
    'newbalanceOrig': 25000,
    'nameDest': 'Bob',
    'oldbalanceDest': 1000,
    'newbalanceDest': 56000
}

pred = get_fraud_prediction(dummy_tx)
print("--- PREDICTION OUTPUT ---")
print(json.dumps(pred, indent=2))
print("--- DONE ---")
