import sys
sys.path.append('.')
from utils.ml_stub import get_fraud_prediction
from utils.llm_stub import generate_context_explanation
import json

dummy_tx = {
    'amount': 25000,
    'nameOrig': 'Alice',
    'oldbalanceOrg': 30000,
    'newbalanceOrig': 5000,
    'nameDest': 'Bob',
    'oldbalanceDest': 0,
    'newbalanceDest': 25000
}

pred = get_fraud_prediction(dummy_tx)
print("Prediction:")
print(json.dumps(pred, indent=2))
print("---")
expl = generate_context_explanation(dummy_tx, pred)
print("LLM Explanation:")
print(expl)
