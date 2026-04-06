import os
from dotenv import load_dotenv

load_dotenv()

def generate_context_explanation(transaction_data, prediction_data):
    """
    Mock OpenAI GPT API call generation based on ensemble evaluation.
    """
    is_fraud = prediction_data.get("is_fraud")
    prob = prediction_data.get("probability")
    
    amount = transaction_data.get("amount", "0")
    sender = transaction_data.get("nameOrig", "Sender")
    receiver = transaction_data.get("nameDest", "Receiver")
    
    ensemble_breakdown = ", ".join([f"{m['name']} ({m['prob']}%)" for m in prediction_data.get("model_breakdown", [])])
    
    shap_vals = prediction_data.get("shap_values", [])
    top_feature = shap_vals[0].get("feature", "Unknown") if shap_vals else "Unknown"
    second_feature = shap_vals[1].get("feature", "Unknown") if len(shap_vals) > 1 else ""
    
    explanation = f"Analyzing transaction of ₹{amount} from {sender} to {receiver}. "
    explanation += f"Ensemble Consensus: {prob}% risk. Models evaluated: {ensemble_breakdown}. "
    
    reason = f"Primary driving features were '{top_feature}'"
    if second_feature:
        reason += f" and '{second_feature}'"
        
    if is_fraud:
        return f"🚨 ALERT: High Risk Fraud Detected. {explanation} {reason}. This transaction exhibits highly suspicious patterns across the ensemble of models. The balances and amount suggest anomalous velocity or structuring. Recommendation: REJECT OR HOLD."
    elif prob > 30:
        return f"⚠️ WARNING: Elevated Risk Found. {explanation} {reason}. The transaction presents some deviations from normal baselines according to certain models. Recommendation: MANUAL REVIEW."
    else:
        return f"✅ SUCCESS: Transaction Verified. {explanation} {reason}. The ensemble consensus indicates this is a normal transaction within expected thresholds. Recommendation: APPROVE."
