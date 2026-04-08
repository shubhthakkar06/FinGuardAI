from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from utils.ml_stub import get_fraud_prediction, ENSEMBLE_MODELS
from utils.llm_stub import generate_context_explanation
import random
import os

app = Flask(__name__)
# Use a stable SECRET_KEY from environment in production; fall back to a random key for local dev
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

# Global in-memory list to track transactions during app runtime
transactions = []

@app.route('/')
def user_dashboard():
    """Renders the main User/Customer interface for transaction requests."""
    # Pass number of loaded models to the template so the UI can reflect actual ensemble size
    models_count = len(ENSEMBLE_MODELS) if ENSEMBLE_MODELS is not None else 0
    model_names = [m.get('name') for m in ENSEMBLE_MODELS] if ENSEMBLE_MODELS else []
    return render_template('index.html', models_count=models_count, model_names=model_names)


@app.route('/api/models')
def api_models():
    """Return a small JSON payload with loaded ensemble model names and count for debugging."""
    models = ENSEMBLE_MODELS or []
    return jsonify({
        'count': len(models),
        'models': [m.get('name') for m in models]
    })

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles admin login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handles admin logout."""
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin_dashboard():
    """Renders the Analytics & Reports monitor for admins."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
        
    # Calculate stats
    total_tx = len(transactions)
    flagged = sum(1 for tx in transactions if tx.get('prediction', {}).get('is_fraud'))
    
    # Calculate aggregate SHAP (Feature Importance)
    shap_aggregate = {}
    for tx in transactions:
        for val in tx.get('prediction', {}).get('ensemble_shap_values', []):
            f = val['feature']
            shap_aggregate[f] = shap_aggregate.get(f, 0) + val['importance']
    
    # Sort and take top items
    sorted_shap = sorted(shap_aggregate.items(), key=lambda x: x[1], reverse=True)
    total_importance = sum(abs(v) for k, v in sorted_shap) or 1
    
    # Format for chart (labels and data)
    shap_labels = [k for k, v in sorted_shap]
    shap_data = [round(v / total_importance * 100, 2) for k, v in sorted_shap]
    
    # Format for timeline chart (last 20 transactions)
    timeline_tx = transactions[-20:]
    timeline_labels = [tx['transaction_id'] for tx in timeline_tx]
    timeline_data = [tx['prediction']['probability'] for tx in timeline_tx]
    
    return render_template('admin.html', 
                           total_tx=total_tx, 
                           flagged=flagged,
                           recent_transactions=list(reversed(transactions))[-10:],
                           shap_labels=shap_labels,
                           shap_data=shap_data,
                           timeline_labels=timeline_labels,
                           timeline_data=timeline_data)

@app.route('/api/admin/feature_importance')
def feature_importance():
    """Returns global feature importance for AJAX updates if needed."""
    shap_aggregate = {}
    for tx in transactions:
        for val in tx.get('prediction', {}).get('ensemble_shap_values', []):
            f = val['feature']
            shap_aggregate[f] = shap_aggregate.get(f, 0) + val['importance']
    
    sorted_shap = sorted(shap_aggregate.items(), key=lambda x: x[1], reverse=True)
    return jsonify({
        "labels": [k for k, v in sorted_shap],
        "values": [round(v, 2) for k, v in sorted_shap]
    })

@app.route('/api/transaction', methods=['POST'])
def handle_transaction():
    """API endpoint to process a transaction and return fraud analysis."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    # Basic numeric validation for amount
    amount_raw = data.get("amount")
    try:
        amount = float(amount_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid amount value"}), 400

    # Check sender's available balance if provided (common field: oldbalanceOrg)
    old_balance_raw = data.get("oldbalanceOrg")
    old_balance = None
    if old_balance_raw is not None:
        try:
            old_balance = float(old_balance_raw)
        except (TypeError, ValueError):
            old_balance = None

    # If we have a sender balance, enforce insufficient funds check
    if old_balance is not None and amount > old_balance:
        return jsonify({
            "error": "Insufficient funds",
            "message": "Transfer amount exceeds available balance",
            "available_balance": old_balance,
            "requested_amount": amount
        }), 400

    # 1. Get ML Prediction and Explainability (SHAP)
    prediction = get_fraud_prediction(data)
    
    # 2. Get LLM Context based on AI suggestion
    explanation = generate_context_explanation(data, prediction)
    
    # Simulate a transaction ID
    tx_id = f"TXN{random.randint(1000000, 9999999)}"

    response_data = {
        "transaction_id": tx_id,
        "amount": amount,
        "recipient": data.get("nameDest", "Unknown"),
        "prediction": prediction,
        "explanation": explanation
    }

    # Store transaction for admin dashboard
    transactions.append(response_data)

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
