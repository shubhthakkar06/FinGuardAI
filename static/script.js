document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('transaction-form');
    if (!form) return; // Only execute on the user dashboard page

    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const resultPanel = document.getElementById('result-panel');
    const leftResultPanel = document.getElementById('left-result-panel');

    // Result Elements
    const statusBadge = document.getElementById('tx-status-badge');
    const probBar = document.getElementById('prob-bar');
    const probValue = document.getElementById('prob-value');
    const llmMessage = document.getElementById('llm-message');
    const shapList = document.getElementById('shap-list');
    const txActions = document.getElementById('tx-actions');

    // Auto-calculate logic
    const amtInput = document.getElementById('amount');
    const oldOrgInput = document.getElementById('oldbalanceOrg');
    const newOrgInput = document.getElementById('newbalanceOrig');
    const oldDestInput = document.getElementById('oldbalanceDest');
    const newDestInput = document.getElementById('newbalanceDest');

    function updateBalances() {
        if (!newOrgInput || !newDestInput) return;
        const amt = parseFloat(amtInput?.value) || 0;
        const oldOrg = parseFloat(oldOrgInput?.value) || 0;
        const oldDest = parseFloat(oldDestInput?.value) || 0;

        newOrgInput.value = Math.max(0, oldOrg - amt).toFixed(2);
        newDestInput.value = (oldDest + amt).toFixed(2);
    }

    if (amtInput) amtInput.addEventListener('input', updateBalances);
    if (oldOrgInput) oldOrgInput.addEventListener('input', updateBalances);
    if (oldDestInput) oldDestInput.addEventListener('input', updateBalances);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI State: Loading
        submitBtn.disabled = true;
        btnText.textContent = 'Processing...';
        loader.classList.remove('hidden');
        resultPanel.classList.remove('hidden');
        if (leftResultPanel) leftResultPanel.classList.remove('hidden');

        // Reset old results visually
        probBar.style.width = '0%';
        probBar.style.backgroundColor = 'var(--text-muted)';
        statusBadge.textContent = 'Analyzing...';
        statusBadge.className = 'status-badge';
        llmMessage.textContent = 'Generating contextual analysis via GPT...';
        shapList.innerHTML = '';
        const modelList = document.getElementById('model-list');
        if (modelList) modelList.innerHTML = '';
        txActions.classList.add('hidden');

        // Capture data & ensure auto-calculated fields are in
        updateBalances();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        // For disabled/readonly inputs missing from formData in some browsers
        if (newOrgInput) data.newbalanceOrig = newOrgInput.value;
        if (newDestInput) data.newbalanceDest = newDestInput.value;

        try {
            const response = await fetch('/api/transaction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            // Try to read JSON body even on non-OK to surface server errors
            let body = null;
            try {
                body = await response.json();
            } catch (e) {
                body = null;
            }

            if (!response.ok) {
                // Prefer server-provided message when available
                const serverMsg = body && (body.message || body.error)
                    ? body.message || body.error
                    : 'A system error occurred while processing your transaction.';

                // Debug: print server body so we can see exact shape if mismatch occurs
                console.debug('Server error body:', body);

                // If it's an insufficient funds error, make it user-friendly and include amounts
                if (body && body.error && body.error.toLowerCase().includes('insufficient')) {
                    let friendly = 'Transfer amount should be less than the available balance.';
                    if (body.available_balance !== undefined && body.requested_amount !== undefined) {
                        friendly += `\nAvailable: ${body.available_balance} — Requested: ${body.requested_amount}`;
                    }
                    llmMessage.textContent = friendly;
                } else {
                    // Fallback: show server message (or the raw JSON if message missing)
                    llmMessage.textContent = serverMsg;
                    if (!body || !(body.message || body.error)) {
                        // show raw JSON for debugging
                        llmMessage.textContent = 'Server response: ' + JSON.stringify(body || {});
                    }
                }

                statusBadge.textContent = 'Error';
                statusBadge.className = 'status-badge danger';

                submitBtn.disabled = false;
                btnText.textContent = 'Retry Payment';
                loader.classList.add('hidden');
                return; // stop further processing
            }

            const result = body; // successful JSON body

            // Artificial delay for effect (Simulating complex ML + LLM latency)
            setTimeout(() => {
                updateUIWithResults(result.prediction, result.explanation);

                // UI State: Reset Button
                submitBtn.disabled = false;
                btnText.textContent = 'Send Payment';
                loader.classList.add('hidden');
            }, 1500);

        } catch (error) {
            console.error('Error:', error);
            statusBadge.textContent = 'Error';
            statusBadge.className = 'status-badge danger';
            llmMessage.textContent = 'A system error occurred while processing your transaction.';

            submitBtn.disabled = false;
            btnText.textContent = 'Retry Payment';
            loader.classList.add('hidden');
        }
    });

    function updateUIWithResults(prediction, explanation) {
        const prob = prediction.probability;
        const isFraud = prediction.is_fraud;

        // Animate progress bar
        probBar.style.width = `${prob}%`;

        // Count up animation for text
        let start = 0;
        const duration = 1000;
        const stepTime = Math.abs(Math.floor(duration / prob));

        const countInt = setInterval(() => {
            start += 1;
            probValue.textContent = start;
            if (start >= Math.floor(prob)) clearInterval(countInt);
        }, stepTime);

        // Styling based on risk level
        if (isFraud || prob >= 65) {
            probBar.style.backgroundColor = 'var(--danger)';
            llmMessage.style.borderLeftColor = 'var(--danger)';
            statusBadge.textContent = 'Fraud Alert';
            statusBadge.className = 'status-badge danger';
            txActions.classList.remove('hidden'); // Show cancel/proceed options for high risk
        } else if (prob >= 30) {
            probBar.style.backgroundColor = 'var(--warning)';
            llmMessage.style.borderLeftColor = 'var(--warning)';
            statusBadge.textContent = 'Review Required';
            statusBadge.className = 'status-badge warning';
            txActions.classList.remove('hidden'); // Optional warning actions
        } else {
            probBar.style.backgroundColor = 'var(--success)';
            llmMessage.style.borderLeftColor = 'var(--success)';
            statusBadge.textContent = 'Verified';
            statusBadge.className = 'status-badge success';
        }

        // Update Context LLM text
        llmMessage.textContent = explanation;

        // Populate Model Breakdown
        const modelList = document.getElementById('model-list');
        if (modelList && prediction.model_breakdown) {
            prediction.model_breakdown.forEach(m => {
                const li = document.createElement('li');
                const barColor = m.prob >= 65 ? 'var(--danger)' : (m.prob >= 30 ? 'var(--warning)' : 'var(--success)');
                li.innerHTML = `
                    <span style="width: 120px; font-size: 0.9em; color: var(--text-muted);">${m.name}</span>
                    <div class="shap-value-bar">
                        <div class="shap-value-fill" style="width: ${m.prob}%; background-color: ${barColor};"></div>
                    </div>
                    <span style="font-size: 0.9em; width: 40px; text-align: right;">${m.prob}%</span>
                `;
                modelList.appendChild(li);
            });
        }

        // Populate SHAP values
        prediction.shap_values.forEach((item, index) => {
            const li = document.createElement('li');
            li.style.animationDelay = `${index * 0.1}s`;
            li.className = 'shap-item';

            // Highlight top impacts
            const isTop = index < 2;
            const fillClass = isTop ? 'shap-fill-primary' : 'shap-fill-secondary';

            li.innerHTML = `
                <div class="shap-header">
                    <span class="shap-feature">${item.feature}</span>
                    <span class="shap-perc ${isTop ? 'highlight' : ''}">${item.importance}%</span>
                </div>
                <div class="shap-bar-container">
                    <div class="shap-bar-fill ${fillClass}" style="width: 0%;" data-target="${item.importance}%"></div>
                </div>
            `;
            shapList.appendChild(li);

            // Trigger animation
            setTimeout(() => {
                const fill = li.querySelector('.shap-bar-fill');
                if (fill) fill.style.width = fill.getAttribute('data-target');
            }, 50);
        });
    }

    // Handlers for action buttons (Mocking behavior)
    if (document.getElementById('cancel-btn')) {
        document.getElementById('cancel-btn').addEventListener('click', () => {
            alert('Transaction successfully cancelled.');
            location.reload();
        });
    }

    if (document.getElementById('proceed-btn')) {
        document.getElementById('proceed-btn').addEventListener('click', () => {
            if (confirm('Are you strictly sure you want to proceed despite the fraud warning?')) {
                alert('Transaction pushed through manual override review queue.');
                location.reload();
            }
        });
    }
});
