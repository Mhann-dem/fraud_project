from fastapi.testclient import TestClient
from fraud_realtime.app.main import app

client = TestClient(app)
payload = {
    'step': 1,
    'type': 'TRANSFER',
    'amount': 850000.0,
    'oldbalanceOrg': 850000.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 850000.0,
    'transaction_velocity': 8.0,
    'amount_deviation': 0.0,
    'balance_drop_flag': 1,
    'counterparty_spread': 5.0,
    'error_balance': 0.0
}
resp = client.post('/predict', json=payload)
print('status', resp.status_code)
print(resp.text)
print('headers', dict(resp.headers))
