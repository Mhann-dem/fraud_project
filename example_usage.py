#!/usr/bin/env python
"""
example_usage.py — Complete examples of feeding data into the fraud detection system.

This script demonstrates:
  1. Single transaction scoring (REST)
  2. Batch CSV upload
  3. WebSocket live stream
  4. Metrics & alerts queries
  5. Loading & evaluating the trained model locally

Run:
  python example_usage.py
"""

import requests
import json
import time
import csv
import io
from pathlib import Path
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

API_BASE = "http://localhost:8000"
ENDPOINTS = {
    "predict": f"{API_BASE}/predict",
    "batch": f"{API_BASE}/predict/batch",
    "metrics": f"{API_BASE}/metrics",
    "alerts": f"{API_BASE}/alerts",
    "health": f"{API_BASE}/health",
    "dashboard": f"{API_BASE}/dashboard",
}


# ============================================================================
# Example 1: Check API Health
# ============================================================================

def check_health():
    """Verify the API is running."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Check API Health")
    print("=" * 70)
    try:
        response = requests.get(ENDPOINTS["health"], timeout=5)
        data = response.json()
        print(f"✓ API Status: {data['status']}")
        print(f"  Model Loaded: {data['model_loaded']}")
        print(f"  Uptime Since: {data['uptime_since']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"  Is the server running? Try: uvicorn fraud_realtime.app.main:app --host 0.0.0.0 --port 8000")
        return False


# ============================================================================
# Example 2: Single Transaction Scoring
# ============================================================================

def score_single_transaction():
    """Send a single transaction and get a fraud score."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Single Transaction Scoring (REST)")
    print("=" * 70)
    
    # Create a sample transaction
    transaction = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500000.0,
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 500000.0,
    }
    
    print(f"\nSending transaction:")
    print(f"  Type: {transaction['type']}")
    print(f"  Amount: {transaction['amount']:,.0f}")
    print(f"  From balance: {transaction['oldbalanceOrg']:,.0f}")
    print(f"  To balance: {transaction['newbalanceDest']:,.0f}")
    
    try:
        response = requests.post(ENDPOINTS["predict"], json=transaction, timeout=5)
        result = response.json()
        
        print(f"\n✓ Prediction received:")
        print(f"  Transaction ID: {result['transaction_id']}")
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Risk Band: {result['risk_band']}")
        print(f"  Decision: {result['decision']}")
        print(f"  Latency: {result['latency_ms']:.2f} ms")
        print(f"  Top Contributing Features:")
        for feature, score in result['shap_top5'].items():
            print(f"    • {feature}: {score:.4f}")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# Example 3: Multiple Single Transactions
# ============================================================================

def score_multiple_transactions():
    """Send several transactions with different risk profiles."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Transactions with Different Risk Profiles")
    print("=" * 70)
    
    transactions = [
        {
            "name": "Normal transfer",
            "step": 1,
            "type": "TRANSFER",
            "amount": 100000.0,
            "oldbalanceOrg": 500000.0,
            "newbalanceOrig": 400000.0,
            "oldbalanceDest": 100000.0,
            "newbalanceDest": 200000.0,
        },
        {
            "name": "Large unusual transfer",
            "step": 2,
            "type": "TRANSFER",
            "amount": 5000000.0,  # Very large
            "oldbalanceOrg": 5000000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 5000000.0,
        },
        {
            "name": "Rapid cash-out",
            "step": 3,
            "type": "CASH_OUT",
            "amount": 750000.0,
            "oldbalanceOrg": 1000000.0,
            "newbalanceOrig": 250000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
    ]
    
    results = []
    for txn in transactions:
        name = txn.pop("name")
        print(f"\nScoring: {name}")
        try:
            response = requests.post(ENDPOINTS["predict"], json=txn, timeout=5)
            result = response.json()
            print(f"  → {result['risk_band']}: {result['fraud_probability']:.4f} "
                  f"(Decision: {result['decision']})")
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return results


# ============================================================================
# Example 4: Batch CSV Upload
# ============================================================================

def batch_upload_csv():
    """Create and upload a CSV file with multiple transactions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch CSV Upload")
    print("=" * 70)
    
    # Create sample CSV data in memory
    csv_data = """step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
1,TRANSFER,100000,500000,400000,100000,200000
2,TRANSFER,5000000,5000000,0,0,5000000
3,CASH_OUT,750000,1000000,250000,0,0
4,PAYMENT,50000,100000,50000,0,0
5,CASH_IN,250000,0,250000,0,0
6,DEBIT,75000,150000,75000,0,0"""
    
    print(f"\nPrepared CSV with 6 transactions")
    print(f"Uploading to {ENDPOINTS['batch']}")
    
    try:
        files = {"file": ("transactions.csv", csv_data, "text/csv")}
        response = requests.post(ENDPOINTS["batch"], files=files, timeout=10)
        result = response.json()
        
        print(f"\n✓ Batch scoring completed:")
        print(f"  Rows Scored: {result['rows_scored']}")
        print(f"  Fraud Flagged: {result['fraud_flagged']}")
        print(f"\n  Preview (first 3):")
        for pred in result['preview'][:3]:
            print(f"    • {pred['transaction_id']}: {pred['risk_band']} "
                  f"(prob={pred['fraud_probability']:.4f}, "
                  f"decision={pred['decision']})")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# Example 5: Query Metrics
# ============================================================================

def get_metrics():
    """Fetch overall fraud detection metrics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Query System Metrics")
    print("=" * 70)
    
    try:
        response = requests.get(ENDPOINTS["metrics"], timeout=5)
        metrics = response.json()
        
        print(f"\n✓ System Metrics:")
        print(f"  Total Scored: {metrics['total_scored']}")
        print(f"  Flagged as Fraud: {metrics['flagged']}")
        print(f"  Fraud Rate: {metrics['fraud_rate_pct']:.2f}%")
        print(f"\n  Risk Distribution:")
        for band, count in metrics['risk_counts'].items():
            pct = count / max(metrics['total_scored'], 1) * 100
            print(f"    • {band.upper()}: {count} ({pct:.1f}%)")
        print(f"\n  Latency Statistics:")
        print(f"    • Mean: {metrics['latency_ms']['mean']:.2f} ms")
        print(f"    • P95: {metrics['latency_ms']['p95']:.2f} ms")
        print(f"    • P99: {metrics['latency_ms']['p99']:.2f} ms")
        return metrics
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# Example 6: Query Recent Alerts
# ============================================================================

def get_alerts():
    """Fetch recent high-risk transactions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Retrieve High-Risk Alerts")
    print("=" * 70)
    
    try:
        response = requests.get(f"{ENDPOINTS['alerts']}?limit=10", timeout=5)
        data = response.json()
        alerts = data.get("alerts", [])
        
        if not alerts:
            print(f"\nNo alerts yet. Send some high-risk transactions first!")
            return None
        
        print(f"\n✓ Recent High-Risk Alerts ({len(alerts)} total):")
        for i, alert in enumerate(alerts[:5], 1):
            print(f"\n  {i}. Transaction: {alert['transaction_id']}")
            print(f"     Fraud Probability: {alert['fraud_probability']:.4f}")
            print(f"     Risk Band: {alert['risk_band']}")
            print(f"     Decision: {alert['decision']}")
            print(f"     Timestamp: {alert['timestamp']}")
        
        return alerts
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# Example 7: Access the Dashboard
# ============================================================================

def access_dashboard():
    """Show how to access the live dashboard."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Live Dashboard Access")
    print("=" * 70)
    print(f"\nOpen your browser and navigate to:")
    print(f"\n  {ENDPOINTS['dashboard']}")
    print(f"\nThe dashboard displays:")
    print(f"  • Live transaction feed (last 500)")
    print(f"  • Real-time metrics (fraud rate, latency, risk distribution)")
    print(f"  • High-risk alerts (last 100 flagged transactions)")
    print(f"  • WebSocket-powered live updates")


# ============================================================================
# Example 8: Evaluate Trained Model Locally
# ============================================================================

def evaluate_model_locally():
    """Load and evaluate the trained model on the test set."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Evaluate Trained Model Locally")
    print("=" * 70)
    
    try:
        import joblib
        from pathlib import Path
        
        model_path = Path("fraud_realtime/models/stacked_ensemble.joblib")
        scaler_path = Path("fraud_realtime/models/scaler.joblib")
        
        if not model_path.exists() or not scaler_path.exists():
            print(f"\n⚠ Models not found. Please train first:")
            print(f"  python fraud_realtime/train_and_save.py --data data/PS_...csv")
            return
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print(f"\n✓ Model loaded:")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Input features: {model.n_features_in_}")
        print(f"  Scaler type: {type(scaler).__name__}")
        
        # Create a sample feature vector
        import numpy as np
        sample_features = np.zeros((1, model.n_features_in_))
        
        # Make a prediction
        prob = model.predict_proba(sample_features)[0, 1]
        print(f"\n  Sample prediction (zero vector): {prob:.4f}")
        
    except ImportError:
        print(f"\n⚠ joblib not installed. Install with: pip install joblib")
    except Exception as e:
        print(f"\n✗ Error: {e}")


# ============================================================================
# Example 9: cURL Commands (for reference)
# ============================================================================

def show_curl_examples():
    """Display cURL command examples."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: cURL Commands for Testing")
    print("=" * 70)
    
    examples = [
        ("Health Check", 'curl http://localhost:8000/health'),
        ("Single Prediction", '''curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"step":1,"type":"TRANSFER","amount":500000,"oldbalanceOrg":500000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":500000}' '''),
        ("Metrics", 'curl http://localhost:8000/metrics'),
        ("Alerts", 'curl "http://localhost:8000/alerts?limit=10"'),
        ("Batch Upload", 'curl -X POST http://localhost:8000/predict/batch -F "file=@transactions.csv"'),
    ]
    
    for name, cmd in examples:
        print(f"\n{name}:")
        print(f"  {cmd}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("FRAUD DETECTION SYSTEM — DATA INPUT EXAMPLES")
    print("=" * 70)
    
    # Check if API is running
    if not check_health():
        print("\n⚠ Cannot reach the API. Please start the server first:")
        print("  cd fraud_project")
        print("  .venv\\Scripts\\activate.bat")
        print("  uvicorn fraud_realtime.app.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Run examples
    score_single_transaction()
    time.sleep(1)
    
    score_multiple_transactions()
    time.sleep(1)
    
    batch_upload_csv()
    time.sleep(1)
    
    get_metrics()
    get_alerts()
    
    access_dashboard()
    
    evaluate_model_locally()
    
    show_curl_examples()
    
    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Open the dashboard: http://localhost:8000/dashboard")
    print("  2. Send transactions using the methods above")
    print("  3. Monitor metrics and alerts in real-time")
    print("  4. Check API docs: http://localhost:8000/docs")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
