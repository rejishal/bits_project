# test_api.py
"""Test script for the Banking ML API"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction endpoint...")
    
    # Complete customer data with all required fields
    customer_data = {
        # Demographics
        'age': 35,
        'income': 75000,
        'education': 'Master',
        'occupation': 'Professional',
        'marital_status': 'Married',
        
        # Account info
        'account_age_months': 48,
        'avg_balance': 25000,
        'num_products': 3,
        
        # Transaction behavior
        'monthly_transactions': 25,
        'avg_transaction_amount': 1500,
        'max_transaction_amount': 5000,
        
        # Credit profile
        'credit_score': 720,
        'existing_loans': 1,
        'previous_defaults': 0,
        'payment_history_score': 0.95,
        
        # Loan details
        'loan_amount_requested': 200000,
        'loan_term_months': 36
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_single_prediction_minimal():
    """Test single prediction with minimal required fields"""
    print("\nTesting single prediction with minimal fields...")
    
    # Minimal customer data (optional fields will use defaults)
    customer_data = {
        'age': 28,
        'income': 45000,
        'education': 'Bachelor',
        'occupation': 'Technical',
        'marital_status': 'Single',
        'account_age_months': 24,
        'avg_balance': 8000,
        'num_products': 2,
        'monthly_transactions': 20,
        'avg_transaction_amount': 800,
        'max_transaction_amount': 2000,
        'credit_score': 680,
        'existing_loans': 0,
        'previous_defaults': 0,
        'payment_history_score': 0.85,
        'loan_amount_requested': 50000,
        'loan_term_months': 24
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_segments():
    """Test segments endpoint"""
    print("\nTesting segments endpoint...")
    response = requests.get(f"{BASE_URL}/segments")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_feature_importance():
    """Test feature importance endpoint"""
    print("\nTesting feature importance endpoint...")
    response = requests.get(f"{BASE_URL}/feature_importance?top_n=10")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction endpoint...")
    
    # Create sample CSV data
    import pandas as pd
    import io
    
    # Sample data for 3 customers
    batch_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'age': [25, 45, 35],
        'income': [35000, 85000, 65000],
        'education': ['Bachelor', 'Master', 'Bachelor'],
        'occupation': ['Technical', 'Management', 'Professional'],
        'marital_status': ['Single', 'Married', 'Married'],
        'account_age_months': [12, 60, 36],
        'avg_balance': [5000, 45000, 20000],
        'num_products': [1, 4, 2],
        'monthly_transactions': [15, 30, 22],
        'avg_transaction_amount': [500, 2500, 1200],
        'max_transaction_amount': [1500, 8000, 4000],
        'credit_score': [620, 780, 690],
        'existing_loans': [0, 2, 1],
        'previous_defaults': [0, 0, 0],
        'payment_history_score': [0.85, 0.98, 0.92],
        'loan_amount_requested': [50000, 300000, 150000],
        'loan_term_months': [24, 48, 36]
    })
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    batch_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Send request
    files = {'file': ('batch_test.csv', csv_buffer, 'text/csv')}
    response = requests.post(f"{BASE_URL}/batch_predict", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def main():
    """Run all tests"""
    print("="*50)
    print("Banking ML API Tests")
    print("="*50)
    
    try:
        # Test all endpoints
        test_health()
        test_single_prediction()
        test_single_prediction_minimal()
        test_model_info()
        test_segments()
        test_feature_importance()
        test_batch_prediction()
        
        print("\nAll tests completed!")
        
    except requests.ConnectionError:
        print("\nERROR: Could not connect to API.")
        print("Make sure the API is running: python -m src.api.app")
    except Exception as e:
        print(f"\nERROR: {str(e)}")

if __name__ == "__main__":
    main()