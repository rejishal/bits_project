# test_installation.py
"""Simple test script to verify the installation"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Banking ML Pipeline Installation...")
print("-" * 50)

# Test imports
try:
    print("1. Testing basic imports...")
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost
    print("   ✓ Basic libraries imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("   Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Test directory structure
print("\n2. Checking directory structure...")
required_dirs = ['src', 'scripts', 'configs', 'data', 'models', 'logs']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ exists")
    else:
        print(f"   ✗ {dir_name}/ missing - creating...")
        os.makedirs(dir_name, exist_ok=True)

# Test module imports
print("\n3. Testing module imports...")
try:
    from src.data.synthetic_data_generator import create_synthetic_banking_data
    print("   ✓ Synthetic data generator imported")
    
    from src.data.data_preprocessor import DataPreprocessor
    print("   ✓ Data preprocessor imported")
    
    from src.models.customer_segmentation import CustomerSegmentation
    print("   ✓ Customer segmentation imported")
    
    from src.models.loan_predictor import LoanEligibilityPredictor
    print("   ✓ Loan predictor imported")
    
    from src.pipeline.integrated_pipeline import IntegratedBankingPipeline
    print("   ✓ Integrated pipeline imported")
    
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("   Make sure all src files are in place")
    sys.exit(1)

# Test synthetic data generation
print("\n4. Testing synthetic data generation...")
try:
    data = create_synthetic_banking_data(n_samples=100)
    print(f"   ✓ Generated synthetic data with shape: {data.shape}")
    print(f"   ✓ Columns: {list(data.columns)[:5]}... (showing first 5)")
except Exception as e:
    print(f"   ✗ Error generating data: {e}")

print("\n" + "="*50)
print("Installation test complete!")
print("You can now run: python scripts/train_model.py --synthetic --n_samples 5000")