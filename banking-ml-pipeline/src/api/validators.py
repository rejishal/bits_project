# src/api/validators.py
"""Input validation for API endpoints"""

from typing import Dict, Tuple, Any
import pandas as pd

def validate_customer_data(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate input data for single customer prediction
    
    Args:
        data: Customer data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Required fields (core fields that must be provided)
    required_fields = [
        'age', 'income', 'education', 'occupation', 'marital_status',
        'account_age_months', 'avg_balance', 'num_products', 'monthly_transactions',
        'avg_transaction_amount', 'max_transaction_amount', 'credit_score',
        'existing_loans', 'loan_amount_requested', 'loan_term_months',
        'previous_defaults', 'payment_history_score'
    ]
    
    # Optional fields with defaults
    optional_fields = {
        'employment_type': 'Full-time',
        'loan_purpose': 'Personal', 
        'digital_usage_rate': 0.7,
        'max_balance': None,  # Will be calculated from avg_balance
        'dependents': 0,
        'total_relationship_value': None,  # Will be calculated
        'risk_score': None,  # Will be calculated
        'engagement_score': None  # Will be calculated
    }
    
    # Check for missing fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate data types and ranges
    try:
        # Age validation
        age = data['age']
        if not isinstance(age, (int, float)) or age < 18 or age > 100:
            return False, "Age must be between 18 and 100"
        
        # Income validation
        income = data['income']
        if not isinstance(income, (int, float)) or income <= 0:
            return False, "Income must be positive"
        
        # Credit score validation
        credit_score = data['credit_score']
        if not isinstance(credit_score, (int, float)) or credit_score < 300 or credit_score > 850:
            return False, "Credit score must be between 300 and 850"
        
        # Education validation
        valid_education = ['High School', 'Bachelor', 'Master', 'PhD']
        if data['education'] not in valid_education:
            return False, f"Education must be one of: {', '.join(valid_education)}"
        
        # Numeric fields validation
        numeric_fields = [
            'account_age_months', 'avg_balance', 'num_products', 
            'monthly_transactions', 'avg_transaction_amount', 'max_transaction_amount',
            'existing_loans', 'loan_amount_requested', 'loan_term_months'
        ]
        
        for field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False, f"{field} must be a non-negative number"
        
        # Payment history score validation
        payment_history = data['payment_history_score']
        if not isinstance(payment_history, (int, float)) or payment_history < 0 or payment_history > 1:
            return False, "Payment history score must be between 0 and 1"
        
        # Previous defaults validation
        previous_defaults = data['previous_defaults']
        if not isinstance(previous_defaults, int) or previous_defaults < 0:
            return False, "Previous defaults must be a non-negative integer"
        
    except (TypeError, ValueError) as e:
        return False, f"Invalid data type: {str(e)}"
    
    return True, "Valid"

def validate_batch_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate batch data for multiple predictions
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if dataframe is empty
    if df.empty:
        return False, "Empty dataframe provided"
    
    # Check for required columns
    required_columns = [
        'age', 'income', 'education', 'occupation', 'marital_status',
        'account_age_months', 'avg_balance', 'num_products', 'monthly_transactions',
        'avg_transaction_amount', 'max_transaction_amount', 'credit_score',
        'existing_loans', 'loan_amount_requested', 'loan_term_months',
        'previous_defaults', 'payment_history_score'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    
    # Check for null values in required columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        null_columns = null_counts[null_counts > 0].index.tolist()
        return False, f"Null values found in columns: {', '.join(null_columns)}"
    
    # Validate data ranges
    try:
        # Age validation
        if (df['age'] < 18).any() or (df['age'] > 100).any():
            return False, "Age values must be between 18 and 100"
        
        # Income validation
        if (df['income'] <= 0).any():
            return False, "Income values must be positive"
        
        # Credit score validation
        if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
            return False, "Credit score values must be between 300 and 850"
        
        # Payment history score validation
        if (df['payment_history_score'] < 0).any() or (df['payment_history_score'] > 1).any():
            return False, "Payment history score values must be between 0 and 1"
        
        # Check categorical values
        valid_education = ['High School', 'Bachelor', 'Master', 'PhD']
        invalid_education = ~df['education'].isin(valid_education)
        if invalid_education.any():
            return False, f"Invalid education values found. Must be one of: {', '.join(valid_education)}"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
    
    return True, "Valid"