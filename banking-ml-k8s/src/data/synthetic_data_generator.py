# src/data/synthetic_data_generator.py
"""Generate synthetic banking data for testing and demonstration"""

import numpy as np
import pandas as pd
from typing import Optional

def create_synthetic_banking_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Create synthetic banking data with realistic distributions
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic banking data
    """
    np.random.seed(random_state)
    print(f"Creating synthetic data with {n_samples} samples")
    
    # Demographics
    data = pd.DataFrame()
    
    # Age (18-80, beta distribution skewed towards working age)
    data['age'] = np.random.beta(5, 2, n_samples) * 62 + 18
    data['age'] = data['age'].astype(int)
    
  

    # Education
    education_choices = []
    for age in data['age']:
        if age < 25:
            probs = [0.4, 0.5, 0.09, 0.01]
        elif age < 35:
            probs = [0.25, 0.4, 0.3, 0.05]
        elif age < 50:
            probs = [0.3, 0.35, 0.28, 0.07]
        else:
            probs = [0.35, 0.35, 0.25, 0.05]
        education_choices.append(np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], p=probs))
    data['education'] = education_choices
    
    # Occupation (based on education)
    occupation_map = {
        'High School': ['Service', 'Sales', 'Manual', 'Clerical'],
        'Bachelor': ['Professional', 'Management', 'Technical', 'Sales'],
        'Master': ['Professional', 'Management', 'Executive', 'Consultant'],
        'PhD': ['Research', 'Executive', 'Professional', 'Academic']
    }
    data['occupation'] = data['education'].apply(lambda x: np.random.choice(occupation_map[x]))
    

    # Income (correlated with age and occupation)
    base_income = 25000
    age_factor = (data['age'] - 18) / 62 * 50000

    # Create occupation-based income multipliers
    occupation_income_multipliers = {
        'Executive': 8.0,
        'Management': 4.0,
        'Professional': 3.0,
        'Consultant': 5.0,
        'Academic': 2.0,
        'Research': 3.0,
        'Technical': 3.0,
        'Sales': 1.2,
        'Clerical': 0.9,
        'Service': 0.8,
        'Manual': 0.7
    }

    # Apply occupation multiplier
    occupation_factor = data['occupation'].map(occupation_income_multipliers).fillna(1.0)
    
    # Generate income with all factors
    income_noise = np.random.lognormal(10.5, 0.6, n_samples)
    data['income'] = (base_income + age_factor) * occupation_factor + income_noise
    data['income'] = data['income'].clip(15000, 500000).astype(int)
    

    # Employment type
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Retired', 'Student']
    employment_probs = []
    for age in data['age']:
        if age < 25:
            probs = [0.3, 0.3, 0.1, 0, 0.3]
        elif age < 65:
            probs = [0.7, 0.1, 0.2, 0, 0]
        else:
            probs = [0.1, 0.1, 0.1, 0.7, 0]
        employment_probs.append(probs)
    data['employment_type'] = [np.random.choice(employment_types, p=probs) for probs in employment_probs]
    
    # Marital status
    marital_choices = []
    for age in data['age']:
        if age < 25:
            probs = [0.8, 0.15, 0.05]
        elif age < 40:
            probs = [0.3, 0.6, 0.1]
        else:
            probs = [0.2, 0.65, 0.15]
        marital_choices.append(np.random.choice(['Single', 'Married', 'Divorced'], p=probs))
    data['marital_status'] = marital_choices
    
    # Dependents
    dependents = []
    for i, row in data.iterrows():
        if row['marital_status'] == 'Single':
            n_deps = np.random.choice([0, 1], p=[0.9, 0.1])
        elif row['marital_status'] == 'Married':
            if row['age'] < 30:
                n_deps = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            else:
                n_deps = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        else:
            n_deps = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        dependents.append(n_deps)
    data['dependents'] = dependents
    
    # Account characteristics
    data['account_age_months'] = np.random.gamma(2, 15, n_samples) * (1 + data['age'] / 100)
    data['account_age_months'] = data['account_age_months'].clip(1, 360).astype(int)
    
    # Number of products
    base_products = 1
    income_factor = (data['income'] / 100000).clip(0, 3)
    age_factor = (data['account_age_months'] / 120).clip(0, 2)
    data['num_products'] = base_products + np.random.poisson(income_factor + age_factor)
    data['num_products'] = data['num_products'].clip(1, 8)
    
    # Balances
    balance_ratio = np.random.beta(2, 5, n_samples) * 0.5
    monthly_income = data['income'] / 12
    data['avg_balance'] = monthly_income * balance_ratio * np.random.uniform(0.5, 3, n_samples)
    data['avg_balance'] = data['avg_balance'].clip(100, None).astype(int)
    data['max_balance'] = data['avg_balance'] * np.random.uniform(1.2, 3, n_samples)
    data['max_balance'] = data['max_balance'].astype(int)
    
    # Transaction behavior
    lifestyle_factor = data['num_products'] * 5
    age_factor = 20 - np.abs(data['age'] - 35) / 3
    data['monthly_transactions'] = np.random.poisson(lifestyle_factor + age_factor)
    data['monthly_transactions'] = data['monthly_transactions'].clip(5, 150)
    
    data['avg_transaction_amount'] = (data['avg_balance'] / data['monthly_transactions'] * 
                                     np.random.uniform(0.5, 2, n_samples))
    data['avg_transaction_amount'] = data['avg_transaction_amount'].astype(int)
    
    data['max_transaction_amount'] = (data['avg_transaction_amount'] * 
                                     np.random.lognormal(1, 0.5, n_samples))
    data['max_transaction_amount'] = data['max_transaction_amount'].astype(int)
    
    # Digital usage
    data['digital_usage_rate'] = 0.9 - (data['age'] - 18) / 100 + np.random.normal(0, 0.1, n_samples)
    data['digital_usage_rate'] = data['digital_usage_rate'].clip(0.1, 1.0)
    
    # Credit profile
    base_score = 650
    income_factor = (data['income'] / 1000).clip(0, 100)
    age_factor = (data['age'] - 18).clip(0, 30)
    account_age_factor = (data['account_age_months'] / 12).clip(0, 20)
    random_factor = np.random.normal(0, 50, n_samples)
    
    data['credit_score'] = (base_score + income_factor + age_factor + 
                           account_age_factor + random_factor)
    data['credit_score'] = data['credit_score'].clip(300, 850).astype(int)
    
    # Existing loans
    loan_probability = (data['income'] / 200000).clip(0, 0.8)
    data['existing_loans'] = np.random.binomial(4, loan_probability)
    
    # Previous defaults
    default_prob = 1 - (data['credit_score'] - 300) / 550
    default_prob = default_prob * 0.3
    data['previous_defaults'] = np.random.binomial(2, default_prob)
    
    # Payment history
    data['payment_history_score'] = (0.5 + (data['credit_score'] - 300) / 1100 + 
                                    np.random.normal(0, 0.1, n_samples))
    data['payment_history_score'] = data['payment_history_score'].clip(0, 1)
    
    # Loan application details
    data['loan_amount_requested'] = data['income'] * np.random.uniform(0.5, 5, n_samples)
    data['loan_amount_requested'] = data['loan_amount_requested'].astype(int)
    
    data['loan_term_months'] = np.random.choice([12, 24, 36, 48, 60, 72, 84], n_samples)
    
    loan_purposes = ['Home', 'Auto', 'Personal', 'Education', 'Business', 'Debt Consolidation']
    data['loan_purpose'] = np.random.choice(loan_purposes, n_samples)
    
    # Additional behavioral features
    data['total_relationship_value'] = (data['avg_balance'] * 0.01 + 
                                       data['monthly_transactions'] * 10 +
                                       data['num_products'] * 1000)
    
    data['risk_score'] = (data['previous_defaults'] * 50 + 
                         (850 - data['credit_score']) / 10 +
                         data['existing_loans'] * 5)
    
    data['engagement_score'] = (data['digital_usage_rate'] * 40 +
                               data['monthly_transactions'] / 3 +
                               data['num_products'] * 5)
    
    # Add customer ID
    data['customer_id'] = ['CUST' + str(i).zfill(6) for i in range(1, n_samples + 1)]
    
        # Loan approval logic (add this before reordering columns)
    approval_score = (
        (data['credit_score'] > 650).astype(int) * 0.3 +
        (data['income'] > 50000).astype(int) * 0.2 +
        (data['previous_defaults'] == 0).astype(int) * 0.3 +
        (data['payment_history_score'] > 0.7).astype(int) * 0.2
    )
    data['loan_approved'] = (approval_score > 0.5).astype(int)

    # Reorder columns (add 'loan_approved' at the end)
    column_order = ['customer_id', 'age', 'income', 'education', 'occupation', 'employment_type',
                   'marital_status', 'dependents', 'account_age_months', 'num_products',
                   'avg_balance', 'max_balance', 'monthly_transactions', 'avg_transaction_amount',
                   'max_transaction_amount', 'digital_usage_rate', 'credit_score', 'existing_loans',
                   'previous_defaults', 'payment_history_score', 'loan_amount_requested',
                   'loan_term_months', 'loan_purpose', 'total_relationship_value',
                   'risk_score', 'engagement_score', 'loan_approved']

    data = data[column_order]
    
    print(f"Synthetic data created successfully with shape {data.shape}")
    
    return data