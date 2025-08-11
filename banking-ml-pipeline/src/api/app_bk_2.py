from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import json
from functools import wraps
import time
from src.utils.config import config

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='logs/api_logs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Load models
try:
    preprocessor = joblib.load('models/preprocessor.pkl')
    segmenter = joblib.load('models/segmenter.pkl')
    predictor = joblib.load('models/predictor.pkl')
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise

# Performance monitoring decorator
def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Log performance metrics
        logging.info(f"Function {f.__name__} took {end_time - start_time:.3f} seconds")
        
        return result
    return decorated_function

# Input validation
def validate_input(data):
    """Validate input data"""
    required_fields = [
        'age', 'income', 'education', 'occupation', 'marital_status',
        'account_age_months', 'avg_balance', 'num_products', 'monthly_transactions',
        'avg_transaction_amount', 'max_transaction_amount', 'credit_score',
        'existing_loans', 'loan_amount_requested', 'loan_term_months',
        'previous_defaults', 'payment_history_score'
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate data types and ranges
    try:
        if not (18 <= data['age'] <= 100):
            return False, "Age must be between 18 and 100"
        
        if data['income'] <= 0:
            return False, "Income must be positive"
        
        if not (300 <= data['credit_score'] <= 850):
            return False, "Credit score must be between 300 and 850"
        
    except (TypeError, ValueError) as e:
        return False, f"Invalid data type: {str(e)}"
    
    return True, "Valid"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
@monitor_performance
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.json
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Create DataFrame
        customer_df = pd.DataFrame([data])
        
        # Feature engineering
        customer_df['debt_to_income_ratio'] = customer_df['existing_loans'] * 5000 / customer_df['income']
        customer_df['loan_to_income_ratio'] = customer_df['loan_amount_requested'] / customer_df['income']
        customer_df['avg_balance_per_product'] = customer_df['avg_balance'] / (customer_df['num_products'] + 1)
        customer_df['transaction_diversity'] = customer_df['avg_transaction_amount'] / (customer_df['max_transaction_amount'] + 1)
        customer_df['high_risk_profile'] = ((customer_df['credit_score'] < 600) | 
                                           (customer_df['previous_defaults'] > 0)).astype(int)
        
        # Preprocess
        X_preprocessed = preprocessor.transform(customer_df)
        
        # Get segment
        segment = segmenter.predict(X_preprocessed)[0]
        
        # Add segment to features
        X_with_segment = np.column_stack([X_preprocessed, [segment]])
        
        # Predict
        prediction = predictor.predict(X_with_segment)[0]
        probability = predictor.predict_proba(X_with_segment)[0, 1]
        
        # Log prediction
        logging.info(f"Prediction made - Segment: {segment}, Approved: {prediction}, Probability: {probability:.3f}")
        
        # Prepare response
        response = {
            'customer_segment': int(segment),
            'loan_approved': bool(prediction),
            'approval_probability': float(probability),
            'recommendation': get_recommendation(segment, probability),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def get_recommendation(segment, probability):
    """Get recommendation based on segment and probability"""
    if probability > 0.8:
        return "Strong candidate for loan approval"
    elif probability > 0.6:
        return "Good candidate, consider with standard terms"
    elif probability > 0.4:
        return "Moderate risk, may require additional verification"
    else:
        return "High risk, recommend alternative products or rejection"

@app.route('/batch_predict', methods=['POST'])
@monitor_performance
def batch_predict():
    """Batch prediction endpoint"""
    try:
        # Get file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Process batch
        results = []
        for idx, row in df.iterrows():
            # Convert row to dict and process
            customer_data = row.to_dict()
            
            # Add feature engineering
            customer_df = pd.DataFrame([customer_data])
            customer_df['debt_to_income_ratio'] = customer_df['existing_loans'] * 5000 / customer_df['income']
            customer_df['loan_to_income_ratio'] = customer_df['loan_amount_requested'] / customer_df['income']
            customer_df['avg_balance_per_product'] = customer_df['avg_balance'] / (customer_df['num_products'] + 1)
            customer_df['transaction_diversity'] = customer_df['avg_transaction_amount'] / (customer_df['max_transaction_amount'] + 1)
            customer_df['high_risk_profile'] = ((customer_df['credit_score'] < 600) | 
                                               (customer_df['previous_defaults'] > 0)).astype(int)
            
            # Predict
            X_preprocessed = preprocessor.transform(customer_df)
            segment = segmenter.predict(X_preprocessed)[0]
            X_with_segment = np.column_stack([X_preprocessed, [segment]])
            prediction = predictor.predict(X_with_segment)[0]
            probability = predictor.predict_proba(X_with_segment)[0, 1]
            
            results.append({
                'customer_id': idx,
                'segment': int(segment),
                'approved': bool(prediction),
                'probability': float(probability)
            })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'preprocessor': 'ColumnTransformer with StandardScaler and OneHotEncoder',
        'segmenter': 'KMeans clustering',
        'predictor': str(type(predictor).__name__),
        'last_updated': '2024-01-01'  # Update this with actual date
    })

if __name__ == '__main__':
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 5000)
    debug = config.get('api.debug', False)
    
    api_logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)