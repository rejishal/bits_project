# src/api/app.py
"""Flask API for the Banking ML Pipeline"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from functools import wraps
import time
from datetime import datetime
import traceback
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.integrated_pipeline import IntegratedBankingPipeline
from api.validators import validate_customer_data, validate_batch_data
from utils.logger import api_logger
from utils.config import config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the pipeline
pipeline = IntegratedBankingPipeline()
try:
    pipeline.load_pipeline(config.get('paths.models_dir', 'models'))
    api_logger.info("Pipeline loaded successfully")
except Exception as e:
    api_logger.error(f"Failed to load pipeline: {str(e)}")
    pipeline = None

# Performance monitoring decorator
def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        api_logger.info(f"{f.__name__} took {end_time - start_time:.3f} seconds")
        
        return result
    return decorated_function

# Error handler
@app.errorhandler(Exception)
def handle_error(error):
    api_logger.error(f"Unhandled exception: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

@app.route("/", methods=['POST', 'GET'])
def index():
    return "Banking ML Pipeline API is running!"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_loaded': pipeline is not None,
        'version': config.get('project.version', '1.0.0')
    })

@app.route('/predict', methods=['POST'])
@monitor_performance
def predict():
    """Single prediction endpoint"""
    try:
        # Check if pipeline is loaded
        if pipeline is None:
            return jsonify({'error': 'Pipeline not loaded'}), 503
        
        # Get and validate input data
        data = request.json
        is_valid, error_message = validate_customer_data(data)
        
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Create DataFrame
        customer_df = pd.DataFrame([data])
        
        # Make prediction
        result = pipeline.predict_new_customer(customer_df)
        
        # Log prediction
        api_logger.info(f"Prediction - Segment: {result['customer_segment']}, "
                       f"Approved: {result['loan_approved']}, "
                       f"Probability: {result['approval_probability']:.3f}")
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result), 200
        
    except Exception as e:
        api_logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
@monitor_performance
def batch_predict():
    """Batch prediction endpoint"""
    try:
        # Check if pipeline is loaded
        if pipeline is None:
            return jsonify({'error': 'Pipeline not loaded'}), 503
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate batch data
        is_valid, error_message = validate_batch_data(df)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Process batch
        results = []
        for idx, row in df.iterrows():
            customer_df = pd.DataFrame([row])
            
            try:
                prediction = pipeline.predict_new_customer(customer_df)
                results.append({
                    'row_index': idx,
                    'customer_id': row.get('customer_id', idx),
                    'segment': prediction['customer_segment'],
                    'segment_name': prediction['segment_name'],
                    'approved': prediction['loan_approved'],
                    'probability': prediction['approval_probability'],
                    'recommendation': prediction['recommendation']
                })
            except Exception as e:
                results.append({
                    'row_index': idx,
                    'customer_id': row.get('customer_id', idx),
                    'error': str(e)
                })
        
        # Summary statistics
        successful_predictions = [r for r in results if 'error' not in r]
        summary = {
            'total_records': len(df),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(df) - len(successful_predictions),
            'approval_rate': np.mean([r['approved'] for r in successful_predictions]) if successful_predictions else 0
        }
        
        return jsonify({
            'results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        api_logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if pipeline is None:
            return jsonify({'error': 'Pipeline not loaded'}), 503
        
        info = {
            'pipeline_version': pipeline.pipeline_metadata.get('version', 'Unknown'),
            'created_at': pipeline.pipeline_metadata.get('created_at', 'Unknown'),
            'segmentation': {
                'algorithm': pipeline.segmenter.algorithm,
                'n_clusters': pipeline.segmenter.optimal_clusters
            },
            'prediction': {
                'model_type': pipeline.predictor.model_type,
                'best_params': pipeline.predictor.best_params
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        api_logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/segments', methods=['GET'])
def get_segments():
    """Get information about customer segments"""
    try:
        if pipeline is None:
            return jsonify({'error': 'Pipeline not loaded'}), 503
        
        if pipeline.segmenter.segment_profiles is None:
            return jsonify({'error': 'No segment profiles available'}), 404
        
        segments = []
        for _, row in pipeline.segmenter.segment_profiles.iterrows():
            segment_info = {
                'segment_id': int(row['cluster']),
                'size': int(row['size']),
                'percentage': float(row['percentage'])
            }
            
            # Add key metrics if available
            for col in ['age_mean', 'income_mean', 'credit_score_mean', 'avg_balance_mean']:
                if col in row:
                    segment_info[col] = float(row[col])
            
            segments.append(segment_info)
        
        return jsonify({
            'segments': segments,
            'total_segments': len(segments)
        }), 200
        
    except Exception as e:
        api_logger.error(f"Get segments error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the prediction model"""
    try:
        if pipeline is None:
            return jsonify({'error': 'Pipeline not loaded'}), 503
        
        if pipeline.predictor.feature_importances is None:
            return jsonify({'error': 'No feature importance data available'}), 404
        
        # Get top N features
        top_n = request.args.get('top_n', 20, type=int)
        
        importance_data = pipeline.predictor.feature_importances.head(top_n)
        
        features = []
        for _, row in importance_data.iterrows():
            features.append({
                'feature': row['feature'],
                'importance': float(row['importance'])
            })
        
        return jsonify({
            'features': features,
            'total_features': len(pipeline.predictor.feature_importances),
            'showing_top': top_n
        }), 200
        
    except Exception as e:
        api_logger.error(f"Feature importance error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 5000)
    debug = config.get('api.debug', False)
    
    api_logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)