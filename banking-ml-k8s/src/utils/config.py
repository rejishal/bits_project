# src/utils/config.py
"""Configuration management for the Banking ML Pipeline"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
import json

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or 'configs/model_config.yaml'
        self.config = self._load_config()
        self._set_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path and os.path.exists(self.config_path):
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        
        # Return default configuration if file doesn't exist
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'project': {
                'name': 'Banking ML Pipeline',
                'version': '1.0.0'
            },
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'logs_dir': 'logs',
                'output_dir': 'outputs'
            },
            'preprocessing': {
                'numeric_features': [
                    'age', 'income', 'account_age_months', 'num_products', 
                    'avg_balance', 'max_balance', 'monthly_transactions',
                    'avg_transaction_amount', 'max_transaction_amount',
                    'digital_usage_rate', 'credit_score', 'existing_loans',
                    'previous_defaults', 'payment_history_score',
                    'loan_amount_requested', 'loan_term_months',
                    'total_relationship_value', 'risk_score', 'engagement_score'
                ],
                'categorical_features': [
                    'education', 'occupation', 'employment_type', 
                    'marital_status', 'loan_purpose'
                ],
                'scaling_method': 'robust',
                'encoding_method': 'onehot'
            },
            'segmentation': {
                'algorithm': 'kmeans',
                'n_clusters_range': [2, 10],
                'random_state': 42,
                'n_init': 10,
                'max_iter': 300
            },
            'prediction': {
                'models': ['logistic_regression', 'random_forest', 'xgboost'],
                'test_size': 0.3,
                'random_state': 42,
                'use_smote': True,
                'cross_validation_folds': 5
            },
            'model_params': {
                'logistic_regression': {
                    'max_iter': 1000,
                    'random_state': 42
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            },
            'monitoring': {
                'drift_threshold': 0.1,
                'performance_threshold': 0.85,
                'retraining_interval_days': 30
            }
        }
    
    def _set_paths(self):
        """Create necessary directories"""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file"""
        path = path or self.config_path
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)

# Global configuration instance
config = Config()