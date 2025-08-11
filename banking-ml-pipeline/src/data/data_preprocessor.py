# src/data/data_preprocessor.py
"""Data preprocessing module for the Banking ML Pipeline"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Optional, Dict, Any
import joblib

from ..utils.logger import data_logger
from ..utils.config import config

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self, 
                 numeric_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None):
        """
        Initialize the preprocessor
        
        Args:
            numeric_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self.numeric_features = numeric_features or config.get('preprocessing.numeric_features', [])
        self.categorical_features = categorical_features or config.get('preprocessing.categorical_features', [])
        self.preprocessor = None
        self.feature_names = None
        data_logger.info("DataPreprocessor initialized")
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        data_logger.info("Creating preprocessing pipeline")
        
        # Numeric transformer
        scaling_method = config.get('preprocessing.scaling_method', 'robust')
        if scaling_method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        data_logger.info(f"Pipeline created with {len(self.numeric_features)} numeric and "
                        f"{len(self.categorical_features)} categorical features")
        
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit the preprocessor and transform data"""
        data_logger.info(f"Fitting preprocessor on data with shape {X.shape}")
        
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()
        
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        self._extract_feature_names()
        
        data_logger.info(f"Data transformed to shape {X_transformed.shape}")
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(X)
    
    def _extract_feature_names(self):
        """Extract feature names after transformation"""
        # Get categorical feature names after encoding
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
        
        # Combine all feature names
        self.feature_names = self.numeric_features + list(cat_feature_names)
        data_logger.info(f"Total features after preprocessing: {len(self.feature_names)}")
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features"""
        data_logger.info("Starting feature engineering")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Add missing base features with defaults if not present
        if 'max_balance' not in df.columns:
            df['max_balance'] = df['avg_balance'] * 1.5
        
        if 'digital_usage_rate' not in df.columns:
            df['digital_usage_rate'] = 0.7
        
        if 'dependents' not in df.columns:
            df['dependents'] = 0
            
        if 'employment_type' not in df.columns:
            df['employment_type'] = 'Full-time'
            
        if 'loan_purpose' not in df.columns:
            df['loan_purpose'] = 'Personal'
        
        # Financial ratios
        df['debt_to_income_ratio'] = df['existing_loans'] * 5000 / (df['income'] + 1)
        df['loan_to_income_ratio'] = df['loan_amount_requested'] / (df['income'] + 1)
        
        # Behavioral features
        df['avg_balance_per_product'] = df['avg_balance'] / (df['num_products'] + 1)
        df['transaction_diversity'] = df['avg_transaction_amount'] / (df['max_transaction_amount'] + 1)
        
        # Risk indicators
        df['high_risk_profile'] = ((df['credit_score'] < 600) | 
                                   (df['previous_defaults'] > 0)).astype(int)
        
        # Additional derived features
        df['total_relationship_value'] = (df['avg_balance'] * 0.01 + 
                                         df['monthly_transactions'] * 10 +
                                         df['num_products'] * 1000)
        
        df['risk_score'] = (df['previous_defaults'] * 50 + 
                           (850 - df['credit_score']) / 10 +
                           df['existing_loans'] * 5)
        
        df['engagement_score'] = (df['digital_usage_rate'] * 40 +
                                 df['monthly_transactions'] / 3 +
                                 df['num_products'] * 5)
        
        # Handle any infinities or NaN values created
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        data_logger.info("Feature engineering completed")
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers in specified columns"""
        data_logger.info(f"Handling outliers using {method} method")
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                data_logger.info(f"Column {col}: {outlier_count} outliers capped")
        
        return df
    
    def save(self, filepath: str):
        """Save preprocessor to file"""
        joblib.dump(self.preprocessor, filepath)
        data_logger.info(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load preprocessor from file"""
        self.preprocessor = joblib.load(filepath)
        data_logger.info(f"Preprocessor loaded from {filepath}")
    
    def get_feature_importance_mask(self, feature_importances: np.ndarray, 
                                   top_k: int = 20) -> np.ndarray:
        """Get mask for top k important features"""
        if len(feature_importances) != len(self.feature_names):
            raise ValueError("Feature importances length doesn't match feature names")
        
        # Get indices of top k features
        top_indices = np.argsort(feature_importances)[-top_k:]
        
        # Create mask
        mask = np.zeros(len(self.feature_names), dtype=bool)
        mask[top_indices] = True
        
        return mask