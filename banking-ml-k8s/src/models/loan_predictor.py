# src/models/loan_predictor.py
"""Loan eligibility prediction module"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.pyplot as plt

from ..utils.logger import model_logger
from ..utils.config import config

class LoanEligibilityPredictor:
    """Handles loan eligibility prediction using multiple ML models"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model to use ('logistic_regression', 'random_forest', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.feature_importances = None
        self.models_evaluated = {}
        model_logger.info(f"LoanEligibilityPredictor initialized with {model_type}")
    
    def prepare_data_with_segments(self, X: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """
        Add segment information to feature matrix
        
        Args:
            X: Feature matrix
            segments: Segment labels
            
        Returns:
            Feature matrix with segments
        """
        model_logger.info("Adding segment information to features")
        X_with_segments = np.column_stack([X, segments])
        return X_with_segments
    
    def handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray, 
                              method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in the dataset
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Method to handle imbalance ('smote', 'class_weight')
            
        Returns:
            Balanced feature matrix and labels
        """
        model_logger.info(f"Handling class imbalance using {method}")
        
        # Log original distribution
        unique, counts = np.unique(y, return_counts=True)
        model_logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
        
        if method == 'smote' and config.get('prediction.use_smote', True):
            smote = SMOTE(random_state=config.get('prediction.random_state', 42))
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Log new distribution
            unique, counts = np.unique(y_balanced, return_counts=True)
            model_logger.info(f"Balanced class distribution: {dict(zip(unique, counts))}")
            
            return X_balanced, y_balanced
        else:
            return X, y
    
    def create_model(self, model_type: Optional[str] = None) -> Any:
        """Create a model instance based on type"""
        model_type = model_type or self.model_type
        
        if model_type == 'logistic_regression':
            params = config.get('model_params.logistic_regression', {})
            return LogisticRegression(**params)
        elif model_type == 'random_forest':
            params = config.get('model_params.random_forest', {})
            return RandomForestClassifier(**params)
        elif model_type == 'xgboost':
            params = config.get('model_params.xgboost', {})
            return xgb.XGBClassifier(eval_metric='logloss', **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a single model and evaluate performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_type: Type of model to train
            
        Returns:
            Dictionary with model and metrics
        """
        model_type = model_type or self.model_type
        model_logger.info(f"Training {model_type} model")
        
        # Create and train model
        model = self.create_model(model_type)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=config.get('prediction.cross_validation_folds', 5),
            scoring='roc_auc'
        )
        metrics['cv_score_mean'] = cv_scores.mean()
        metrics['cv_score_std'] = cv_scores.std()
        
        model_logger.info(f"{model_type} - AUC-ROC: {metrics['auc_roc']:.3f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Train multiple models and compare performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        model_types = config.get('prediction.models', ['logistic_regression', 'random_forest', 'xgboost'])
        results = []
        
        for model_type in model_types:
            result = self.train_single_model(X_train, y_train, X_test, y_test, model_type)
            self.models_evaluated[model_type] = result
            
            # Add to results
            metrics = result['metrics']
            metrics['model'] = model_type
            results.append(metrics)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
        
        # Select best model based on AUC-ROC
        best_model_idx = comparison_df['auc_roc'].idxmax()
        best_model_type = comparison_df.loc[best_model_idx, 'model']
        self.model_type = best_model_type
        self.model = self.models_evaluated[best_model_type]['model']
        
        model_logger.info(f"Best model: {best_model_type}")
        
        return comparison_df
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for GridSearchCV
            
        Returns:
            Best parameters and model
        """
        model_logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        
        # Default parameter grids
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0]
                }
            else:
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2']
                }
        
        # Create model
        base_model = self.create_model()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=config.get('prediction.cross_validation_folds', 5),
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        model_logger.info(f"Best parameters: {self.best_params}")
        model_logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'best_model': self.model
        }
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.feature_importances = importance_df
            model_logger.info(f"Top 5 features: {importance_df.head()['feature'].tolist()}")
            
            return importance_df
        else:
            model_logger.warning("Model doesn't support feature importances")
            return pd.DataFrame()
    
    def plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray) -> plt.Figure:
        """
        Plot ROC curves for all evaluated models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_type, result in self.models_evaluated.items():
            model = result['model']
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            
            ax.plot(fpr, tpr, label=f'{model_type} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        return self.model.predict_proba(X)
    
    def evaluate_by_segment(self, X_test: np.ndarray, y_test: np.ndarray, 
                          segments: np.ndarray) -> pd.DataFrame:
        """
        Evaluate model performance by customer segment
        
        Args:
            X_test: Test features
            y_test: Test labels
            segments: Segment labels for test data
            
        Returns:
            DataFrame with segment-wise metrics
        """
        model_logger.info("Evaluating model performance by segment")
        
        segment_results = []
        unique_segments = np.unique(segments)
        
        for segment in unique_segments:
            mask = segments == segment
            if mask.sum() > 0:
                y_pred = self.predict(X_test[mask])
                y_prob = self.predict_proba(X_test[mask])[:, 1]
                
                metrics = {
                    'segment': segment,
                    'n_samples': mask.sum(),
                    'accuracy': accuracy_score(y_test[mask], y_pred),
                    'precision': precision_score(y_test[mask], y_pred, zero_division=0),
                    'recall': recall_score(y_test[mask], y_pred, zero_division=0),
                    'f1_score': f1_score(y_test[mask], y_pred, zero_division=0),
                    'auc_roc': roc_auc_score(y_test[mask], y_prob) if len(np.unique(y_test[mask])) > 1 else 0
                }
                segment_results.append(metrics)
        
        return pd.DataFrame(segment_results)
    
    def save(self, filepath: str):
        """Save the prediction model"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importances': self.feature_importances
        }
        joblib.dump(model_data, filepath)
        model_logger.info(f"Prediction model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a saved prediction model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.feature_importances = model_data['feature_importances']
        model_logger.info(f"Prediction model loaded from {filepath}")