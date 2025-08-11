# src/pipeline/integrated_pipeline.py
"""Integrated pipeline combining segmentation and prediction"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Any, Tuple
import joblib
from datetime import datetime
import json

#from ..data.data_preprocessor import DataPreprocessor
from src.data.data_preprocessor import DataPreprocessor
from src.models.customer_segmentation import CustomerSegmentation
from src.models.loan_predictor import LoanEligibilityPredictor
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger('integrated_pipeline')

class IntegratedBankingPipeline:
    """Main pipeline integrating segmentation and prediction"""
    
    #def __init__(self):
    def __init__(self, directory: str = 'models'):
        self.directory = directory
        logger.info(f"Pipeline loaded from {directory}")#"""Initialize the integrated pipeline"""
        self.preprocessor = DataPreprocessor()
        self.segmenter = CustomerSegmentation(algorithm='kmeans')
        self.predictor = LoanEligibilityPredictor(model_type='random_forest')
        self.pipeline_metadata = {
            'created_at': datetime.now().isoformat(),
            'version': config.get('project.version', '1.0.0')
        }
        logger.info("IntegratedBankingPipeline initialized")
    
    def run_pipeline(self, data: pd.DataFrame, target_column: str = 'loan_approved') -> Dict[str, Any]:
        """
        Execute the complete pipeline
        
        Args:
            data: Input dataframe
            target_column: Name of target column
            
        Returns:
            Dictionary with results and models
        """
        logger.info("="*50)
        logger.info("BANKING ML PIPELINE - EXECUTION STARTED")
        logger.info("="*50)
        
        results = {}
        
        # Step 1: Data Preprocessing
        logger.info("\n[Step 1] Data Preprocessing")
        X, y, feature_names = self._preprocess_data(data, target_column)
        results['preprocessing'] = {
            'n_features_original': len(data.columns) - 1,
            'n_features_processed': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # Step 2: Customer Segmentation
        logger.info("\n[Step 2] Customer Segmentation")
        cluster_labels, segmentation_results = self._perform_segmentation(X, data)
        results['segmentation'] = segmentation_results
        
        # Step 3: Data Splitting
        logger.info("\n[Step 3] Data Splitting")
        X_train, X_test, y_train, y_test, segments_train, segments_test = self._split_data(
            X, y, cluster_labels
        )
        
        # Step 4: Loan Eligibility Prediction
        logger.info("\n[Step 4] Loan Eligibility Prediction")
        prediction_results = self._train_prediction_models(
            X_train, X_test, y_train, y_test, segments_train, segments_test, feature_names
        )
        results['prediction'] = prediction_results
        
        # Step 5: Segment-wise Analysis
        logger.info("\n[Step 5] Segment-wise Analysis")
        segment_analysis = self._analyze_segments(X_test, y_test, segments_test)
        results['segment_analysis'] = segment_analysis
        
        # Update metadata
        self.pipeline_metadata['completed_at'] = datetime.now().isoformat()
        self.pipeline_metadata['results_summary'] = self._create_summary(results)
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("="*50)
        
        return results
    
    def _preprocess_data(self, data: pd.DataFrame, 
                        target_column: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """Preprocess the data"""
        # Feature engineering
        data = self.preprocessor.feature_engineering(data)
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column].values
        else:
            logger.warning(f"Target column '{target_column}' not found. Creating synthetic target.")
            # Create synthetic target for demonstration
            y = np.random.randint(0, 2, len(data))
            X = data
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        feature_names = self.preprocessor.feature_names
        
        logger.info(f"Data preprocessed: {X_processed.shape}")
        
        return X_processed, y, feature_names
    
    def _perform_segmentation(self, X: np.ndarray, 
                            original_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Perform customer segmentation"""
        # Find optimal clusters
        cluster_evaluation = self.segmenter.find_optimal_clusters(X)
        
        # Perform clustering
        cluster_labels = self.segmenter.fit(X)
        
        # Create segment profiles
        key_features = ['age', 'income', 'avg_balance', 'credit_score', 'num_products']
        available_features = [f for f in key_features if f in original_data.columns]
        
        segment_profiles = self.segmenter.create_segment_profiles(
            original_data, cluster_labels, available_features
        )
        
        # Get segment names
        segment_names = self.segmenter.get_segment_names(segment_profiles)
        
        results = {
            'n_clusters': self.segmenter.optimal_clusters,
            'silhouette_score': cluster_evaluation['silhouette'][-1],
            'segment_sizes': dict(zip(segment_profiles['cluster'], segment_profiles['size'])),
            'segment_names': segment_names
        }
        
        return cluster_labels, results
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, 
                   segments: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train and test sets"""
        # Add segments to features
        X_with_segments = self.predictor.prepare_data_with_segments(X, segments)
        
        # Split data
        test_size = config.get('prediction.test_size', 0.3)
        random_state = config.get('prediction.random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_segments, y, test_size=test_size, 
            random_state=random_state, stratify=y
        )
        
        # Extract segments from the last column
        segments_train = X_train[:, -1].astype(int)
        segments_test = X_test[:, -1].astype(int)
        
        logger.info(f"Data split: Train {len(X_train)}, Test {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, segments_train, segments_test
    
    def _train_prediction_models(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               segments_train: np.ndarray, segments_test: np.ndarray,
                               feature_names: list) -> Dict[str, Any]:
        """Train and evaluate prediction models"""
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.predictor.handle_imbalanced_data(
            X_train, y_train
        )
        
        # Train multiple models
        comparison_results = self.predictor.train_multiple_models(
            X_train_balanced, y_train_balanced, X_test, y_test
        )
        
        # Get feature importance
        feature_names_with_segment = feature_names + ['segment']
        feature_importance = self.predictor.get_feature_importance(feature_names_with_segment)
        
        # Evaluate by segment
        segment_metrics = self.predictor.evaluate_by_segment(X_test, y_test, segments_test)
        
        results = {
            'model_comparison': comparison_results.to_dict(),
            'best_model': self.predictor.model_type,
            'best_model_metrics': comparison_results[
                comparison_results['model'] == self.predictor.model_type
            ].to_dict('records')[0],
            'top_features': feature_importance.head(10).to_dict() if not feature_importance.empty else {},
            'segment_metrics': segment_metrics.to_dict()
        }
        
        return results
    
    def _analyze_segments(self, X_test: np.ndarray, y_test: np.ndarray,
                        segments_test: np.ndarray) -> Dict[str, Any]:
        """Perform detailed segment analysis"""
        analysis = {}
        
        # Get predictions
        y_pred = self.predictor.predict(X_test)
        y_prob = self.predictor.predict_proba(X_test)[:, 1]
        
        # Analyze each segment
        for segment in np.unique(segments_test):
            mask = segments_test == segment
            if mask.sum() > 0:
                analysis[f'segment_{segment}'] = {
                    'n_samples': int(mask.sum()),
                    'approval_rate_actual': float(y_test[mask].mean()),
                    'approval_rate_predicted': float(y_pred[mask].mean()),
                    'avg_approval_probability': float(y_prob[mask].mean()),
                    'std_approval_probability': float(y_prob[mask].std())
                }
        
        return analysis
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of pipeline results"""
        summary = {
            'total_samples': results['preprocessing']['n_samples'],
            'n_features_processed': results['preprocessing']['n_features_processed'],
            'n_segments': results['segmentation']['n_clusters'],
            'segmentation_quality': results['segmentation']['silhouette_score'],
            'best_prediction_model': results['prediction']['best_model'],
            'prediction_auc_roc': results['prediction']['best_model_metrics']['auc_roc']
        }
        return summary
    
    def predict_new_customer(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction for a new customer
        
        Args:
            customer_data: DataFrame with customer features
            
        Returns:
            Prediction results
        """
        # Preprocess
        customer_data = self.preprocessor.feature_engineering(customer_data)
        X_new = self.preprocessor.transform(customer_data)
        
        # Get segment
        segment = self.segmenter.predict(X_new)[0]
        
        # Add segment to features
        X_new_with_segment = np.column_stack([X_new, [segment]])
        
        # Predict
        prediction = self.predictor.predict(X_new_with_segment)[0]
        probability = self.predictor.predict_proba(X_new_with_segment)[0, 1]
        
        # Get recommendation
        recommendation = self._get_recommendation(segment, probability)
        
        return {
            'customer_segment': int(segment),
            'segment_name': self.segmenter.segment_profiles.get(segment, f'Segment {segment}'),
            'loan_approved': bool(prediction),
            'approval_probability': float(probability),
            'recommendation': recommendation,
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium'
        }
    
    def _get_recommendation(self, segment: int, probability: float) -> str:
        """Generate recommendation based on segment and probability"""
        if probability > 0.8:
            return "Strong candidate for loan approval with standard terms"
        elif probability > 0.6:
            return "Good candidate, recommend approval with standard verification"
        elif probability > 0.4:
            return "Moderate risk, suggest additional documentation or collateral"
        else:
            return "High risk profile, recommend alternative products or rejection"
    
    def save_pipeline(self, directory: str = 'models'):
        """Save all pipeline components"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save components
        self.preprocessor.save(f"{directory}/preprocessor.pkl")
        self.segmenter.save(f"{directory}/segmenter.pkl")
        self.predictor.save(f"{directory}/predictor.pkl")
        
        # Save metadata
        with open(f"{directory}/pipeline_metadata.json", 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=4)
        
        logger.info(f"Pipeline saved to {directory}")
    
    def load_pipeline(self, directory: str = 'models'):
        """Load pipeline components"""
        # Load components
        self.preprocessor.load(f"{directory}/preprocessor.pkl")
        self.segmenter.load(f"{directory}/segmenter.pkl")
        self.predictor.load(f"{directory}/predictor.pkl")
        
        # Load metadata
        with open(f"{directory}/pipeline_metadata.json", 'r') as f:
            self.pipeline_metadata = json.load(f)