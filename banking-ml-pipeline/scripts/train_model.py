# scripts/train_model.py
"""Script to train the banking ML pipeline"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.integrated_pipeline import IntegratedBankingPipeline
from src.data.synthetic_data_generator import create_synthetic_banking_data
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger('training_script')

def main():
    parser = argparse.ArgumentParser(description='Train Banking ML Pipeline')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--synthetic', action='store_true', 
                      help='Use synthetic data for training')
    parser.add_argument('--n_samples', type=int, default=5000,
                      help='Number of synthetic samples to generate')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("BANKING ML PIPELINE TRAINING")
    logger.info("="*50)
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Load configuration if provided
        if args.config:
            config.config_path = args.config
            config.config = config._load_config()
            logger.info(f"Loaded configuration from: {args.config}")
        
        # Load or create data
        if args.synthetic:
            logger.info(f"Creating synthetic data with {args.n_samples} samples")
            data = create_synthetic_banking_data(n_samples=args.n_samples)
            
            # Add loan approval target (for demonstration)
            # In real scenario, this would come from actual data
            approval_score = (
                (data['credit_score'] > 650).astype(int) * 0.3 +
                (data['income'] > 50000).astype(int) * 0.2 +
                (data['previous_defaults'] == 0).astype(int) * 0.3 +
                (data['payment_history_score'] > 0.7).astype(int) * 0.2
            )
            data['loan_approved'] = (approval_score > 0.5).astype(int)
            
        elif args.data:
            logger.info(f"Loading data from: {args.data}")
            data = pd.read_csv(args.data)
        else:
            raise ValueError("Either --data or --synthetic must be specified")
        
        logger.info(f"Data shape: {data.shape}")
        
        # Initialize pipeline
        pipeline = IntegratedBankingPipeline()
        
        # Run training pipeline
        results = pipeline.run_pipeline(data)
        
        # Save models
        pipeline.save_pipeline(args.output_dir)
        
        # Print results summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*50)
        
        summary = results.get('prediction', {}).get('best_model_metrics', {})
        logger.info(f"Best Model: {results.get('prediction', {}).get('best_model', 'Unknown')}")
        logger.info(f"AUC-ROC Score: {summary.get('auc_roc', 0):.3f}")
        logger.info(f"Accuracy: {summary.get('accuracy', 0):.3f}")
        logger.info(f"Precision: {summary.get('precision', 0):.3f}")
        logger.info(f"Recall: {summary.get('recall', 0):.3f}")
        logger.info(f"F1-Score: {summary.get('f1_score', 0):.3f}")
        
        logger.info(f"\nNumber of segments: {results.get('segmentation', {}).get('n_clusters', 0)}")
        logger.info(f"Segmentation quality: {results.get('segmentation', {}).get('silhouette_score', 0):.3f}")
        
        logger.info(f"\nModels saved to: {args.output_dir}")
        logger.info(f"Completed at: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()