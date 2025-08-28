# src/__init__.py
"""Banking ML Pipeline - Main Package"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make key classes available at package level
from .pipeline.integrated_pipeline import IntegratedBankingPipeline
from .data.data_preprocessor import DataPreprocessor
from .models.customer_segmentation import CustomerSegmentation
from .models.loan_predictor import LoanEligibilityPredictor

__all__ = [
    'IntegratedBankingPipeline',
    'DataPreprocessor',
    'CustomerSegmentation',
    'LoanEligibilityPredictor'
]