"""MLRun function handlers for the Banking ML Pipeline.

This module exposes functions that can be imported and used by MLRun runtime (local, Kubernetes, etc.)
for training, evaluating, and deploying the models in a reproducible and trackable way.

Usage (example):
    import mlrun
    fn = mlrun.code_to_function(name="banking_pipeline", filename="scripts/mlrun_handlers.py", kind="job")
    run = fn.run(handler="train_pipeline", params={"n_samples": 2000, "use_synthetic": True})

To build an MLRun project (basic example):
    mlrun project banking-ml -u . --name banking-ml

"""
from __future__ import annotations
import os
import typing as t
from datetime import datetime

import pandas as pd

try:
    import mlrun
except ImportError:  # graceful if executed without mlrun context
    mlrun = None  # type: ignore

# Ensure src is on path if running via MLRun in a different workdir
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.integrated_pipeline import IntegratedBankingPipeline
from src.data.synthetic_data_generator import create_synthetic_banking_data
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger("mlrun_handlers")


def _log_results_to_mlrun(results: dict):
    """Push key metrics and artifacts to MLRun context if available."""
    if mlrun is None:
        return
    context = mlrun.get_or_create_ctx("banking_pipeline")

    # Prediction metrics
    pred = results.get("prediction", {})
    metrics = pred.get("best_model_metrics", {})
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            context.log_result(k, v)

    if "best_model" in pred:
        context.log_result("best_model", pred.get("best_model"))

    # Segmentation metrics
    seg = results.get("segmentation", {})
    if "n_clusters" in seg:
        context.log_result("seg_n_clusters", seg.get("n_clusters"))
    if "silhouette_score" in seg:
        context.log_result("seg_silhouette", seg.get("silhouette_score"))

    # Optionally log artifacts (e.g., profiles)
    segment_profiles = seg.get("segment_profiles")
    if isinstance(segment_profiles, pd.DataFrame):
        context.log_dataset("segment_profiles", df=segment_profiles, format="parquet")


def train_pipeline(
    data_path: str | None = None,
    use_synthetic: bool = True,
    n_samples: int = 5000,
    output_dir: str = "models",
    config_path: str | None = None,
) -> dict:
    """Train full banking pipeline within MLRun.

    Parameters
    ----------
    data_path : Path to existing CSV data; ignored if use_synthetic True.
    use_synthetic : Generate synthetic data if True.
    n_samples : Number of synthetic rows.
    output_dir : Where to write trained models.
    config_path : Optional alt config YAML.

    Returns
    -------
    dict with training results summary.
    """
    logger.info("Starting MLRun training handler")
    if config_path:
        config.config_path = config_path
        config.config = config._load_config()
        logger.info(f"Loaded configuration from {config_path}")

    if use_synthetic:
        logger.info(f"Generating synthetic data: n_samples={n_samples}")
        data = create_synthetic_banking_data(n_samples=n_samples)
    else:
        if not data_path:
            raise ValueError("data_path required when use_synthetic is False")
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)

    logger.info(f"Data shape: {data.shape}")

    pipeline = IntegratedBankingPipeline()
    results = pipeline.run_pipeline(data)

    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_pipeline(output_dir)

    _log_results_to_mlrun(results)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "best_model": results.get("prediction", {}).get("best_model"),
        **results.get("prediction", {}).get("best_model_metrics", {}),
        "n_clusters": results.get("segmentation", {}).get("n_clusters"),
        "silhouette_score": results.get("segmentation", {}).get("silhouette_score"),
        "models_dir": output_dir,
    }
    logger.info(f"Training summary: {summary}")
    return summary


def load_and_predict(
    model_dir: str = "models",
    data_path: str | None = None,
    use_synthetic: bool = True,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """Load saved pipeline models and produce predictions (mainly classification scores).

    Returns a dataframe with predictions appended.
    """
    from src.pipeline.integrated_pipeline import IntegratedBankingPipeline

    if use_synthetic:
        data = create_synthetic_banking_data(n_samples=n_samples)
    else:
        if not data_path:
            raise ValueError("data_path required when use_synthetic is False")
        data = pd.read_csv(data_path)

    pipeline = IntegratedBankingPipeline()
    pipeline.load_pipeline(model_dir)

    # Re-run prediction stage only
    processed = pipeline._preprocess_data(data)  # type: ignore (private usage for inference convenience)
    pred_results = pipeline._prediction_stage(processed)  # type: ignore
    df_preds = pred_results["predictions_df"]

    if mlrun is not None:
        ctx = mlrun.get_or_create_ctx("banking_pipeline")
        ctx.log_dataset("predictions", df=df_preds.head(100), format="parquet")
    return df_preds


__all__ = ["train_pipeline", "load_and_predict"]
