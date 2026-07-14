"""
Utility functions for the Predictive Maintenance ML System.

This module contains functions for:
- Data loading and validation
- Preprocessing
- Feature engineering
- Model evaluation helpers
- Logging setup
"""

import logging
import json
import pickle
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from config import (
    TARGET_COLUMN,
    EXCLUDE_COLUMNS,
    FEATURE_SCALING,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    RANDOM_STATE,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(log_file: str = LOG_FILE) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file (str): Path to log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================


def load_data(
    file_path: str, validate: bool = True
) -> pd.DataFrame:
    """
    Load CSV data file with validation.

    Args:
        file_path (str): Path to CSV file.
        validate (bool): Whether to validate data integrity.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If data validation fails.
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data shape: {df.shape}")
        
        if validate:
            _validate_data(df)
        
        return df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def _validate_data(df: pd.DataFrame) -> None:
    """
    Validate data integrity.

    Args:
        df (pd.DataFrame): Data to validate.

    Raises:
        ValueError: If validation fails.
    """
    # Check for empty dataframe
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    logger.info(f"Data validation passed. Columns: {df.shape[1]}")


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def extract_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract features from dataframe (exclude target column).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Features only.
    """
    feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    return df[feature_columns]


def extract_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract target variable from dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.Series: Target variable.
    """
    return df[TARGET_COLUMN]


# ============================================================================
# FEATURE SCALING
# ============================================================================


def create_scaler(
    method: str = FEATURE_SCALING["method"],
) -> StandardScaler | MinMaxScaler:
    """
    Create feature scaler.

    Args:
        method (str): Scaling method ("standard" or "minmax").

    Returns:
        StandardScaler or MinMaxScaler: Scaler instance.
    """
    if method == "standard":
        logger.info("Creating StandardScaler")
        return StandardScaler()
    elif method == "minmax":
        logger.info("Creating MinMaxScaler")
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def fit_and_scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler]:
    """
    Fit scaler on training data and scale both train and test data.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        scaler: Scaler instance. If None, creates new scaler.

    Returns:
        Tuple: (scaled_X_train, scaled_X_test, scaler)
    """
    if scaler is None:
        scaler = create_scaler()
    
    logger.info("Fitting scaler on training data")
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info("Scaling test data")
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Features scaled. Shape: {X_train_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels (binary).
        y_pred_proba (np.ndarray): Predicted probabilities. Optional.

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Compute ROC-AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {str(e)}")
            metrics["roc_auc"] = None
    
    # Compute specificity and sensitivity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> str:
    """
    Print and return classification report.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Name of the model.

    Returns:
        str: Classification report.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Fault", "Fault"],
        digits=4,
    )
    
    logger.info(f"\n{model_name} Classification Report:\n{report}")
    return report


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    step: float = 0.01,
    min_precision: float = 0.5,
    target_recall: float = 0.95,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal probability threshold for recall optimization.

    This is critical for predictive maintenance - we want to maximize recall
    (catch all failures) while maintaining acceptable precision.

    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        min_threshold (float): Minimum threshold to try.
        max_threshold (float): Maximum threshold to try.
        step (float): Step size for threshold search.
        min_precision (float): Minimum acceptable precision.
        target_recall (float): Target recall threshold.

    Returns:
        Tuple: (optimal_threshold, threshold_metrics)
    """
    logger.info("Starting threshold optimization for recall maximization")
    
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Only consider thresholds meeting minimum precision requirement
        if precision >= min_precision:
            results.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "valid": True,
            })
    
    if not results:
        logger.warning(
            f"No threshold found with precision >= {min_precision}. "
            f"Using threshold with highest recall."
        )
        # Find threshold with highest recall regardless of precision
        best_result = max(
            results,
            key=lambda x: x["recall"]
        )
    else:
        # Find threshold closest to target recall with highest precision
        best_result = min(
            results,
            key=lambda x: (
                abs(x["recall"] - target_recall),
                -x["precision"],  # Negative because we want to maximize
            ),
        )
    
    optimal_threshold = best_result["threshold"]
    logger.info(
        f"Optimal threshold: {optimal_threshold:.3f} "
        f"(Precision: {best_result['precision']:.4f}, "
        f"Recall: {best_result['recall']:.4f})"
    )
    
    return optimal_threshold, {
        "results": results,
        "best_result": best_result,
    }


# ============================================================================
# CONFUSION MATRIX HELPER
# ============================================================================


def get_confusion_matrix_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """
    Get detailed confusion matrix components.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        Dict[str, int]: Dictionary with TN, FP, FN, TP.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================


def get_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get precision-recall curve.

    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.

    Returns:
        Tuple: (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return precision, recall, thresholds


def get_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve.

    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.

    Returns:
        Tuple: (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


# ============================================================================
# JSON SAVING HELPERS
# ============================================================================


def save_metrics_to_json(
    metrics: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary.
        filepath (str): Path to save JSON file.
    """
    # Convert non-serializable items
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
            metrics_serializable[key] = value
        else:
            metrics_serializable[key] = str(value)
    
    with open(filepath, "w") as f:
        json.dump(metrics_serializable, f, indent=4)
    
    logger.info(f"Metrics saved to {filepath}")


def load_metrics_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.

    Args:
        filepath (str): Path to JSON file.

    Returns:
        Dict[str, Any]: Metrics dictionary.
    """
    with open(filepath, "r") as f:
        metrics = json.load(f)
    
    logger.info(f"Metrics loaded from {filepath}")
    return metrics


# ============================================================================
# CLASS DISTRIBUTION ANALYSIS
# ============================================================================


def analyze_class_distribution(
    y: pd.Series,
    dataset_name: str = "Dataset",
) -> Dict[str, Any]:
    """
    Analyze class distribution in target variable.

    Args:
        y (pd.Series): Target variable.
        dataset_name (str): Name of dataset for logging.

    Returns:
        Dict[str, Any]: Class distribution statistics.
    """
    value_counts = y.value_counts()
    distribution = {
        "class_0_count": int(value_counts.get(0, 0)),
        "class_1_count": int(value_counts.get(1, 0)),
    }
    
    total = sum(distribution.values())
    distribution["class_0_percentage"] = (
        distribution["class_0_count"] / total * 100
    )
    distribution["class_1_percentage"] = (
        distribution["class_1_count"] / total * 100
    )
    distribution["imbalance_ratio"] = (
        distribution["class_1_count"] / distribution["class_0_count"]
        if distribution["class_0_count"] > 0
        else 0
    )
    
    logger.info(
        f"\n{dataset_name} Class Distribution:"
        f"\n  Class 0 (No Fault): {distribution['class_0_count']} "
        f"({distribution['class_0_percentage']:.2f}%)"
        f"\n  Class 1 (Fault): {distribution['class_1_count']} "
        f"({distribution['class_1_percentage']:.2f}%)"
        f"\n  Imbalance Ratio: {distribution['imbalance_ratio']:.4f}"
    )
    
    return distribution


if __name__ == "__main__":
    """Test utility functions."""
    logger.info("Utility functions loaded successfully!")