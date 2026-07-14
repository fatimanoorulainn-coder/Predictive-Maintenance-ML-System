"""
Configuration module for the Predictive Maintenance ML System.

This module centralizes all configuration parameters for reproducibility
and easy modification of hyperparameters, paths, and model settings.
"""

import os
from typing import Dict, Any, List

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Input data files
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_test.csv")

# Target column name
TARGET_COLUMN = "fault_label"

# Train-test split parameters
TRAIN_TEST_SPLIT_RATIO = 0.2
STRATIFY_ENABLED = True
RANDOM_STATE = 42

# ============================================================================
# IMBALANCE HANDLING CONFIGURATION
# ============================================================================

# SMOTE parameters for oversampling
SMOTE_CONFIG = {
    "sampling_strategy": "auto" ,  # Oversample minority class to 50% of majority
    "random_state": RANDOM_STATE,
    "k_neighbors": 5,
}

# Class weights for imbalanced classification
USE_CLASS_WEIGHTS = True

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random Forest hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# Random Forest hyperparameter tuning grid
RF_TUNING_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [1, 2, 4],
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
    "eval_metric": "logloss",
}

# XGBoost hyperparameter tuning grid
XGBOOST_TUNING_GRID = {
    "max_depth": [5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

# Logistic Regression baseline parameters
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "solver": "lbfgs",
}

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================

# GridSearchCV parameters
GRID_SEARCH_CONFIG = {
    "cv": 5,  # 5-fold cross-validation
    "scoring": "recall",  # Optimize for recall (high priority for missing failures)
    "n_jobs": -1,
    "verbose": 1,
}

# ============================================================================
# THRESHOLD TUNING CONFIGURATION
# ============================================================================

# Threshold tuning for recall optimization
THRESHOLD_CONFIG = {
    "min_threshold": 0.1,
    "max_threshold": 0.9,
    "step": 0.01,
    "min_precision": 0.5,  # Minimum acceptable precision
    "target_recall": 0.95,  # Target recall threshold
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics to compute
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "specificity",
    "sensitivity",
]

# ROC curve parameters
ROC_CURVE_PARAMS = {
    "drop_intermediate": True,
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot settings
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 300,
    "font_size": 12,
    "title_size": 14,
    "label_size": 12,
}

# Color palette
COLORS = {
    "positive": "#2ecc71",  # Green
    "negative": "#e74c3c",  # Red
    "neutral": "#3498db",   # Blue
    "warning": "#f39c12",   # Orange
}

# ============================================================================
# MODEL SAVING CONFIGURATION
# ============================================================================

# Model file names
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
THRESHOLD_SAVE_PATH = os.path.join(MODELS_DIR, "optimal_threshold.pkl")
FEATURE_NAMES_SAVE_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
SMOTE_SAVE_PATH = os.path.join(MODELS_DIR, "smote.joblib")

# Metrics save path
METRICS_SAVE_PATH = os.path.join(OUTPUTS_DIR, "model_metrics.json")
THRESHOLD_METRICS_PATH = os.path.join(OUTPUTS_DIR, "threshold_metrics.json")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FILE = os.path.join(LOGS_DIR, "training.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================

# Random seeds for reproducibility
NUMPY_SEED = RANDOM_STATE
SKLEARN_SEED = RANDOM_STATE
TF_SEED = RANDOM_STATE

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Features to use (all features except target)
EXCLUDE_COLUMNS = [TARGET_COLUMN]

# Feature scaling
FEATURE_SCALING = {
    "enabled": True,
    "method": "standard",  # "standard" or "minmax"
}

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

# Acceptable performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_recall": 0.85,  # Minimum recall for production
    "min_precision": 0.50,  # Minimum precision
    "min_auc": 0.80,  # Minimum ROC-AUC
}

# ============================================================================
# FUNCTION TO GET ALL CONFIGURATIONS
# ============================================================================


def get_config() -> Dict[str, Any]:
    """
    Get all configuration parameters as a dictionary.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": DATA_DIR,
            "models_dir": MODELS_DIR,
            "outputs_dir": OUTPUTS_DIR,
            "plots_dir": PLOTS_DIR,
            "logs_dir": LOGS_DIR,
        },
        "data": {
            "train_path": TRAIN_DATA_PATH,
            "test_path": TEST_DATA_PATH,
            "target_column": TARGET_COLUMN,
            "test_split_ratio": TRAIN_TEST_SPLIT_RATIO,
            "random_state": RANDOM_STATE,
        },
        "model_params": {
            "random_forest": RANDOM_FOREST_PARAMS,
            "xgboost": XGBOOST_PARAMS,
            "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
        },
        "imbalance": {
            "smote": SMOTE_CONFIG,
            "use_class_weights": USE_CLASS_WEIGHTS,
        },
        "tuning": {
            "grid_search": GRID_SEARCH_CONFIG,
            "threshold": THRESHOLD_CONFIG,
        },
    }


if __name__ == "__main__":
    """Test configuration loading."""
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Project root: {config['paths']['project_root']}")
    print(f"Train data: {config['data']['train_path']}")
    print(f"Models will be saved to: {config['paths']['models_dir']}")