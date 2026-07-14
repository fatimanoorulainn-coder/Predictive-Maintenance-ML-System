"""
Model Training Module for Predictive Maintenance System.

This module implements:
- Data loading and preprocessing
- Train-test splitting with stratification
- Imbalance handling (SMOTE + class weights)
- Model training (Random Forest, XGBoost, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Threshold optimization for recall maximization
- Model persistence

Key Engineering Decision:
The system prioritizes RECALL because missing a machinery failure is extremely
costly (equipment damage, production loss, safety hazards). We use:
1. SMOTE for oversampling minority class
2. Class weights to penalize false negatives
3. Threshold tuning to maximize recall while maintaining acceptable precision
"""

import logging
import pickle
import warnings
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
    STRATIFY_ENABLED,
    RANDOM_STATE,
    RANDOM_FOREST_PARAMS,
    RF_TUNING_GRID,
    XGBOOST_PARAMS,
    XGBOOST_TUNING_GRID,
    LOGISTIC_REGRESSION_PARAMS,
    GRID_SEARCH_CONFIG,
    THRESHOLD_CONFIG,
    MODEL_SAVE_PATH,
    THRESHOLD_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
    SCALER_SAVE_PATH,
    SMOTE_SAVE_PATH,
    METRICS_SAVE_PATH,
    THRESHOLD_METRICS_PATH,
)
from utils import (
    setup_logging,
    load_data,
    extract_features,
    extract_target,
    fit_and_scale_features,
    create_scaler,
    compute_metrics,
    print_classification_report,
    analyze_class_distribution,
    find_optimal_threshold,
    save_metrics_to_json,
    get_confusion_matrix_labels,
)

warnings.filterwarnings("ignore")

# Initialize logger
logger = setup_logging()

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================


class PredictiveMaintenanceTrainer:
    """
    Comprehensive ML training pipeline for predictive maintenance.
    
    Attributes:
        X_train (np.ndarray): Scaled training features
        X_test (np.ndarray): Scaled test features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        feature_names (List[str]): Feature names
        scaler: Feature scaler
        smote: SMOTE sampler
        models (Dict): Trained models
        optimal_threshold (float): Optimal prediction threshold
    """
    
    def __init__(self):
        """Initialize the trainer."""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.smote = None
        self.models = {}
        self.optimal_threshold = 0.5
        
        logger.info("=" * 80)
        logger.info("Predictive Maintenance ML System Initialized")
        logger.info("=" * 80)
    
    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    def load_and_prepare_data(self) -> None:
        """
        Load and prepare training and test data.
        
        This function:
        1. Loads training and test data from CSV
        2. Validates data integrity
        3. Separates features and labels
        4. Performs train-test split on training data
        5. Analyzes class distribution
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING AND PREPARATION")
        logger.info("=" * 80)
        
        # Load all available training data
        logger.info(f"Loading data from {TRAIN_DATA_PATH}")
        df_train = load_data(TRAIN_DATA_PATH)
        logger.info(f"Full dataset shape: {df_train.shape}")
        
        # Extract features and target
        X_full = extract_features(df_train)
        y_full = extract_target(df_train)
        
        # Store feature names
        self.feature_names = list(X_full.columns)
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info(f"Features: {self.feature_names}")
        
        # Split data: use 80% for training, 20% for internal testing
        logger.info(f"\nSplitting data (train: {1-TRAIN_TEST_SPLIT_RATIO}, "
                   f"test: {TRAIN_TEST_SPLIT_RATIO})")
        
        if STRATIFY_ENABLED:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(
                    X_full,
                    y_full,
                    test_size=TRAIN_TEST_SPLIT_RATIO,
                    stratify=y_full,
                    random_state=RANDOM_STATE,
                )
            )
            logger.info("Using stratified split for imbalanced classes")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(
                    X_full,
                    y_full,
                    test_size=TRAIN_TEST_SPLIT_RATIO,
                    random_state=RANDOM_STATE,
                )
            )
        
        logger.info(f"Training set size: {self.X_train.shape}")
        logger.info(f"Test set size: {self.X_test.shape}")
        
        # Analyze class distribution
        logger.info("\nClass Distribution Analysis:")
        train_dist = analyze_class_distribution(self.y_train, "Training Set")
        test_dist = analyze_class_distribution(self.y_test, "Test Set")
        
        logger.info("Data loading and preparation completed successfully!")
    
    # ========================================================================
    # FEATURE SCALING
    # ========================================================================
    
    def scale_features(self) -> None:
        """
        Fit scaler on training data and scale both train and test features.
        
        This prevents data leakage by fitting the scaler only on training data.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FEATURE SCALING")
        logger.info("=" * 80)
        
        self.scaler = create_scaler()
        
        logger.info("Fitting StandardScaler on training data...")
        self.X_train, self.X_test, self.scaler = fit_and_scale_features(
            self.X_train,
            self.X_test,
            self.scaler,
        )
        
        logger.info("Feature scaling completed!")
    
    # ========================================================================
    # IMBALANCE HANDLING
    # ========================================================================
    
    def handle_imbalance(self) -> None:
        """
        Handle class imbalance using SMOTE oversampling.
        
        SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic
        samples for the minority class, which helps the model learn the
        minority class better without duplicating existing samples.
        
        Engineering Note:
        We also use class_weight='balanced' in model initialization.
        This combination of techniques is very effective for imbalanced data.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: IMBALANCE HANDLING WITH SMOTE")
        logger.info("=" * 80)
        
        logger.info("Applying SMOTE oversampling...")
        logger.info(f"Original training set distribution:")
        logger.info(f"  Class 0: {np.sum(self.y_train == 0)}")
        logger.info(f"  Class 1: {np.sum(self.y_train == 1)}")
        
        self.smote = SMOTE(
            sampling_strategy=THRESHOLD_CONFIG.get("sampling_strategy", "auto"),
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
        
        self.X_train, self.y_train = self.smote.fit_resample(
            self.X_train,
            self.y_train,
        )
        
        logger.info(f"After SMOTE training set distribution:")
        logger.info(f"  Class 0: {np.sum(self.y_train == 0)}")
        logger.info(f"  Class 1: {np.sum(self.y_train == 1)}")
        logger.info(f"New training set size: {self.X_train.shape}")
        
        logger.info("SMOTE oversampling completed!")
    
    # ========================================================================
    # BASELINE MODEL: LOGISTIC REGRESSION
    # ========================================================================
    
    def train_logistic_regression_baseline(self) -> None:
        """
        Train Logistic Regression as baseline model.
        
        A simple, interpretable baseline helps us understand the complexity
        of the problem and provides a reference point for more complex models.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4a: TRAINING BASELINE LOGISTIC REGRESSION")
        logger.info("=" * 80)
        
        logger.info("Training Logistic Regression baseline...")
        
        lr_model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        lr_model.fit(self.X_train, self.y_train)
        
        self.models["logistic_regression"] = {
            "model": lr_model,
            "type": "baseline",
        }
        
        # Evaluate on test set
        y_pred = lr_model.predict(self.X_test)
        y_pred_proba = lr_model.predict_proba(self.X_test)[:, 1]
        
        metrics = compute_metrics(self.y_test, y_pred, y_pred_proba)
        
        logger.info("\nLogistic Regression Baseline Results (Threshold=0.5):")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        print_classification_report(self.y_test, y_pred, "Logistic Regression")
    
    # ========================================================================
    # RANDOM FOREST TRAINING AND TUNING
    # ========================================================================
    
    def train_random_forest(self, tune: bool = True) -> None:
        """
        Train Random Forest classifier.
        
        Args:
            tune (bool): Whether to perform hyperparameter tuning.
        
        Random Forests are excellent for tabular sensor data because:
        1. They capture non-linear relationships
        2. They handle feature interactions well
        3. They provide feature importance scores
        4. They're robust to outliers and scaling
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4b: TRAINING RANDOM FOREST CLASSIFIER")
        logger.info("=" * 80)
        
        if tune:
            logger.info("Performing hyperparameter tuning with GridSearchCV...")
            logger.info(f"Tuning grid: {RF_TUNING_GRID}")
            
            base_model = RandomForestClassifier(
                n_estimators=50,  # Lower initial value for faster tuning
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )
            
            grid_search = GridSearchCV(
                base_model,
                RF_TUNING_GRID,
                **GRID_SEARCH_CONFIG,
            )
            
            logger.info("Running GridSearchCV (this may take a few minutes)...")
            grid_search.fit(self.X_train, self.y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV recall score: {grid_search.best_score_:.4f}")
            
            rf_model = grid_search.best_estimator_
        
        else:
            logger.info("Training Random Forest with default parameters...")
            rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
            rf_model.fit(self.X_train, self.y_train)
        
        self.models["random_forest"] = {
            "model": rf_model,
            "type": "ensemble",
        }
        
        # Evaluate on test set
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        metrics = compute_metrics(self.y_test, y_pred, y_pred_proba)
        
        logger.info("\nRandom Forest Results (Threshold=0.5):")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        print_classification_report(self.y_test, y_pred, "Random Forest")
    
    # ========================================================================
    # XGBOOST TRAINING AND TUNING
    # ========================================================================
    
    def train_xgboost(self, tune: bool = True) -> None:
        """
        Train XGBoost classifier.
        
        Args:
            tune (bool): Whether to perform hyperparameter tuning.
        
        XGBoost is particularly suitable for this application because:
        1. Optimized for tabular data (sensor readings)
        2. Fast training and prediction
        3. Handles feature interactions via tree structures
        4. Native support for class imbalance via scale_pos_weight
        5. Provides feature importance and SHAP values
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4c: TRAINING XGBOOST CLASSIFIER")
        logger.info("=" * 80)
        
        # Calculate scale_pos_weight for class imbalance
        class_weights = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
        logger.info(f"Scale pos weight (for imbalance): {class_weights:.2f}")
        
        if tune:
            logger.info("Performing hyperparameter tuning with GridSearchCV...")
            logger.info(f"Tuning grid: {XGBOOST_TUNING_GRID}")
            
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                scale_pos_weight=class_weights,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )
            
            grid_search = GridSearchCV(
                base_model,
                XGBOOST_TUNING_GRID,
                **GRID_SEARCH_CONFIG,
            )
            
            logger.info("Running GridSearchCV (this may take a few minutes)...")
            grid_search.fit(self.X_train, self.y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV recall score: {grid_search.best_score_:.4f}")
            
            xgb_model = grid_search.best_estimator_
        
        else:
            logger.info("Training XGBoost with default parameters...")
            xgb_params = XGBOOST_PARAMS.copy()
            xgb_params["scale_pos_weight"] = class_weights
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(self.X_train, self.y_train)
        
        self.models["xgboost"] = {
            "model": xgb_model,
            "type": "gradient_boosting",
        }
        
        # Evaluate on test set
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)[:, 1]
        
        metrics = compute_metrics(self.y_test, y_pred, y_pred_proba)
        
        logger.info("\nXGBoost Results (Threshold=0.5):")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        print_classification_report(self.y_test, y_pred, "XGBoost")
    
    # ========================================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================================
    
    def optimize_thresholds(self) -> None:
        """
        Optimize prediction thresholds for recall maximization.
        
        This is the MOST IMPORTANT STEP for predictive maintenance.
        
        Why threshold tuning matters:
        - Default 0.5 threshold is designed for balanced datasets
        - For imbalanced + high-cost failures, we want HIGHER RECALL
        - We lower the threshold to catch more failures (higher recall)
        - We accept lower precision (more false alarms) in exchange
        
        Engineering reasoning:
        - Cost of missing a failure: Equipment damage, safety risks, downtime
        - Cost of false alarm: Unnecessary inspection/maintenance
        - False alarms are MUCH cheaper than missed failures
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: THRESHOLD OPTIMIZATION FOR RECALL")
        logger.info("=" * 80)
        
        # Find optimal threshold for the best model (XGBoost)
        best_model = self.models["xgboost"]["model"]
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            self.y_test,
            y_pred_proba,
            min_threshold=THRESHOLD_CONFIG["min_threshold"],
            max_threshold=THRESHOLD_CONFIG["max_threshold"],
            step=THRESHOLD_CONFIG["step"],
            min_precision=THRESHOLD_CONFIG["min_precision"],
            target_recall=THRESHOLD_CONFIG["target_recall"],
        )
        
        self.optimal_threshold = optimal_threshold
        
        # Evaluate model with optimal threshold
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
        metrics_optimized = compute_metrics(
            self.y_test,
            y_pred_optimized,
            y_pred_proba,
        )
        
        logger.info(f"\nXGBoost with Optimal Threshold ({optimal_threshold:.3f}):")
        for metric_name, metric_value in metrics_optimized.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Detailed confusion matrix
        cm_labels = get_confusion_matrix_labels(self.y_test, y_pred_optimized)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives: {cm_labels['true_negatives']}")
        logger.info(f"  False Positives: {cm_labels['false_positives']}")
        logger.info(f"  False Negatives: {cm_labels['false_negatives']}")
        logger.info(f"  True Positives: {cm_labels['true_positives']}")
        
        print_classification_report(
            self.y_test,
            y_pred_optimized,
            "XGBoost (Optimized Threshold)"
        )
        
        # Save threshold metrics
        save_metrics_to_json(
            {
                "optimal_threshold": float(optimal_threshold),
                "metrics": metrics_optimized,
                "threshold_search_results": [
                    {k: float(v) if isinstance(v, (int, float, np.number)) else v
                     for k, v in result.items()}
                    for result in threshold_metrics.get("results", [])
                ],
            },
            THRESHOLD_METRICS_PATH,
        )
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    
    def save_best_model(self) -> None:
        """
        Save the best model and related artifacts.
        
        This includes:
        - Trained model (XGBoost)
        - Optimal threshold
        - Feature names
        - Fitted scaler
        - SMOTE sampler (for reproducibility)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL PERSISTENCE")
        logger.info("=" * 80)
        
        # Save best model (XGBoost)
        logger.info(f"Saving best model to {MODEL_SAVE_PATH}")
        best_model = self.models["xgboost"]["model"]
        joblib.dump(best_model, MODEL_SAVE_PATH)
        
        # Save optimal threshold
        logger.info(f"Saving optimal threshold to {THRESHOLD_SAVE_PATH}")
        with open(THRESHOLD_SAVE_PATH, "wb") as f:
            pickle.dump(self.optimal_threshold, f)
        
        # Save feature names
        logger.info(f"Saving feature names to {FEATURE_NAMES_SAVE_PATH}")
        with open(FEATURE_NAMES_SAVE_PATH, "wb") as f:
            pickle.dump(self.feature_names, f)
        
        # Save scaler
        logger.info(f"Saving scaler to {SCALER_SAVE_PATH}")
        joblib.dump(self.scaler, SCALER_SAVE_PATH)
        
        # Save SMOTE for reference
        logger.info(f"Saving SMOTE to {SMOTE_SAVE_PATH}")
        joblib.dump(self.smote, SMOTE_SAVE_PATH)
        
        logger.info("All models and artifacts saved successfully!")
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_full_pipeline(self, tune_models: bool = True) -> Dict[str, Any]:
        """
        Execute the full training pipeline.
        
        Args:
            tune_models (bool): Whether to perform hyperparameter tuning.
        
        Returns:
            Dict[str, Any]: Results and metrics from training.
        """
        logger.info("\n\n")
        logger.info("*" * 80)
        logger.info("PREDICTIVE MAINTENANCE ML SYSTEM - FULL TRAINING PIPELINE")
        logger.info("*" * 80)
        
        # Execute pipeline steps
        self.load_and_prepare_data()
        self.scale_features()
        self.handle_imbalance()
        self.train_logistic_regression_baseline()
        self.train_random_forest(tune=tune_models)
        self.train_xgboost(tune=tune_models)
        self.optimize_thresholds()
        self.save_best_model()
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
        logger.info(f"Best model: XGBoost")
        logger.info(f"Models saved to: {MODEL_SAVE_PATH}")
        logger.info("=" * 80)
        
        return {
            "models": self.models,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self.feature_names,
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main entry point for model training."""
    logger.info("Starting Predictive Maintenance Model Training")
    
    # Initialize trainer
    trainer = PredictiveMaintenanceTrainer()
    
    # Run full pipeline
    results = trainer.run_full_pipeline(tune_models=True)
    
    logger.info("\nTraining completed! Ready for evaluation.")


if __name__ == "__main__":
    main()