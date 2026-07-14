"""
Model Evaluation and Visualization Module.

This module provides:
- Comprehensive model evaluation
- Visualization generation (confusion matrix, ROC, PR curve, feature importance)
- Model comparison
- Results reporting

Visualizations are saved to outputs/plots/ for inclusion in reports and presentations.
"""

import logging
import pickle
import os
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import (
    TEST_DATA_PATH,
    TARGET_COLUMN,
    PLOTS_DIR,
    MODEL_SAVE_PATH,
    THRESHOLD_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
    SCALER_SAVE_PATH,
    COLORS,
    PLOT_CONFIG,
    RANDOM_STATE,
)
from utils import (
    setup_logging,
    load_data,
    extract_features,
    extract_target,
    compute_metrics,
    get_precision_recall_curve,
    get_roc_curve,
    save_metrics_to_json,
    get_confusion_matrix_labels,
)

# Initialize logger
logger = setup_logging()

# Configure matplotlib
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ============================================================================
# MODEL LOADING
# ============================================================================


def load_trained_model(model_path: str = MODEL_SAVE_PATH):
    """
    Load trained model from disk.
    
    Args:
        model_path (str): Path to saved model.
    
    Returns:
        Trained model object.
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully!")
    return model


def load_optimal_threshold(threshold_path: str = THRESHOLD_SAVE_PATH) -> float:
    """
    Load optimal threshold from disk.
    
    Args:
        threshold_path (str): Path to saved threshold.
    
    Returns:
        float: Optimal threshold value.
    """
    logger.info(f"Loading threshold from {threshold_path}")
    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)
    logger.info(f"Threshold loaded: {threshold:.3f}")
    return threshold


def load_feature_names(feature_names_path: str = FEATURE_NAMES_SAVE_PATH) -> list:
    """
    Load feature names from disk.
    
    Args:
        feature_names_path (str): Path to saved feature names.
    
    Returns:
        list: Feature names.
    """
    logger.info(f"Loading feature names from {feature_names_path}")
    with open(feature_names_path, "rb") as f:
        feature_names = pickle.load(f)
    logger.info(f"Loaded {len(feature_names)} feature names")
    return feature_names


def load_scaler(scaler_path: str = SCALER_SAVE_PATH):
    """
    Load fitted scaler from disk.
    
    Args:
        scaler_path (str): Path to saved scaler.
    
    Returns:
        Fitted scaler object.
    """
    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded successfully!")
    return scaler


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save confusion matrix heatmap.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        title (str): Plot title.
        save_path (str): Path to save plot. If None, saves to default location.
    """
    logger.info("Generating confusion matrix plot...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    
    # Labels and title
    ax.set_xlabel("Predicted Label", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_ylabel("True Label", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"], weight="bold", pad=20)
    ax.set_xticklabels(["No Fault", "Fault"], fontsize=PLOT_CONFIG["font_size"])
    ax.set_yticklabels(["No Fault", "Fault"], fontsize=PLOT_CONFIG["font_size"])
    
    # Add metrics as text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    textstr = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}"
    ax.text(
        0.98,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    logger.info(f"Confusion matrix plot saved to {save_path}")
    plt.close()


# ============================================================================
# ROC CURVE VISUALIZATION
# ============================================================================


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> float:
    """
    Plot and save ROC curve.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        save_path (str): Path to save plot.
    
    Returns:
        float: AUC score.
    """
    logger.info("Generating ROC curve plot...")
    
    # Compute ROC curve
    fpr, tpr, thresholds = get_roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])
    
    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        color=COLORS["neutral"],
        lw=3,
        label=f"ROC Curve (AUC = {auc:.3f})",
    )
    
    # Plot diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=2,
        linestyle="--",
        label="Random Classifier",
    )
    
    # Labels and formatting
    ax.set_xlabel("False Positive Rate", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_title("ROC Curve - Model Performance", fontsize=PLOT_CONFIG["title_size"], weight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=PLOT_CONFIG["font_size"])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    logger.info(f"ROC curve plot saved to {save_path}")
    plt.close()
    
    return auc


# ============================================================================
# PRECISION-RECALL CURVE VISUALIZATION
# ============================================================================


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    optimal_threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save Precision-Recall curve.
    
    This curve is especially important for imbalanced datasets.
    It shows the trade-off between precision and recall.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        optimal_threshold (float): Optimal threshold for visualization.
        save_path (str): Path to save plot.
    """
    logger.info("Generating Precision-Recall curve plot...")
    
    # Compute PR curve
    precision, recall, thresholds = get_precision_recall_curve(y_true, y_pred_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])
    
    # Plot PR curve
    ax.plot(
        recall,
        precision,
        color=COLORS["positive"],
        lw=3,
        label="Precision-Recall Curve",
    )
    
    # Mark optimal threshold
    y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
    opt_precision = precision_score(y_true, y_pred_opt, zero_division=0)
    opt_recall = recall_score(y_true, y_pred_opt, zero_division=0)
    
    ax.scatter(
        opt_recall,
        opt_precision,
        color=COLORS["negative"],
        s=200,
        zorder=5,
        label=f"Optimal Threshold ({optimal_threshold:.3f})",
        marker="*",
    )
    
    # Labels and formatting
    ax.set_xlabel("Recall", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_ylabel("Precision", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_title(
        "Precision-Recall Curve - Critical for Imbalanced Data",
        fontsize=PLOT_CONFIG["title_size"],
        weight="bold",
        pad=20,
    )
    ax.legend(loc="best", fontsize=PLOT_CONFIG["font_size"])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "precision_recall_curve.png")
    
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    logger.info(f"Precision-Recall curve plot saved to {save_path}")
    plt.close()


# ============================================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================================


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): Names of features.
        top_n (int): Number of top features to display.
        save_path (str): Path to save plot.
    """
    logger.info("Generating feature importance plot...")
    
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=PLOT_CONFIG["dpi"])
    
    # Create bar plot
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_importances, color=colors_gradient)
    
    # Labels and formatting
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=PLOT_CONFIG["font_size"])
    ax.set_xlabel("Feature Importance", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_title(
        f"Top {top_n} Most Important Features",
        fontsize=PLOT_CONFIG["title_size"],
        weight="bold",
        pad=20,
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=9,
        )
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    logger.info(f"Feature importance plot saved to {save_path}")
    plt.close()


# ============================================================================
# CLASS DISTRIBUTION VISUALIZATION
# ============================================================================


def plot_class_distribution(
    y_true: np.ndarray,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save class distribution.
    
    Args:
        y_true (np.ndarray): True labels.
        title (str): Plot title.
        save_path (str): Path to save plot.
    """
    logger.info("Generating class distribution plot...")
    
    # Count classes
    unique, counts = np.unique(y_true, return_counts=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=PLOT_CONFIG["dpi"])
    
    # Create bar plot
    colors_class = [COLORS["positive"], COLORS["negative"]]
    bars = ax.bar(
        ["No Fault", "Fault"],
        counts,
        color=colors_class[:len(counts)],
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(count)}\n({count/sum(counts)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )
    
    # Labels and formatting
    ax.set_ylabel("Count", fontsize=PLOT_CONFIG["label_size"], weight="bold")
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"], weight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    logger.info(f"Class distribution plot saved to {save_path}")
    plt.close()


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================


def evaluate_model_comprehensive(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimal_threshold: float = 0.5,
    feature_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        optimal_threshold (float): Optimal threshold.
        feature_names (list): Feature names for importance plot.
    
    Returns:
        Dict[str, Any]: Comprehensive evaluation results.
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Apply optimal threshold
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Compute metrics with default and optimal thresholds
    metrics_default = compute_metrics(y_test, y_pred, y_pred_proba)
    metrics_optimized = compute_metrics(y_test, y_pred_optimized, y_pred_proba)
    
    logger.info("\nMetrics with Default Threshold (0.5):")
    for metric, value in metrics_default.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\nMetrics with Optimal Threshold ({optimal_threshold:.3f}):")
    for metric, value in metrics_optimized.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_class_distribution(y_test, save_path=os.path.join(PLOTS_DIR, "test_class_distribution.png"))
    plot_confusion_matrix(y_test, y_pred_optimized, save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_pred_proba, save_path=os.path.join(PLOTS_DIR, "roc_curve.png"))
    plot_precision_recall_curve(y_test, y_pred_proba, optimal_threshold, save_path=os.path.join(PLOTS_DIR, "precision_recall_curve.png"))
    
    if feature_names is not None:
        plot_feature_importance(model, feature_names, save_path=os.path.join(PLOTS_DIR, "feature_importance.png"))
    
    # Prepare results
    results = {
        "metrics_default_threshold": metrics_default,
        "metrics_optimal_threshold": metrics_optimized,
        "optimal_threshold": optimal_threshold,
        "confusion_matrix": get_confusion_matrix_labels(y_test, y_pred_optimized),
    }
    
    # Save results
    save_metrics_to_json(results, os.path.join(os.path.dirname(PLOTS_DIR), "evaluation_results.json"))
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 80)
    
    return results


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================


class ModelEvaluator:
    """Comprehensive model evaluation pipeline."""
    
    def __init__(self):
        """Initialize evaluator and load artifacts."""
        logger.info("Initializing Model Evaluator")
        
        self.model = load_trained_model()
        self.optimal_threshold = load_optimal_threshold()
        self.feature_names = load_feature_names()
        self.scaler = load_scaler()
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.
        
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        logger.info("\n\n")
        logger.info("*" * 80)
        logger.info("MODEL EVALUATION PIPELINE")
        logger.info("*" * 80)
        
        # Load test data
        logger.info(f"Loading test data from {TEST_DATA_PATH}")
        df_test = load_data(TEST_DATA_PATH)
        X_test = extract_features(df_test).values
        y_test = extract_target(df_test).values
        
        # Scale features
        logger.info("Scaling test features...")
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluate model
        results = evaluate_model_comprehensive(
            self.model,
            X_test_scaled,
            y_test,
            self.optimal_threshold,
            self.feature_names,
        )
        
        return results


def main():
    """Main entry point for model evaluation."""
    logger.info("Starting Model Evaluation")
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate()
    
    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()