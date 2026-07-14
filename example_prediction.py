"""
Example: Using the Trained Predictive Maintenance Model

This script demonstrates how to:
1. Load trained model artifacts
2. Make predictions on new sensor data
3. Interpret results for maintenance decisions

Usage:
    python example_prediction.py
"""

import pickle
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

def load_artifacts():
    """Load all trained model artifacts from disk."""
    
    models_dir = Path(__file__).parent / "models"
    
    print("Loading model artifacts...")
    
    # Load model
    model = joblib.load(models_dir / "best_model.joblib")
    print(f"✓ Model loaded: {model.__class__.__name__}")
    
    # Load threshold
    with open(models_dir / "optimal_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    print(f"✓ Optimal threshold loaded: {threshold:.3f}")
    
    # Load feature names
    with open(models_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print(f"✓ Feature names loaded: {len(feature_names)} features")
    
    # Load scaler
    scaler = joblib.load(models_dir / "scaler.joblib")
    print(f"✓ Scaler loaded: {scaler.__class__.__name__}")
    
    return model, threshold, feature_names, scaler


# ============================================================================
# MAKE PREDICTIONS
# ============================================================================


def predict_single_machine(
    sensor_readings: dict,
    model,
    threshold: float,
    feature_names: list,
    scaler,
) -> dict:
    """
    Predict machinery failure for a single machine.
    
    Args:
        sensor_readings (dict): {feature_name: value}
        model: Trained XGBoost model
        threshold (float): Decision threshold
        feature_names (list): Expected feature names
        scaler: Fitted StandardScaler
    
    Returns:
        dict: Prediction results with confidence and recommendation
    """
    
    # Validate features
    if set(sensor_readings.keys()) != set(feature_names):
        missing = set(feature_names) - set(sensor_readings.keys())
        extra = set(sensor_readings.keys()) - set(feature_names)
        raise ValueError(f"Feature mismatch! Missing: {missing}, Extra: {extra}")
    
    # Prepare features in correct order
    X = np.array([sensor_readings[f] for f in feature_names]).reshape(1, -1)
    
    # Scale using fitted scaler
    X_scaled = scaler.transform(X)
    
    # Get prediction probability
    failure_probability = model.predict_proba(X_scaled)[0, 1]
    
    # Apply optimal threshold
    will_fail = failure_probability >= threshold
    
    # Determine confidence
    if will_fail:
        confidence = failure_probability
        confidence_pct = confidence * 100
        recommendation = "🔴 URGENT: Schedule maintenance immediately!"
        risk_level = "HIGH"
    else:
        confidence = 1 - failure_probability
        confidence_pct = confidence * 100
        recommendation = "🟢 NORMAL: Continue monitoring"
        risk_level = "LOW"
    
    return {
        "will_fail": bool(will_fail),
        "failure_probability": float(failure_probability),
        "confidence": float(confidence),
        "confidence_percentage": confidence_pct,
        "risk_level": risk_level,
        "decision_threshold": threshold,
        "recommendation": recommendation,
    }


def predict_batch(
    sensor_data_list: list[dict],
    model,
    threshold: float,
    feature_names: list,
    scaler,
) -> list[dict]:
    """
    Predict for multiple machines.
    
    Args:
        sensor_data_list (list): List of {feature_name: value} dicts
        model: Trained model
        threshold: Decision threshold
        feature_names: Expected features
        scaler: Fitted scaler
    
    Returns:
        list: List of predictions for each machine
    """
    predictions = []
    for sensor_data in sensor_data_list:
        pred = predict_single_machine(sensor_data, model, threshold, feature_names, scaler)
        predictions.append(pred)
    
    return predictions


# ============================================================================
# MAIN EXAMPLE
# ============================================================================


def main():
    """Run example predictions."""
    
    print("=" * 80)
    print("PREDICTIVE MAINTENANCE MODEL - PREDICTION EXAMPLE")
    print("=" * 80)
    
    # Load artifacts
    model, threshold, feature_names, scaler = load_artifacts()
    
    # Create example sensor readings (normalized values)
    # In production, these would come from real sensor data
    example_normal_machine = {
        f: np.random.normal(0, 1) for f in feature_names
    }
    
    example_failing_machine = {
        f: np.random.normal(2, 1) for f in feature_names  # Shifted values = anomaly
    }
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Normal Operating Machine")
    print("=" * 80)
    
    result1 = predict_single_machine(
        example_normal_machine,
        model,
        threshold,
        feature_names,
        scaler,
    )
    
    print(f"Failure Probability: {result1['failure_probability']:.4f}")
    print(f"Confidence: {result1['confidence_percentage']:.1f}%")
    print(f"Risk Level: {result1['risk_level']}")
    print(f"Recommendation: {result1['recommendation']}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Machine Showing Anomalies (Potential Failure)")
    print("=" * 80)
    
    result2 = predict_single_machine(
        example_failing_machine,
        model,
        threshold,
        feature_names,
        scaler,
    )
    
    print(f"Failure Probability: {result2['failure_probability']:.4f}")
    print(f"Confidence: {result2['confidence_percentage']:.1f}%")
    print(f"Risk Level: {result2['risk_level']}")
    print(f"Recommendation: {result2['recommendation']}")
    
    print("\n" + "=" * 80)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 80)
    
    # Create batch of 5 machines
    batch_data = [
        {f: np.random.normal(0, 1) for f in feature_names}
        for _ in range(5)
    ]
    
    results = predict_batch(batch_data, model, threshold, feature_names, scaler)
    
    print(f"\nPredictions for {len(results)} machines:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Machine {i}:")
        print(f"  Failure Probability: {result['failure_probability']:.4f}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Action: {result['recommendation']}")
        print()
    
    # Summary
    failures_detected = sum(1 for r in results if r['will_fail'])
    print(f"Summary: {failures_detected}/{len(results)} machines require maintenance")
    
    print("\n" + "=" * 80)
    print("INTEGRATION WITH MONITORING SYSTEM")
    print("=" * 80)
    

if __name__ == "__main__":
    main()