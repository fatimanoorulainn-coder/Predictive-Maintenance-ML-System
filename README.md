# Predictive Maintenance ML System

## Overview
A machine learning system for predicting machinery failures from sensor data using time-series feature engineering and gradient-boosted trees. Optimized for **recall** as the primary objective, because missing a failure is far more costly than a false alarm.

> **Note on metrics below:** numbers in this README are illustrative placeholders reflecting realistic performance for a ~530-sample sensor dataset. Replace every number in this file with your actual `outputs/evaluation_results.json` / `outputs/model_metrics.json` output once you run `train.py` and `evaluate.py` on your real data — do not present placeholder numbers as final results.

## Key Features
- **High Recall Focus:** Optimized to catch failures while keeping precision usable
- **Imbalance Handling:** SMOTE oversampling + class weights for imbalanced fault data
- **Multiple Models:** Logistic Regression baseline, Random Forest, XGBoost
- **Threshold Optimization:** Decision threshold tuned to trade precision for recall
- **Production Ready:** Logging, error handling, model serialization
- **Reproducibility:** Fixed random seeds, configuration-driven design
- **Comprehensive Evaluation:** Cross-validated metrics, confusion matrix, ROC/PR curves, feature importance

## Project Structure
```
project/
├── data/
│   └── sensor_data.csv                 # Raw sensor data
│
├── processed/
│   ├── processed_train.csv             # Preprocessed training data
│   └── processed_test.csv              # Preprocessed test data
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration parameters
│   ├── utils.py                        # Utility functions
│   ├── train.py                        # Training pipeline
│   └── evaluate.py                     # Evaluation & visualization
│
├── models/
│   ├── best_model.joblib               # Trained XGBoost model
│   ├── optimal_threshold.pkl           # Optimal decision threshold
│   ├── feature_names.pkl               # Feature names
│   ├── scaler.joblib                   # Fitted scaler
│   └── smote.joblib                    # SMOTE transformer
│
├── outputs/
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   └── feature_importance.png
│   ├── logs/
│   │   └── training.log
│   ├── model_metrics.json
│   └── evaluation_results.json
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
```bash
cd project/
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models
```bash
python src/train.py
```
This will:
- Load and validate data
- Split into stratified train/test sets
- Apply SMOTE for imbalance handling
- Train and cross-validate 3 models (Logistic Regression, Random Forest, XGBoost)
- Perform hyperparameter tuning
- Optimize the decision threshold for recall
- Save the best model and artifacts

### 2. Evaluate Models
```bash
python src/evaluate.py
```
This will load the trained model, evaluate on the held-out test set, and save plots to `outputs/plots/`.

## Engineering Design Decisions

### 1. Why Recall Is the Primary Metric
| Scenario | Cost | Impact |
|---|---|---|
| Miss failure (False Negative) | HIGH | Equipment damage, production loss, safety hazard |
| False alarm (False Positive) | LOW | Unnecessary inspection, minor operational cost |

**Action:** Tune the threshold to raise recall while keeping precision in a usable range (target: precision ≥ 0.65).

### 2. Why Threshold Tuning Matters
The default 0.5 probability threshold assumes a balanced dataset and equal misclassification costs — neither holds here. Lowering the threshold trades some precision for higher recall, which matches the real cost asymmetry between missed failures and false alarms.

### 3. Why XGBoost for Tabular Sensor Data
| Aspect | XGBoost Advantage |
|---|---|
| Performance | Strong default choice for tabular data at this scale |
| Speed | Fast to train and tune on ~500 samples |
| Feature interactions | Captures non-linear sensor relationships |
| Native imbalance handling | `scale_pos_weight` parameter |
| Interpretability | Feature importance, SHAP values available |
| Production | Small model size, fast inference |

**Why not deep learning?** ~530 samples is too little data for a neural net to reliably outperform gradient boosting, and it adds training/debugging overhead without a clear payoff at this scale.

### 4. SMOTE + Class Weights
- **SMOTE:** creates synthetic minority-class samples to balance the training set
- **Class weights:** penalizes false negatives directly during training
- Combined, these attack the imbalance problem from both the data side and the loss-function side.

## Configuration
```python
# src/config.py

TARGET_COLUMN = "fault_label"
TRAIN_TEST_SPLIT_RATIO = 0.2

SMOTE_CONFIG = {
    "sampling_strategy": 0.5,
    "random_state": 42,
    "k_neighbors": 5,
}

THRESHOLD_CONFIG = {
    "min_threshold": 0.1,
    "max_threshold": 0.9,
    "step": 0.01,
    "min_precision": 0.65,
    "target_recall": 0.85,
}

GRID_SEARCH_CONFIG = {
    "cv": 5,
    "scoring": "recall",
    "n_jobs": -1,
}
```

## Model Usage in Production (FastAPI Example)
```python
import joblib
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")
with open("models/optimal_threshold.pkl", "rb") as f:
    optimal_threshold = pickle.load(f)
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

class SensorData(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    will_fail: bool
    failure_probability: float
    confidence: float
    recommended_action: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "predictive_maintenance_v1"}

@app.post("/predict")
def predict(data: SensorData) -> PredictionResponse:
    if set(data.features.keys()) != set(feature_names):
        return {"error": "Feature mismatch"}

    X = np.array([data.features[f] for f in feature_names]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    failure_prob = model.predict_proba(X_scaled)[0, 1]
    will_fail = failure_prob >= optimal_threshold

    if will_fail:
        confidence, action = failure_prob, "SCHEDULE MAINTENANCE SOON"
    else:
        confidence, action = 1 - failure_prob, "CONTINUE MONITORING"

    return PredictionResponse(
        will_fail=bool(will_fail),
        failure_probability=float(failure_prob),
        confidence=float(confidence),
        recommended_action=action,
    )
```

```bash
pip install fastapi uvicorn
python -m uvicorn main:app --reload --port 8000

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {...feature values...}}'
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY models/ models/
COPY src/ src/
COPY main.py .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Key Metrics Explained
- **Accuracy** = (TP + TN) / Total — overall correctness; misleading on imbalanced data.
- **Precision** = TP / (TP + FP) — of predicted failures, how many were real?
- **Recall (Sensitivity)** = TP / (TP + FN) — of all real failures, how many did we catch? **Primary metric here.**
- **F1** = 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC** — threshold-independent ranking quality (1.0 = perfect, 0.5 = random). On a dataset this size, values much above ~0.95 usually indicate leakage or an unrealistically easy split rather than a stronger model.
- **Specificity** = TN / (TN + FP) — true negative rate.

## Evaluation Results

### Cross-validated performance (5-fold, stratified)
Reported as mean over folds — this is what the resume bullet should quote, since it reflects generalization rather than a single lucky split.

| Metric | Mean (5-fold CV) |
|---|---|
| Accuracy | 89.4% |
| Precision | 73% |
| Recall | 87.6% |
| F1 | 79.6% |
| ROC-AUC | 0.93 |

### Single held-out test set (illustrative — replace with your real run)
This is a separate measurement from a specific train/test split, shown here so the confusion matrix and derived metrics are self-consistent (unlike the earlier draft, where the confusion matrix didn't match the reported accuracy/specificity).

```
Confusion Matrix (n = 206, threshold = 0.20):
                 Predicted No-Fault   Predicted Fault
Actual No-Fault         140                18
Actual Fault              6                42

Derived metrics:
  accuracy    = (140+42)/206 = 88.3%
  precision   = 42/(42+18)   = 70.0%
  recall      = 42/(42+6)    = 87.5%
  specificity = 140/(140+18) = 88.6%
  f1          = 2*(0.70*0.875)/(0.70+0.875) = 77.8%
```

**Interpretation:** at threshold 0.20, the model catches 42 of 48 real failures (87.5% recall) at the cost of 18 false alarms out of 158 healthy readings (88.6% specificity) — a reasonable trade given false negatives cost far more than false alarms.

## Troubleshooting

**`FileNotFoundError: processed_train.csv not found`**
Ensure processed data is in `processed/`: `ls -la processed/processed_train.csv`

**`Module not found: xgboost`**
`pip install -r requirements.txt`

**Training is slow**
Reduce the grid search scope in `config.py`:
```python
RF_TUNING_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15],
}
```

**Low recall despite threshold tuning**
- Check data quality and class distribution
- Lower `min_precision` in `THRESHOLD_CONFIG`
- Increase SMOTE `sampling_strategy`

## Monitoring and Maintenance

**Track in production:**
- Recall (alert if it drops below ~80%)
- Precision / false-alarm rate
- Prediction distribution drift over time
- Actual outcomes vs. predictions (feedback loop)

**Retrain when:**
- Recall drops meaningfully below the validated baseline
- Data distribution shifts
- New failure modes are discovered
- On a regular schedule (e.g., quarterly)

**Feedback logging example:**
```json
{
    "timestamp": "2026-06-15T10:30:00Z",
    "machine_id": "M001",
    "predicted_failure": true,
    "predicted_probability": 0.62,
    "actual_failure": true,
    "maintenance_performed": true,
    "feedback": "CORRECT_POSITIVE"
}
```

## References
- SMOTE Paper: Chawla et al. (2002) — "SMOTE: Synthetic Minority Over-sampling Technique"
- XGBoost: https://xgboost.readthedocs.io/
- Imbalanced-learn: https://imbalanced-learn.org/
- Threshold optimization: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

## License
This project is provided as-is for educational use.

## Changelog

### Version 1.0
- Initial release
- SMOTE + class weights for imbalance handling
- Threshold tuning for recall optimization
- Logistic Regression, Random Forest, XGBoost models
- Cross-validated evaluation and visualization
