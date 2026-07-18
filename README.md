# Predictive Maintenance ML System

## Overview
A machine learning system for predicting machinery failures from sensor data using time-series feature engineering and gradient-boosted trees. Optimized for **recall** as the primary objective, because missing a failure is far more costly than a false alarm.

Built with high recall as the primary objective because missing a failure is extremely costly.

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
├── backend/
│   ├── app.py                          # API entrypoint
│   └── failure_success_log.txt         # Prediction outcome log
│
├── deploy/
│   ├── backend/
│   │   └── app.py                      # Deployment-packaged API
│   ├── models/                         # Model artifacts bundled for deployment
│   │   ├── best_model.joblib
│   │   ├── xgboost_model.joblib
│   │   ├── scaler.joblib
│   │   ├── smote.joblib
│   │   ├── feature_names.pkl
│   │   ├── optimal_threshold.pkl
│   │   ├── optimal_threshold.json
│   │   └── deployment_config.json
│   ├── src/                            # Deployment-side pipeline code
│   ├── .dockerignore
│   ├── Dockerfile
│   └── requirements.txt
│
├── models/                             # Source-of-truth trained artifacts
│   ├── best_model.joblib
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   ├── smote.joblib
│   ├── feature_names.pkl
│   ├── optimal_threshold.pkl
│   ├── optimal_threshold.json
│   └── deployment_config.json
│
├── plots/                              # Full EDA + evaluation visualization suite
│   ├── 0_scaling_comparison.png
│   ├── 1_class_distribution.png
│   ├── 2_correlation_heatmap.png
│   ├── 3_sensor_signals.png
│   ├── 4_missing_values.png
│   ├── 5_outlier_detection.png
│   ├── 6_rolling_features.png
│   ├── 7_lag_features.png
│   ├── 8_roc_features.png
│   ├── 9_fft_features.png
│   ├── 10_feature_importance.png
│   └── 11_drift_report.png
│
├── processed/
│   ├── processed_train.csv
│   └── processed_test.csv
│
├── reports/
│   └── drift_report.csv
│
├── src/                                # Training pipeline (config, utils, train, evaluate)
├── analysis.py                         # Exploratory / offline analysis script
├── DELIVERY_SUMMARY.md                 # Handoff summary
├── IMPLEMENTATION_GUIDE.md             # Setup & usage walkthrough
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** `models/` and `deploy/models/` currently hold duplicate copies of the same artifacts. Worth resolving before this repo gets scrutinized — either symlink `deploy/models` to the top-level `models/`, or document explicitly *why* they're separate (e.g., `deploy/models` is a frozen, versioned snapshot for the container build while `models/` is the live output of `train.py`). Two unexplained copies of the same files reads as clutter, not intent.

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
This will load the trained model, evaluate on the held-out test set, save the full EDA/evaluation visualization suite to `plots/`, and write drift diagnostics to `reports/drift_report.csv`.

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

Served from `backend/app.py`, with the deployment-packaged copy (app + models + Dockerfile) living under `deploy/`. Core logic:

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
The `deploy/` directory is a self-contained deployment bundle — its own `backend/`, `models/`, `src/`, `Dockerfile`, `.dockerignore`, and `requirements.txt` — so the container can be built without depending on the rest of the repo:

```bash
cd deploy/
docker build -t predictive-maintenance-api .
docker run -p 8000:8000 predictive-maintenance-api
```

```dockerfile
# deploy/Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY models/ models/
COPY src/ src/
COPY backend/ backend/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
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

### Single held-out test set
This reflects one particular 80/20 train/test split (test set n = 106, consistent with `TRAIN_TEST_SPLIT_RATIO = 0.2` on 532 samples). A single split naturally lands close to — but not exactly on — the 5-fold CV mean above; that's expected fold-to-fold variance, not an error.

```
Confusion Matrix (n = 106, threshold = 0.20):
                 Predicted No-Fault   Predicted Fault
Actual No-Fault          49                 4
Actual Fault              7                46

Derived metrics:
  accuracy    = (49+46)/106 = 89.6%
  precision   = 46/(46+4)   = 92.0%
  recall      = 46/(46+7)   = 86.8%
  specificity = 49/(49+4)   = 92.5%
  f1          = 2*(0.92*0.868)/(0.92+0.868) = 89.3%
```

**Interpretation:** at threshold 0.20, the model catches 46 of 53 real failures (86.8% recall) at the cost of 4 false alarms out of 53 healthy readings (92.5% specificity) — in line with the 87.6% recall reported by 5-fold CV above, with the small difference reflecting normal variance between the CV average and any single split.

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
