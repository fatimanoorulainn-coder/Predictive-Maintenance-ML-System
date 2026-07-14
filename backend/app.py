from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import json

app = FastAPI()

# -------------------
# INPUT SCHEMA (NEW - CLEAN API)
# -------------------
class InputData(BaseModel):
    data: List[float]

# -------------------
# LOAD ARTIFACTS
# -------------------
model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")

feature_names = joblib.load("models/feature_names.pkl")

with open("models/optimal_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# -------------------
# HOME ROUTE
# -------------------
@app.get("/")
def home():
    return {"status": "ML API running"}

# -------------------
# PREDICTION ROUTE (CLEAN VERSION)
# -------------------
@app.post("/predict")
def predict(input: InputData):

    try:
        x = input.data
        expected = len(feature_names)

        # -------------------
        # AUTO-PADDING LOGIC
        # -------------------

        # If too short → pad with zeros
        if len(x) < expected:
            x = x + [0] * (expected - len(x))

        # If too long → truncate
        elif len(x) > expected:
            x = x[:expected]

        # Convert to numpy
        X = np.array(x).reshape(1, -1)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = 1 if prob >= threshold else 0

        return {
            "probability": float(prob),
            "prediction": int(prediction),
            "status": "Failure" if prediction == 1 else "Normal"
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.post("/test")
def test_cases():
    test_inputs = [
        [0.1] * len(feature_names),   # likely Normal
        [0.5] * len(feature_names),   # uncertain
        [3.0] * len(feature_names)    # likely Failure
    ]

    results = []

    for i, x in enumerate(test_inputs):

        X = np.array(x).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        prediction = 1 if prob >= threshold else 0

        reason = (
            "Low anomaly score → normal behavior"
            if prediction == 0
            else "High anomaly score → failure pattern detected"
        )

        results.append({
            "case": i + 1,
            "prediction": "Failure" if prediction == 1 else "Normal",
            "probability": float(prob),
            "reason": reason
        })

    return results
    
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Predictive Maintenance Dashboard</title>
        <style>
            body {
                font-family: Arial;
                background: #0f172a;
                color: white;
                text-align: center;
                padding-top: 50px;
            }

            input {
                padding: 10px;
                width: 300px;
                border-radius: 8px;
                border: none;
            }

            button {
                padding: 10px 20px;
                border-radius: 8px;
                border: none;
                background: #22c55e;
                color: white;
                cursor: pointer;
                margin-left: 10px;
            }

            #result {
                margin-top: 20px;
                font-size: 20px;
            }
        </style>
    </head>

    <body>
        <h1>🔧 Machine Failure Prediction</h1>

        <form id="form">
            <input id="data" placeholder="Enter sensor values (comma separated)" />
            <button type="submit">Predict</button>
        </form>

        <h2 id="result"></h2>

     <script>
    const form = document.getElementById('form');

    form.onsubmit = async (e) => {
        e.preventDefault();

        const raw = document.getElementById('data').value;
        const arr = raw.split(',').map(x => parseFloat(x.trim()));

        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: arr })
        });

        const data = await res.json();

        document.getElementById('result').innerText =
            `Status: True`;
    };
</script>
    </body>
    </html>
    """