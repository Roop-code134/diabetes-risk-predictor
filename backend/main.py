import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
pipe = joblib.load(ROOT / "model" / "model.joblib")
info = json.loads((ROOT / "model" / "info.json").read_text())
FEATURES = info["features"]
THRESHOLD = info["threshold"]

LABELS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Blood glucose",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin level",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family history score",
    "Age": "Age",
}

app = FastAPI(title="Diabetes Risk Predictor", version="1.0")

# Lets the JavaScript page (served from another address) call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Patient(BaseModel):
    # Impossible values are rejected automatically (HTTP 422)
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: Optional[float] = Field(default=None, ge=40, le=400, description="mg/dL")
    BloodPressure: Optional[float] = Field(default=None, ge=30, le=200, description="diastolic, mm Hg")
    SkinThickness: Optional[float] = Field(default=None, ge=5, le=100, description="mm")
    Insulin: Optional[float] = Field(default=None, ge=10, le=900, description="mu U/ml")
    BMI: Optional[float] = Field(default=None, ge=10, le=70)
    DiabetesPedigreeFunction: float = Field(ge=0.05, le=2.5)
    Age: int = Field(ge=21, le=100)


@app.get("/health")
def health():
    return {"status": "ok", "model": info["model"]}


@app.get("/model-info")
def model_info():
    return info


@app.post("/predict")
def predict(patient: Patient):
    # Missing optional values stay empty; the model fills them with the median
    X = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)

    prob = float(pipe.predict_proba(X)[0, 1])
    flagged = prob >= THRESHOLD

    # Why this score: each feature's push on the risk (same math as explain.py)
    Z = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X))
    contrib = Z[0] * pipe.named_steps["clf"].coef_[0]
    ranked = sorted(zip(FEATURES, contrib), key=lambda p: abs(p[1]), reverse=True)

    factors = [
        {
            "feature": LABELS[name],
            "effect": "raises risk" if value > 0 else "lowers risk",
            "strength": round(float(abs(value)), 2),
        }
        for name, value in ranked[:4]
    ]

    return {
        "probability": round(prob, 3),
        "flagged": flagged,
        "risk_level": "Higher risk" if flagged else "Lower risk",
        "threshold": THRESHOLD,
        "top_factors": factors,
        "disclaimer": "Educational demo only. Not a medical diagnosis.",
    }