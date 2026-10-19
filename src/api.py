
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.database import get_recent_predictions, save_prediction


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "logistic_regression_baseline.joblib"

CLASSIFICATION_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.80
EXPLANATION_FEATURE_COUNT = 5


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

saved = joblib.load(MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]

classifier = model.named_steps["classifier"]
scaler = model.named_steps["scaler"]
imputer = model.named_steps["imputer"]

coefficients = classifier.coef_[0]


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI-Based Network Threat Detection API",
    description=(
        "API for network traffic threat detection and analysis. "
        "Classification threshold: 0.60. "
        "Explanations summarize model-based feature contributions."
    ),
    version="1.2.0",
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    timestamp: str


class PredictionRequest(BaseModel):
    flow_id: str = Field(min_length=1, max_length=100)
    features: dict[str, float]


# --------------------------------------------------
# Prediction explanation
# --------------------------------------------------

def explain_prediction(feature_values, attack_probability):
    """
    Estimate feature contributions to the Logistic Regression
    decision score using standardized feature values.

    Positive contributions support the attack class.
    Negative contributions support the benign class.

    These are approximate model contributions, not causal evidence.
    """

    values = np.asarray(
        feature_values,
        dtype=float,
    ).reshape(1, -1)

    if values.shape[1] != len(feature_columns):
        raise ValueError(
            f"Expected {len(feature_columns)} features, "
            f"received {values.shape[1]}."
        )

    # Preserve feature names expected by the fitted imputer.
    feature_frame = pd.DataFrame(
        values,
        columns=feature_columns,
    )

    # Apply the same fitted preprocessing used during training.
    imputed = imputer.transform(feature_frame)
    standardized = scaler.transform(imputed)[0]

    # Approximate contributions to the linear decision score.
    contributions = standardized * coefficients

    strongest = np.argsort(
        np.abs(contributions)
    )[::-1]

    details = []

    for index in strongest[:EXPLANATION_FEATURE_COUNT]:
        contribution = float(contributions[index])

        if contribution > 0:
            direction = "supports attack"
        elif contribution < 0:
            direction = "supports benign"
        else:
            direction = "has negligible contribution"

        details.append(
            {
                "feature": feature_columns[index],
                "contribution": round(contribution, 4),
                "direction": direction,
            }
        )

    if attack_probability >= CLASSIFICATION_THRESHOLD:
        summary = (
            "Classified as attack because the estimated attack "
            f"probability ({attack_probability:.4f}) meets or exceeds "
            f"the {CLASSIFICATION_THRESHOLD:.2f} threshold."
        )
    else:
        summary = (
            "Classified as benign because the estimated attack "
            f"probability ({attack_probability:.4f}) is below "
            f"the {CLASSIFICATION_THRESHOLD:.2f} threshold."
        )

    return {
        "summary": summary,
        "method": (
            "Top absolute standardized-feature contributions from the "
            "fitted Logistic Regression coefficients; not causal evidence."
        ),
        "top_features": details,
    }


# --------------------------------------------------
# General endpoints
# --------------------------------------------------

@app.get("/", tags=["General"])
def home():
    return {
        "message": "Network Threat Detection API is running",
        "docs": "/docs",
        "health": "/health",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
)
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------
# Network traffic prediction endpoint
# --------------------------------------------------

@app.post("/predict", tags=["Detection"])
def predict(request: PredictionRequest):
    received = set(request.features)
    expected = set(feature_columns)

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    # Validate exact feature names.
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Feature names do not match the trained model."
                ),
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    try:
        values = np.asarray(
            [
                request.features[name]
                for name in feature_columns
            ],
            dtype=float,
        )

        # Reject NaN and infinite values.
        if not np.isfinite(values).all():
            raise HTTPException(
                status_code=422,
                detail="Feature values must be finite numbers.",
            )

        # Preserve feature names during preprocessing.
        row = pd.DataFrame(
            [values],
            columns=feature_columns,
        )

        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        attack_index = classes.index(1)

        attack_probability = float(
            probabilities[attack_index]
        )

        # Classification threshold is 0.60.
        predicted_class = int(
            attack_probability >= CLASSIFICATION_THRESHOLD
        )

        prediction_label = (
            "attack" if predicted_class == 1 else "benign"
        )

        # Risk bands remain independent of the classification threshold.
        if attack_probability >= HIGH_RISK_THRESHOLD:
            risk_level = "high"
        elif attack_probability >= MEDIUM_RISK_THRESHOLD:
            risk_level = "medium"
        else:
            risk_level = "low"

        explanation = explain_prediction(
            values,
            attack_probability,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        model_name = (
            "Logistic Regression baseline (threshold=0.60)"
        )

        # Store the complete explanation as JSON in the existing
        # explanation TEXT column. No database migration is required.
        explanation_json = json.dumps(
            explanation,
            ensure_ascii=False,
            allow_nan=False,
        )

        try:
            record_id = save_prediction(
                flow_id=request.flow_id,
                prediction=prediction_label,
                predicted_class=predicted_class,
                attack_probability=round(
                    attack_probability, 4
                ),
                risk_level=risk_level,
                model_name=model_name,
                created_at=timestamp,
                explanation=explanation_json,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Prediction was calculated but "
                    "could not be saved."
                ),
            ) from exc

        return {
            "record_id": record_id,
            "flow_id": request.flow_id,
            "prediction": prediction_label,
            "predicted_class": predicted_class,
            "attack_probability": round(
                attack_probability, 4
            ),
            "classification_threshold": (
                CLASSIFICATION_THRESHOLD
            ),
            "risk_level": risk_level,
            "model": model_name,
            "timestamp": timestamp,
            "explanation": explanation,
            "saved_to_database": True,
        }

    except HTTPException:
        raise

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process feature values: {exc}",
        ) from exc


# --------------------------------------------------
# Prediction history endpoint
# --------------------------------------------------

@app.get("/predictions", tags=["Alerts"])
def list_predictions(
    limit: int = Query(default=20, ge=1, le=200),
):
    try:
        records = get_recent_predictions(limit)

        return {
            "count": len(records),
            "predictions": records,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve saved predictions.",
        ) from exc
