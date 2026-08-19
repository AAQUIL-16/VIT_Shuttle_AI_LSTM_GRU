import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from keras.models import load_model

app = FastAPI(
    title="Passenger-Centric Micro-Transit RL & ETA Engine",
    description="Inference Engine for Dwell-Inferred Crowding, ETA, and Passenger RL Decision Making",
    version="3.0"
)

# Load Model Artifacts
MODEL_PATH = os.path.join("model", "lstm_gru_eta_model.keras")
FEATURE_SCALER_PATH = os.path.join("model", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join("model", "target_scaler.pkl")

try:
    model = load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    print("✅ All AI model artifacts loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning loading model artifacts: {e}")
    model, feature_scaler, target_scaler = None, None, None


class ShuttleTelemetryPoint(BaseModel):
    latitude: float = Field(..., example=12.9692)
    longitude: float = Field(..., example=79.1559)
    speed_kmph: float = Field(..., example=18.5)
    distance_km: float = Field(..., example=1.2)
    segment_id: Optional[float] = Field(default=1.0, example=1.0)


class RLDecisionRequest(BaseModel):
    sequence: List[ShuttleTelemetryPoint]
    passenger_destination: Optional[str] = "Main Gate"
    walking_distance_km: Optional[float] = 1.0


def infer_latent_occupancy(sequence: List[ShuttleTelemetryPoint]) -> tuple:
    """
    Infers latent vehicle occupancy belief from telemetry dwell duration 
    and stop patterns without physical passenger sensors.
    """
    speeds = [pt.speed_kmph for pt in sequence]
    # Dwell points identified where speed is sub-walking (< 3 km/h)
    dwell_steps = sum(1 for s in speeds if s < 3.0)
    dwell_ratio = dwell_steps / len(speeds)

    if dwell_ratio >= 0.6:
        crowding_state = "High"
        boarding_prob = 0.25
        crowding_penalty = 8.0
    elif dwell_ratio >= 0.3:
        crowding_state = "Medium"
        boarding_prob = 0.70
        crowding_penalty = 3.0
    else:
        crowding_state = "Low"
        boarding_prob = 0.95
        crowding_penalty = 0.0

    return crowding_state, boarding_prob, crowding_penalty


def rl_passenger_policy(eta_minutes: float, boarding_prob: float, crowding_penalty: float, walking_dist_km: float) -> str:
    """
    Evaluates expected reward utilities across actions: a in {Board, Wait, Walk}
    R(a) = - (Travel_Time + Expected_Delay + Crowding_Disutility)
    """
    # Action 1: Walk (assuming average 4.8 km/h campus walking pace)
    t_walk = (walking_dist_km / 4.8) * 60.0
    q_walk = -t_walk

    # Action 2: Board Targeted Shuttle
    # Expected wait/travel penalized by boarding failure risk
    expected_wait = (1.0 - boarding_prob) * 12.0  # Assumes 12-min penalty if left behind
    q_board = -(eta_minutes + expected_wait + crowding_penalty)

    # Action 3: Wait for Next Alternative Shuttle
    q_wait = -(eta_minutes + 7.0 + 1.0)  # Next dispatch headway estimate

    # Select optimal policy action
    utilities = {"Board": q_board, "Wait": q_wait, "Walk": q_walk}
    best_action = max(utilities, key=utilities.get)
    return best_action


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "system": "VIT Micro-Transit Latent Occupancy & RL Decision System",
        "model_loaded": model is not None,
        "endpoint": "/predict_eta"
    }


@app.post("/predict_eta")
def predict_eta_and_decision(data: RLDecisionRequest):
    if len(data.sequence) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Recurrent input sequence must contain exactly 10 time steps, received {len(data.sequence)}."
        )

    # 1. Infer Latent Crowding and Boarding Probability
    crowding_state, boarding_prob, crowding_penalty = infer_latent_occupancy(data.sequence)

    # 2. Extract 4 training features matching feature_scaler.pkl
    raw_seq = [
        [pt.latitude, pt.longitude, pt.speed_kmph, pt.distance_km]
        for pt in data.sequence
    ]
    raw_seq_np = np.array(raw_seq)

    # 3. Compute ETA with fallback if model is uninitialized
    predicted_minutes = 5.0
    if model is not None and feature_scaler is not None and target_scaler is not None:
        try:
            scaled_seq = feature_scaler.transform(raw_seq_np)
            input_3d = np.expand_dims(scaled_seq, axis=0)
            scaled_prediction = model.predict(input_3d, verbose=0)
            unscaled_eta = target_scaler.inverse_transform(scaled_prediction)
            predicted_minutes = float(np.ravel(unscaled_eta)[0])
            predicted_minutes = max(0.5, round(predicted_minutes, 2))
        except Exception as e:
            print(f"Inference error: {e}")
            predicted_minutes = max(0.5, round(raw_seq_np[-1][3] / 0.25, 2))
    else:
        # Fallback estimation based on distance
        predicted_minutes = max(0.5, round(raw_seq_np[-1][3] / 0.25, 2))

    # 4. Compute Reinforcement Learning Recommendation
    walking_dist = data.walking_distance_km if data.walking_distance_km else 1.0
    recommendation = rl_passenger_policy(predicted_minutes, boarding_prob, crowding_penalty, walking_dist)

    return {
        "status": "success",
        "predicted_travel_time_min": predicted_minutes,
        "recommendation": recommendation,
        "crowding_state": crowding_state,
        "boarding_prob": boarding_prob,
        "unit": "minutes"
    }
