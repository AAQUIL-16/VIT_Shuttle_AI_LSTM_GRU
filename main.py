import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from keras.models import load_model

app = FastAPI(
    title="Micro-Transit Latent Occupancy & EPA Recommendation Engine",
    description="LSTM-GRU ETA prediction paired with Hardware-Free Latent Occupancy and Prescriptive Arbitration",
    version="3.0"
)

# Load AI Model Artifacts
MODEL_PATH = os.path.join("model", "lstm_gru_eta_model.keras")
FEATURE_SCALER_PATH = os.path.join("model", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join("model", "target_scaler.pkl")

try:
    model = load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    print("✅ LSTM-GRU Model and Scalers loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Running in fallback mode ({e})")
    model, feature_scaler, target_scaler = None, None, None

# Schema Definitions
class TelemetryPoint(BaseModel):
    latitude: float
    longitude: float
    speed_kmph: float
    distance_km: float
    segment_id: float = 1.0
    dwell_time_sec: float = Field(default=0.0, description="Stop dwell time in seconds")

class EPARequest(BaseModel):
    sequence: List[TelemetryPoint]
    user_walking_distance_km: float = Field(..., example=0.6)
    bus_capacity_limit: int = Field(default=40, example=40)

# =====================================================================
# NOVEL PILLAR 1: HARDWARE-FREE LATENT OCCUPANCY INFERENCE
# =====================================================================
def infer_latent_occupancy(dwell_time_sec: float, speed_kmph: float, capacity: int):
    """
    Estimates hidden crowding state and boarding probability from GPS dwell dynamics
    without physical passenger counter hardware.
    """
    if speed_kmph > 3.0:
        # Shuttle in motion -> Dwell baseline zero
        boarding_rate = 0.0
    else:
        # Non-linear boarding queue estimate (approx 2.5s per passenger boarding/alighting)
        boarding_rate = dwell_time_sec / 2.5 

    estimated_occupancy = min(capacity, int(boarding_rate * 3)) # Scaled occupancy estimate
    occupancy_ratio = estimated_occupancy / capacity

    # Boarding Failure Risk (P_fail): Sigmoid transformation of occupancy ratio
    p_fail = 1.0 / (1.0 + np.exp(-10 * (occupancy_ratio - 0.75)))
    
    return {
        "estimated_occupancy": estimated_occupancy,
        "occupancy_ratio": round(float(occupancy_ratio), 2),
        "boarding_failure_risk": round(float(p_fail), 2)
    }

# =====================================================================
# NOVEL PILLAR 2: EXPLAINABLE PRESCRIPTIVE ARBITRATION (EPA) ENGINE
# =====================================================================
def run_epa_arbitration(predicted_eta_min: float, p_fail: float, walk_dist_km: float):
    """
    Calculates multi-objective disutility scores to output [BOARD, WAIT, WALK].
    """
    walking_speed_kmh = 4.5
    walk_time_min = (walk_dist_km / walking_speed_kmh) * 60.0

    # Disutility Weights
    w_wait, w_travel, w_risk = 0.4, 0.3, 0.3

    # Disutility Formulations
    u_board = (w_wait * predicted_eta_min) + (w_risk * p_fail * 15.0)
    u_wait  = (w_wait * (predicted_eta_min + 8.0)) + (w_risk * (p_fail * 0.5) * 15.0) # Assume next bus in +8 min
    u_walk  = (w_travel * walk_time_min)

    # Arbitration Decision Logic
    if p_fail > 0.70 and walk_time_min <= (predicted_eta_min + 5.0):
        decision = "WALK"
        reason = f"High boarding failure risk ({int(p_fail*100)}%). Walking takes {round(walk_time_min, 1)} mins vs waiting."
    elif p_fail > 0.85:
        decision = "WAIT"
        reason = f"Shuttle at capacity ({int(p_fail*100)}% risk). Wait for the next upcoming shuttle."
    else:
        decision = "BOARD"
        reason = f"Optimal choice. Low boarding risk with estimated arrival in {round(predicted_eta_min, 1)} mins."

    return {
        "recommendation": decision,
        "explanation": reason,
        "metrics": {
            "disutility_board": round(float(u_board), 2),
            "disutility_wait": round(float(u_wait), 2),
            "disutility_walk": round(float(u_walk), 2),
            "walk_time_min": round(float(walk_time_min), 1)
        }
    }

@app.post("/predict_epa")
def predict_epa(data: EPARequest):
    if len(data.sequence) != 10:
        raise HTTPException(status_code=400, detail="Sequence must contain exactly 10 points.")

    # 1. Infer Occupancy from latest telemetry point
    latest_pt = data.sequence[-1]
    occ_info = infer_latent_occupancy(latest_pt.dwell_time_sec, latest_pt.speed_kmph, data.bus_capacity_limit)

    # 2. Predict ETA using LSTM-GRU Model
    if model and feature_scaler and target_scaler:
        raw_seq = [[pt.latitude, pt.longitude, pt.speed_kmph, pt.distance_km, pt.segment_id] for pt in data.sequence]
        scaled_seq = feature_scaler.transform(np.array(raw_seq))
        input_3d = np.expand_dims(scaled_seq, axis=0)
        scaled_pred = model.predict(input_3d, verbose=0)
        eta_min = float(np.ravel(target_scaler.inverse_transform(scaled_pred))[0])
        eta_min = max(0.1, eta_min)
    else:
        # Mathematical fallback if local model is offline
        eta_min = (latest_pt.distance_km / max(latest_pt.speed_kmph, 10.0)) * 60.0

    # 3. Execute Explainable Prescriptive Arbitration
    epa_result = run_epa_arbitration(eta_min, occ_info["boarding_failure_risk"], data.user_walking_distance_km)

    return {
        "status": "success",
        "predicted_eta_min": round(eta_min, 2),
        "latent_occupancy": occ_info,
        "epa_prescriptive_decision": epa_result
    }
