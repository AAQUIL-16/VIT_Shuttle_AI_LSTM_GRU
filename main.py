import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from keras.models import load_model

app = FastAPI(
    title="Adaptive Route-Segment Synchronized Shuttle ETA Prediction API",
    description="LSTM-GRU Deep Learning Inference Engine for Dynamic Campus Shuttle ETA",
    version="2.0"
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
    print(f"⚠️ Error loading model artifacts: {e}")
    model, feature_scaler, target_scaler = None, None, None

# Define Input Schemas matching Patent Specification
class ShuttleTelemetryPoint(BaseModel):
    latitude: float = Field(..., example=12.9692)
    longitude: float = Field(..., example=79.1559)
    speed_kmph: float = Field(..., example=18.5)
    distance_km: float = Field(..., example=1.2)
    segment_id: float = Field(default=1.0, example=1.0) # Route segment identification

class SequenceInput(BaseModel):
    sequence: List[ShuttleTelemetryPoint]

@app.get("/")
def health_check():
    return {
        "status": "Online",
        "system": "Adaptive Route-Segment Synchronized Dynamic Shuttle Dispatch",
        "model_loaded": model is not None
    }

@app.post("/predict_eta")
def predict_eta(data: SequenceInput):
    if not model or not feature_scaler or not target_scaler:
        raise HTTPException(status_code=500, detail="Model assets are not initialized on server.")
    
    # 1. Validate sequence length required for recurrent architecture
    if len(data.sequence) != 10:
        raise HTTPException(
            status_code=400, 
            detail=f"Recurrent input sequence must contain exactly 10 time steps, got {len(data.sequence)}."
        )

    try:
        # 2. Convert incoming JSON telemetry stream to 2D numpy array [10, 5]
        raw_seq = [
            [pt.latitude, pt.longitude, pt.speed_kmph, pt.distance_km, pt.segment_id] 
            for pt in data.sequence
        ]
        raw_seq_np = np.array(raw_seq)

        # 3. Scale input sequence features using saved fit transformation
        scaled_seq = feature_scaler.transform(raw_seq_np)

        # 4. Reshape for LSTM-GRU network expecting shape [batch_size=1, timesteps=10, features=5]
        input_3d = np.expand_dims(scaled_seq, axis=0)

        # 5. Run inference
        scaled_prediction = model.predict(input_3d)

        # 6. Inverse transform model output back to real-world minutes
        unscaled_eta = target_scaler.inverse_transform(scaled_prediction)
        predicted_minutes = float(np.ravel(unscaled_eta)[0])

        return {
            "status": "success",
            "predicted_travel_time_min": round(max(0.1, predicted_minutes), 2),
            "unit": "minutes",
            "sync_status": "Route-Segment Synchronized"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
