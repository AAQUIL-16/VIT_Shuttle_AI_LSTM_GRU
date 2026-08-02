import os
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="VIT Smart Shuttle ETA API",
    description="LSTM-GRU Deep Learning Inference Engine for Shuttle Arrival Time Prediction"
)

# Enable CORS for Flutter mobile & web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_gru_eta_model.keras")
FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.pkl")

# Global variables for loaded artifacts
model = None
feature_scaler = None
target_scaler = None

@app.on_event("startup")
def load_artifacts():
    """Load model and scalers when the API starts up."""
    global model, feature_scaler, target_scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ LSTM-GRU Model loaded successfully.")
        else:
            print(f"⚠️ Model file missing at: {MODEL_PATH}")

        if os.path.exists(FEATURE_SCALER_PATH):
            with open(FEATURE_SCALER_PATH, "rb") as f:
                feature_scaler = pickle.load(f)
            print("✅ Feature scaler loaded.")

        if os.path.exists(TARGET_SCALER_PATH):
            with open(TARGET_SCALER_PATH, "rb") as f:
                target_scaler = pickle.load(f)
            print("✅ Target scaler loaded.")

    except Exception as e:
        print(f"❌ Critical Error loading artifacts: {str(e)}")

# Request schema definitions
class LocationPoint(BaseModel):
    latitude: float
    longitude: float
    speed_kmph: float
    distance_km: float

class SequenceRequest(BaseModel):
    sequence: List[LocationPoint] = Field(
        ..., 
        min_items=10, 
        max_items=10, 
        description="Array of 10 historical location snapshots"
    )

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "VIT Smart Shuttle LSTM-GRU Inference Server",
        "model_loaded": model is not None
    }

@app.post("/predict_eta")
def predict_eta(data: SequenceRequest):
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model artifacts are not fully loaded on the server."
        )

    try:
        # Convert Pydantic points to 2D numpy array: (10, 4)
        raw_sequence = [
            [pt.latitude, pt.longitude, pt.speed_kmph, pt.distance_km] 
            for pt in data.sequence
        ]
        sequence_np = np.array(raw_sequence)

        # Scale features using pre-fitted MinMax Scaler
        scaled_seq = feature_scaler.transform(sequence_np)

        # Reshape to 3D tensor expected by Keras: (batch_size=1, sequence_length=10, features=4)
        input_tensor = np.expand_dims(scaled_seq, axis=0)

        # Run Deep Learning Inference
        scaled_pred = model.predict(input_tensor, verbose=0)

        # Inverse transform to get ETA in real-world minutes
        actual_eta = target_scaler.inverse_transform(scaled_pred)[0][0]

        # Prevent negative predictions and round to 2 decimal places
        final_eta = round(max(0.1, float(actual_eta)), 2)

        return {
            "status": "success",
            "predicted_travel_time_min": final_eta
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")
