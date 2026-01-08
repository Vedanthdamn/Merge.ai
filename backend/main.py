"""
FastAPI Inference Server for Healthcare ML Model

This server provides inference-only functionality for the trained model.
It does NOT participate in training, federated learning, or any other
research components.

Endpoints:
- POST /predict: Make predictions on patient data

IMPORTANT: Research Prototype - Not for Clinical Use
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import sys
import os
from typing import Optional

# Add parent directory to path to import baseline model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline.model import BaselineHealthcareModel

# Compute BASE_DIR for absolute path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare ML Inference API",
    description="Research Prototype - Inference endpoint for healthcare risk prediction",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
# WARNING: In production, replace allow_origins=["*"] with specific origins
# e.g., allow_origins=["http://localhost:5173", "https://yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DEVELOPMENT ONLY - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[BaselineHealthcareModel] = None


class PatientData(BaseModel):
    """Patient data input schema"""
    age: float = Field(..., description="Patient age in years", ge=0, le=120)
    sex: int = Field(..., description="Sex (0=Female, 1=Male)", ge=0, le=1)
    systolic_bp: float = Field(..., description="Systolic blood pressure (mmHg)", ge=0, le=300)
    diastolic_bp: float = Field(..., description="Diastolic blood pressure (mmHg)", ge=0, le=200)
    cholesterol: float = Field(..., description="Cholesterol level (mg/dL)", ge=0, le=500)
    fasting_glucose: float = Field(..., description="Fasting glucose level (mg/dL)", ge=0, le=500)
    bmi: float = Field(..., description="Body Mass Index", ge=10, le=80)
    heart_rate: float = Field(..., description="Heart rate (bpm)", ge=30, le=200)
    smoking: int = Field(..., description="Smoking status (0=No, 1=Yes)", ge=0, le=1)
    family_history: int = Field(..., description="Family history (0=No, 1=Yes)", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 62.5,
                "sex": 0,
                "systolic_bp": 136.2,
                "diastolic_bp": 93.3,
                "cholesterol": 268.8,
                "fasting_glucose": 70.0,
                "bmi": 29.2,
                "heart_rate": 98.4,
                "smoking": 0,
                "family_history": 1
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output schema"""
    probability: float = Field(..., description="Predicted probability (0-1)")
    risk_class: str = Field(..., description="Risk classification (Low Risk / High Risk)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    disclaimer: str = Field(
        default="Research Prototype – Not for Clinical Use. Always consult qualified healthcare professionals.",
        description="Important disclaimer"
    )


@app.on_event("startup")
async def load_model_on_startup():
    """Load the trained model and preprocessing components on server startup"""
    global model
    
    model_path = os.path.join(BASE_DIR, "models", "baseline_model.h5")
    
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. "
            "Please train the model first by running: python src/baseline/model.py"
        )
    
    try:
        model = BaselineHealthcareModel()
        model.load_model(model_path)
        print("✓ Model loaded successfully")
        print("✓ Imputer loaded successfully")
        print("✓ Scaler loaded successfully")
        print("Server ready for inference!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Healthcare ML Inference API",
        "version": "1.0.0",
        "status": "ready" if model is not None else "model not loaded",
        "endpoints": {
            "/predict": "POST - Make predictions on patient data",
            "/health": "GET - Check server health"
        },
        "disclaimer": "Research Prototype – Not for Clinical Use"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Make a prediction on patient data
    
    Args:
        patient_data: Patient features including age, sex, blood pressure, etc.
        
    Returns:
        Prediction probability and risk classification
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array in the correct order
        features = np.array([[
            patient_data.age,
            patient_data.sex,
            patient_data.systolic_bp,
            patient_data.diastolic_bp,
            patient_data.cholesterol,
            patient_data.fasting_glucose,
            patient_data.bmi,
            patient_data.heart_rate,
            patient_data.smoking,
            patient_data.family_history
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0][0]
        
        # Convert to float (in case it's a numpy type)
        probability = float(prediction)
        
        # Determine risk class based on threshold
        threshold = 0.5
        risk_class = "High Risk" if probability >= threshold else "Low Risk"
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2  # 0 = uncertain, 1 = very confident
        
        return PredictionResponse(
            probability=probability,
            risk_class=risk_class,
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
