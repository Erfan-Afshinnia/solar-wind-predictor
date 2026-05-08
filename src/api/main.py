from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict_single

app = FastAPI(
    title= "Solar Power Output Predictor",
    description= "Predict Ac power output (KW) from solar plant sensor readings.",
    version= "1.0.0"
)

# Request & Response schemas
class PredictionRequest(BaseModel):
    irradiation:            float = Field(..., ge=0.0, le=1.5, example=0.8)
    module_temperature:     float = Field(..., ge=0.0, le=100.0, example=45.0)
    ambient_temperature:    float = Field(..., ge=10.0, le=60.0, example=32.0)
    date_time:              str   = Field(..., example= "2020-06-01  12:00:00")

class PredictionResponse(BaseModel):
    predicted_ac_power_kw: float
    date_time:             str
    irradiation:           float
    status:                str

# Routes
@app.get("/")
def root():
    return {"message": "Solar Power Predictor API is running"}

@app.get("/health")
def health():
    return{"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        power = predict_single(
            irradiation=request.irradiation,
            module_temperature=request.module_temperature,
            ambient_temperature=request.ambient_temperature,
            date_time=request.date_time
        )
        return PredictionResponse(
            predicted_ac_power_kw=round(power, 2),
            date_time= request.date_time,
            irradiation=request.irradiation,
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    