import io
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
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

@app.post("/predict/batch")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Accept a CSV file and return predictions for all rows.
    Required CSV columns:
    DATE_TIME, IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE
    """

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Run batch prediction
        from src.models.predict import predict_batch
        result_df = predict_batch(df)

        # Convert result to CSV for download
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type= "text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=predictions.csv"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    