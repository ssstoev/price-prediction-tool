from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from typing import Optional
import traceback
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model_total_price = mlflow.pyfunc.load_model("models:/RealEstateTotalPrice/latest")
model_per_sqm = mlflow.pyfunc.load_model("models:/RealEstatePricePerSqm/latest")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    size_m2: float 
    nr_of_rooms: int 
    total_floors: int 
    appartment_floor: int 
    neighbourhood: str 
    is_first_floor: int 
    is_last_floor: int 
    includes_parking: int
    near_public_transport: int 
    furnished: int 
    new_building: int 
    akt16: int

class PredictionResponse(BaseModel):
    log_result: float
    normalized_result: float

@app.post("/predictTotalPrice", response_model=PredictionResponse)
async def predict_total_price(request: PredictRequest):
    try:
        df = pd.DataFrame([request.model_dump()])
        log_result = model_total_price.predict(df)[0]
        normalized_result = np.exp(log_result)

        return PredictionResponse(log_result=log_result, normalized_result=normalized_result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictPricePerSqm", response_model=PredictionResponse)
async def predict_price_per_sqm(request: PredictRequest):
    try:
        df = pd.DataFrame([request.model_dump()])
        log_result = model_per_sqm.predict(df)[0]
        normalized_result = np.exp(log_result)

        return PredictionResponse(log_result=log_result, normalized_result=normalized_result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))