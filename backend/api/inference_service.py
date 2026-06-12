from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mlflow
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",") #Cross-Origin Resource Sharing

models = {}  # populated at startup

#run this setup code once when the server boots, then keep the result alive for everyone to use.
@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    models["total_price"] = mlflow.pyfunc.load_model("models:/RealEstateTotalPrice/latest")
    models["per_sqm"]     = mlflow.pyfunc.load_model("models:/RealEstatePricePerSqm/latest")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

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
        log_result = models["total_price"].predict(df)[0]
        normalized_result = np.exp(log_result)

        return PredictionResponse(log_result=log_result, normalized_result=normalized_result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictPricePerSqm", response_model=PredictionResponse)
async def predict_price_per_sqm(request: PredictRequest):
    try:
        df = pd.DataFrame([request.model_dump()])
        log_result = models["per_sqm"].predict(df)[0]
        normalized_result = np.exp(log_result)

        return PredictionResponse(log_result=log_result, normalized_result=normalized_result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))