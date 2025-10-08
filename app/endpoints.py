from fastapi import APIRouter, Request
from app.models import Transaction
import pandas as pd

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Fraud detection API is running."}

@router.post("/predict/")
def predict(request: Request, input_data: Transaction):
    model = request.app.state.model
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return {
        "prediction": bool(prediction),
        "fraud_probability": f"{(float(prob) * 100):.2f}"
    }
