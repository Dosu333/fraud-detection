from fastapi import APIRouter, Request
from app.models import Transaction, FraudPredictionResponse
from app.logger import logger
import pandas as pd
import time


router = APIRouter()

@router.get("/")
def root():
    return {"message": "Fraud detection API is running."}

@router.post(
    "/predict/", 
    summary="Predict fraud", 
    description="This endpoint takes transaction data and predicts whether it is fraudulent.",
    response_model=FraudPredictionResponse
)
def predict(request: Request, input_data: Transaction):
    """
    ### Predict Fraud
    - **transaction**: JSON object containing transaction details
    - **Returns**: {
        prediction: bool
        fraud_probability: float
    }
    """
    start_time = time.time()
    model = request.app.state.model
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    elapsed = round(time.time() - start_time, 3)

    logger.info(
        f"Prediction completed in {elapsed}s | Fraud Probability: {prob:.4f} | Prediction: {bool(prediction)}"
    )

    return FraudPredictionResponse(
        prediction=bool(prediction),
        fraud_probability=float(round(prob, 4)),
        processing_time=elapsed,
    )
