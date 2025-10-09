from fastapi import APIRouter, Request, HTTPException, UploadFile, File, status
from celery.result import AsyncResult
from app.worker import celery_app
from app.tasks import retrain_model
from app.models import (
    Transaction,
    FraudPredictionResponse, 
    RetrainRequest,
    TriggerRetrainResponse,
    RetrainStatusResponse
)
from app.logger import logger
import pandas as pd
import time
import os
import shutil


router = APIRouter()

@router.get("/")
def home():
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

@router.post(
    "/retrain/",
    summary="Trigger model retraining",
    description="This endpoint triggers background retraining of the fraud detection model using new data.",
    response_model=TriggerRetrainResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def retrain_model(file: UploadFile = File(...)):
    """Upload CSV and start background model retraining."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Create an upload path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_dir, "uploads")  
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Start Celery task
    task = celery_app.send_task("app.tasks.retrain_model", args=[file_path])
    return TriggerRetrainResponse(status_code=202, message="Retraining started", task_id=task.id)

@router.get(
    "/retrain/status/{task_id}",
    summary="Check retraining status",
    description="Check the status of a retraining task using its task ID.",
    response_model=RetrainStatusResponse
)
def get_retraining_status(task_id: str):
    """
    Check the retraining task status.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state == "PENDING":
        return RetrainStatusResponse(task_id=task_id, status="Pending")
    elif task_result.state == "STARTED":
        return RetrainStatusResponse(task_id=task_id, status="Running")
    elif task_result.state == "SUCCESS":
        return RetrainStatusResponse(task_id=task_id, status="Completed", result=task_result.result)
    elif task_result.state == "FAILURE":
        return RetrainStatusResponse(task_id=task_id, status="Failed", error=str(task_result.result))
    else:
        return RetrainStatusResponse(task_id=task_id, status=task_result.state)