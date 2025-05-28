from fastapi import APIRouter
from app.models.transaction import Transaction
from app.utils.predict import predict_transaction

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Fraud detection API is running."}

@router.post("/predict/")
def predict(input_data: Transaction):
    result = predict_transaction(input_data)
    return result
