from pydantic import BaseModel


class Transaction(BaseModel):
    step: int
    type: str  # e.g. "TRANSFER", "CASH_OUT", etc.
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

class FraudPredictionResponse(BaseModel):
    prediction: bool
    fraud_probability: float
    processing_time: float

    class Config:
        schema_extra = {
            "example": {
                "prediction": True,
                "fraud_probability": 0.8734,
                "processing_time": 0.024
            }
        }