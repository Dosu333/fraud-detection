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
        
class RetrainRequest(BaseModel):
    new_data_path: str

    class Config:
        schema_extra = {
            "example": {
                "new_data_path": "data/new_transactions.csv"
            }
        }
        
class TriggerRetrainResponse(BaseModel):
    status_code: int
    task_id: str
    message: str

    class Config:
        schema_extra = {
            "example": {
                "status_code": 202,
                "task_id": "abc123xyz",
                "message": "Retraining started"
            }
        }
        
class RetrainStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
    error: str | None = None

    class Config:
        schema_extra = {
            "example": {
                "task_id": "abc123xyz",
                "status": "SUCCESS",
                "result": {
                    "status": "Model retrained successfully",
                    "data_size": 10000,
                    "validation_score": 0.95,
                    "model_path": "model/fraud_model_pipeline_20231010_101010.pkl"
                }
            }
        }