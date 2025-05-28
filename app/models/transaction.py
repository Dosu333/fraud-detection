from pydantic import BaseModel
import joblib
import os

class Transaction(BaseModel):
    step: int
    type_TRANSFER: int
    type_CASH_OUT: int
    type_PAYMENT: int
    type_DEBIT: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    balanceDiffOrig: float
    balanceDiffDest: float
    orig_balance_ratio: float
    dest_balance_ratio: float


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../model/fraud_model.pkl")
model = joblib.load(MODEL_PATH)
