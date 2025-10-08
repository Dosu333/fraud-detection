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
