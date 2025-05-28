import pandas as pd
from app.models.transaction import model, Transaction

def predict_transaction(input_data: Transaction):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return {
        "prediction": int(prediction),
        "fraud_probability": round(prob, 4)
    }
