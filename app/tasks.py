# app/tasks/retrain_model.py
from app.worker import celery_app
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

@celery_app.task
def retrain_model(new_data_path: str):
    """
    Retrains the full fraud detection pipeline (feature engineering + model)
    using new labeled transaction data.
    """
    try:
        df = pd.read_csv(new_data_path)
        required_cols = {
            "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest", "isFraud"
        }
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        model_name = os.environ.get('MODEL_NAME')
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_DIR = os.path.join(BASE_DIR, "model")
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        pipeline = joblib.load(MODEL_PATH)

        # Load new transaction data
        df = pd.read_csv(new_data_path)

        # Separate features and label
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]

        # Split for training/testing
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Retrain the pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate for metrics
        score = pipeline.score(X_val, y_val)

        # Save updated pipeline with timestamped version
        new_model_name = f"fraud_model_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        new_model_path = os.path.join(MODEL_DIR, new_model_name)

        joblib.dump(pipeline, new_model_path)

        return {
            "status": "Model retrained successfully",
            "data_size": len(df),
            "validation_score": score,
            "model_path": new_model_path,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
