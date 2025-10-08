from fastapi import FastAPI
from app.preprocess import FeatureEngineer
from app.endpoints import router
import joblib
import os

app = FastAPI()

@app.on_event("startup")
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/fraud_model_pipeline_v1.pkl")
    app.state.model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    
app.include_router(router)
