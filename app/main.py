from fastapi import FastAPI
from dotenv import load_dotenv
from app.preprocess import FeatureEngineer
from app.endpoints import router
import joblib
import os

load_dotenv()

app = FastAPI()

@app.on_event("startup")
def load_model():
    model_name = os.environ.get('MODEL_NAME')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), f"../model/{model_name}")
    app.state.model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    
app.include_router(router)
