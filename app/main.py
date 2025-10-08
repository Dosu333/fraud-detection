from fastapi import FastAPI
from dotenv import load_dotenv
from app.preprocess import FeatureEngineer
from app.endpoints import router
from app.logger import logger
import joblib
import time
import os

# Load .env
load_dotenv()

app = FastAPI()

@app.on_event("startup")
def load_model():
    start_time = time.time()
    model_name = os.environ.get('MODEL_NAME')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), f"../model/{model_name}")
    app.state.model = joblib.load(MODEL_PATH)
    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ Model loaded successfully in {elapsed}s")
    
app.include_router(router)
