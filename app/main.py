from fastapi import FastAPI
from dotenv import load_dotenv
from app.endpoints import router
from app.logger import logger
import joblib
import time
import os

# Load .env
load_dotenv()


app = FastAPI(
    title="Online Fraud Detection API",
    description="""This API detects fraudulent transactions
                    using a machine learning model.""",
    version="1.0.0",
    contact={
        "name": "Oladosu Larinde",
        "email": "larindeakin@gmail.com",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)


@app.on_event("startup")
def load_model():
    start_time = time.time()
    model_name = os.environ.get("MODEL_NAME")
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), f"../model/{model_name}")
    app.state.model = joblib.load(MODEL_PATH)
    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ Model loaded successfully in {elapsed}s")


app.include_router(router)
