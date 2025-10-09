import os
import logging
from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger
from pythonjsonlogger import jsonlogger

# Celery configuration
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "fraud_detection", 
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks"]
)

# Log directory
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/celery.log"


def setup_json_logger(logger):
    """Attach JSON formatter to logger handlers."""
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@after_setup_logger.connect
def setup_celery_logger(logger, *args, **kwargs):
    """Setup global Celery logger."""
    setup_json_logger(logger)


@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """Setup per-task Celery logger."""
    setup_json_logger(logger)
