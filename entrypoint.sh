#!/bin/bash
set -e

echo "Starting Celery worker in the background..."
celery -A app.worker.celery_app worker --loglevel=info &

echo "Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
