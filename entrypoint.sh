#!/bin/bash
set -e

if [ "$SERVICE_TYPE" = "api" ]; then
  echo "Starting FastAPI server..."
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000
elif [ "$SERVICE_TYPE" = "worker" ]; then
  echo "Starting Celery worker..."
  exec celery -A app.worker.celery_app worker --loglevel=info
elif [ "$SERVICE_TYPE" = "flower" ]; then
  echo "Starting Flower dashboard..."
  exec celery -A app.worker.celery_app flower --port=5555 --url_prefix=/flower
else
  echo "Unknown SERVICE_TYPE: $SERVICE_TYPE"
  exit 1
fi
