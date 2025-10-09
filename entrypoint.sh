#!/bin/bash
set -e

echo "🔧 Starting service: ${SERVICE_TYPE:-api}"

case "$SERVICE_TYPE" in
  "worker")
    echo "Starting Celery worker..."
    exec celery -A app.worker.celery_app worker --loglevel=info
    ;;
  "flower")
    echo "Starting Flower dashboard..."
    exec celery -A app.worker.celery_app flower --port=5555 --url_prefix=/flower
    ;;
  *)
    echo "Starting FastAPI app..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
    ;;
esac
