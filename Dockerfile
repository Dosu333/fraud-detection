FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/logs

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
