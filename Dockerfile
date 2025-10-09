FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make sure logs directory exists
RUN mkdir -p /app/logs

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose FastAPI default port
EXPOSE 8000

# Use entrypoint
ENTRYPOINT ["/entrypoint.sh"]
