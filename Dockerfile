# Use slim Python base — keeps image size small
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching —
# dependencies only reinstall if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and saved model artifacts
COPY app/ ./app/
COPY model/ ./model/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
