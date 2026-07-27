FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime code
COPY api/ ./api/
COPY mlflow_settings.py .

# Expose port (Render uses 8000 by default)
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.inference_service:app", "--host", "0.0.0.0", "--port", "8000"]