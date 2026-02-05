# Base image
FROM python:3.9-slim

# Install build essentials only if needed by requirements
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY src /app/src
COPY yolo /app/yolo
COPY serve_config.yaml /app/serve_config.yaml

# Ray Serve default HTTP port
EXPOSE 8000
EXPOSE 8265

WORKDIR /app/src
# Start Ray Serve application
CMD ["serve", "run", "main_service:app"]

