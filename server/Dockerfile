# Base image
# FROM rayproject/ray:2.6.3

# # Install build essentials only if needed by requirements
# WORKDIR /app

# # Install dependencies
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code and model files
# COPY src /app/src
# COPY yolo /app/yolo

# # Ray Serve default HTTP port
# EXPOSE 8000
# EXPOSE 8265
# EXPOSE 6379

# WORKDIR /app/src
# # Start Ray Serve application
# CMD ["serve", "run", "main_service:app"]

FROM rayproject/ray:2.38.0-cpu
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt