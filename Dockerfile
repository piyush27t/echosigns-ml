FROM python:3.10-slim

# Environment settings (recommended)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed for OpenCV & MediaPipe)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code (includes models)
COPY app app
COPY run.py run.py

# Expose Socket.IO port (frontend expects 8080)
EXPOSE 8080

# Start ML inference service with Flask-SocketIO
CMD ["python", "run.py"]
