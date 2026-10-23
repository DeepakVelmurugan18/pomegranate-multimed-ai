# Pomegranate-MultiMed AI Production Container
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

WORKDIR /workspace

# Install system dependencies for OpenCV and imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (fast, lightweight for inference) & dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source and trained baseline weights
COPY app/ app/
COPY outputs/baseline/ outputs/baseline/
COPY xai/ xai/

EXPOSE 8080

CMD ["python", "app/server.py"]
