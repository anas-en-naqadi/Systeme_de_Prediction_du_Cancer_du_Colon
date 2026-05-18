FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/training

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY . /app/training

# Default: run training script which will emit model artifacts into /app/model
VOLUME ["/app/model"]

ENTRYPOINT ["python", "train.py"]
