FROM python:3.11-slim

# OpenCV-headless still needs a couple of system libs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so they get cached when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source files. Anything not listed here (srcImg/, output/, .git/) stays out
# via .dockerignore.
COPY vectorize.py api.py ./

ENV PYTHONUNBUFFERED=1 \
    PORT=10000

EXPOSE 10000

# Render injects $PORT — honour it, but default to 10000 for local runs.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
