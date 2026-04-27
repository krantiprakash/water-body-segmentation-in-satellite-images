# ── Base Image ─────────────────────────────────────────────────────────────
# Python 3.12 slim — matches local development environment
FROM python:3.12-slim

# ── Working Directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── System Dependencies ────────────────────────────────────────────────────
# OpenCV requires these Linux libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python Dependencies ────────────────────────────────────────────
# Copy requirements first — Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Project Code ──────────────────────────────────────────────────────
COPY app/        ./app/
COPY inference/  ./inference/
COPY training/   ./training/
COPY data/       ./data/
COPY configs/    ./configs/

# ── Copy Model Weights ─────────────────────────────────────────────────────
COPY outputs/results_UNet++/outputs/checkpoints/best_model.pth \
     ./outputs/results_UNet++/outputs/checkpoints/best_model.pth

# ── Create Required Directories ────────────────────────────────────────────
RUN mkdir -p outputs/temp logs

# ── Expose Port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Run Command ────────────────────────────────────────────────────────────
CMD ["python", "-m", "app.app"]