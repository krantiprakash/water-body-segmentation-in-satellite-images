# ── Base Image ─────────────────────────────────────────────────────────────
# Python 3.12 slim — matches local development environment
FROM python:3.12-slim

# ── Working Directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── System Dependencies ────────────────────────────────────────────────────
# OpenCV requires these Linux libraries to run inside container
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Install PyTorch CPU only ───────────────────────────────────────────────
# CPU-only build — avoids downloading 2GB+ CUDA packages
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    torch==2.11.0+cpu \
    torchvision==0.26.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Install Inference Dependencies ─────────────────────────────────────────
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    segmentation-models-pytorch==0.5.0 \
    timm==1.0.26 \
    safetensors==0.7.0 \
    albumentations==2.0.8 \
    opencv-python-headless==4.13.0.92 \
    numpy==2.4.4 \
    pillow==12.2.0 \
    matplotlib==3.10.9 \
    scipy==1.17.1 \
    fastapi==0.136.1 \
    uvicorn==0.46.0 \
    python-multipart==0.0.26 \
    PyYAML==6.0.3

# ── Cache EfficientNet-B4 Encoder Weights ──────────────────────────────────
# Downloads weights during build — no internet needed at runtime
# Must run AFTER timm is installed
RUN python -c "import timm; timm.create_model('efficientnet_b4', pretrained=True)"

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