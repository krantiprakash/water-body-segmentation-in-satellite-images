import os
import sys
import time
import uuid
import logging
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from contextlib import asynccontextmanager
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Change these before running ────────────────────────────────────────────
# MODEL_PATH   = r"outputs\results_UNet_retrain\outputs\checkpoints\best_model.pth"
MODEL_PATH = "outputs/results_UNet++/outputs/checkpoints/best_model.pth"
# MODEL_NAME   = "unet"   # "unet" or "unetplusplus"
MODEL_NAME   = "unetplusplus"
# ──────────────────────────────────────────────────────────────────────────

# ── Constants ──────────────────────────────────────────────────────────────
IMAGE_SIZE = 256
MEAN       = (0.485, 0.456, 0.406)
STD        = (0.229, 0.224, 0.225)
THRESHOLD  = 0.5
TEMP_DIR   = "outputs/temp"
LOG_DIR    = "logs"
LOG_FILE   = os.path.join(LOG_DIR, "app.log")

# ── Setup Directories ──────────────────────────────────────────────────────
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── Setup Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Global model variables ─────────────────────────────────────────────────
model  = None
device = None


# ── Build Model ────────────────────────────────────────────────────────────
def build_model(model_name):
    attention = "scse" if model_name == "unetplusplus" else None

    if model_name == "unet":
        m = smp.Unet(
            encoder_name    = "efficientnet-b4",
            encoder_weights = "imagenet",
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
    elif model_name == "unetplusplus":
        m = smp.UnetPlusPlus(
            encoder_name           = "efficientnet-b4",
            encoder_weights        = "imagenet",
            in_channels            = 3,
            classes                = 1,
            activation             = None,
            decoder_attention_type = attention,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'unet' or 'unetplusplus'.")
    return m


# ── Preprocess Image ───────────────────────────────────────────────────────
def preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()

    transform = Compose([
        Resize(IMAGE_SIZE, IMAGE_SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"].unsqueeze(0)  # (1, 3, 256, 256)
    return tensor, original


# ── Run Inference ──────────────────────────────────────────────────────────
def run_inference(tensor):
    global model, device
    tensor = tensor.to(device)

    with torch.no_grad():
        t_start = time.time()
        logits  = model(tensor)
        elapsed = time.time() - t_start

    probs = torch.sigmoid(logits)
    mask  = (probs > THRESHOLD).float()
    mask  = mask[0, 0].cpu().numpy()   # (256, 256)

    return mask, elapsed


# ── Save Outputs ───────────────────────────────────────────────────────────
def save_outputs(original, mask, base_name, input_ext):
    # Resize original to 256×256 for consistent output
    original_resized = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

    # ── Binary mask ──
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_path  = os.path.join(TEMP_DIR, f"{base_name}_mask{input_ext}")
    cv2.imwrite(mask_path, mask_uint8)

    # ── Overlay — water highlighted in blue ──
    overlay      = original_resized.copy().astype(np.float32)
    water_pixels = mask.astype(bool)
    overlay[water_pixels, 0] = overlay[water_pixels, 0] * 0.4
    overlay[water_pixels, 1] = overlay[water_pixels, 1] * 0.4
    overlay[water_pixels, 2] = np.clip(
        overlay[water_pixels, 2] * 0.4 + 180, 0, 255
    )
    overlay      = overlay.astype(np.uint8)
    overlay_path = os.path.join(TEMP_DIR, f"{base_name}_overlay{input_ext}")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return mask_path, overlay_path


# ── Lifespan — Load model once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device       : {device}")
    logger.info(f"Model        : {MODEL_NAME}")
    logger.info(f"Loading from : {MODEL_PATH}")

    model = build_model(MODEL_NAME)
    model.load_state_dict(torch.load(
        MODEL_PATH,
        map_location = device,
        weights_only = True
    ))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully. Server ready.")
    logger.info("--------------------------------------------------")
    logger.info("Web UI   : http://localhost:8000/")
    logger.info("API docs : http://localhost:8000/docs")
    logger.info("Health   : http://localhost:8000/health")
    logger.info("--------------------------------------------------")
    yield
    logger.info("Server shutting down.")


# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Water Body Segmentation API",
    description = "Segment water bodies from satellite images using U-Net++ with EfficientNet-B4",
    version     = "1.0.0",
    lifespan    = lifespan
)


# ── GET / — Home Page ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Water Body Segmentation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        h1   { color: #2c3e50; }
        .upload-box { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        input[type=file] { margin: 10px 0; }
        button { background: #2980b9; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; }
        button:hover { background: #1a5276; }
        #results { margin-top: 30px; display: none; }
        .images { display: flex; gap: 20px; flex-wrap: nowrap; overflow-x: auto; margin-top: 15px; }
        .img-box { text-align: center; min-width: 256px; }
        .img-box img { width: 256px; height: 256px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd; }
        .img-box p { margin: 6px 0; font-size: 13px; color: #555; font-weight: bold; }
        .stats { background: #eaf4fb; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .download-btn { background: #27ae60; margin-top: 8px; display: inline-block; }
        #loading { display: none; color: #888; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🌊 Water Body Segmentation</h1>
    <div class="upload-box">
        <p>Upload a satellite image to detect water bodies.</p>
        <input type="file" id="imageInput" accept=".jpg,.jpeg,.png"><br>
        <button onclick="predict()">Run Inference</button>
        <p id="loading">⏳ Running inference, please wait...</p>
    </div>

    <div id="results">
        <h2>Results</h2>
        <div class="stats" id="stats"></div>
        <div class="images">
            <div class="img-box">
                <img id="originalImg" src="" alt="Original Image">
                <p>Original Image</p>
            </div>
            <div class="img-box">
                <img id="maskImg" src="" alt="Predicted Mask">
                <p>Predicted Mask</p>
                <a id="maskDownload" href="#" download>
                    <button class="download-btn">⬇ Download Mask</button>
                </a>
            </div>
            <div class="img-box">
                <img id="overlayImg" src="" alt="Water Overlay">
                <p>Water Overlay (Blue)</p>
                <a id="overlayDownload" href="#" download>
                    <button class="download-btn">⬇ Download Overlay</button>
                </a>
            </div>
        </div>
    </div>

    <script>
        async function predict() {
            const input = document.getElementById('imageInput');
            if (!input.files[0]) { alert('Please select an image first.'); return; }

            // Show original image immediately from local file
            const reader = new FileReader();
            reader.onload = e => { document.getElementById('originalImg').src = e.target.result; };
            reader.readAsDataURL(input.files[0]);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            formData.append('file', input.files[0]);

            const response = await fetch('/predict', { method: 'POST', body: formData });

            if (!response.ok) {
                const err = await response.json();
                alert('Error: ' + err.detail);
                document.getElementById('loading').style.display = 'none';
                return;
            }

            // Get headers
            const waterPct  = response.headers.get('x-water-percentage');
            const inferTime = response.headers.get('x-inference-time-ms');
            const overlayUrl = response.headers.get('x-overlay-available');

            // Mask image from response body
            const blob    = await response.blob();
            const maskUrl = URL.createObjectURL(blob);

            // Show stats
            document.getElementById('stats').innerHTML =
                `<b>Water detected:</b> ${waterPct}% &nbsp;|&nbsp; <b>Inference time:</b> ${inferTime} ms`;

            // Show mask
            document.getElementById('maskImg').src = maskUrl;
            document.getElementById('maskDownload').href = maskUrl;

            // Show overlay
            document.getElementById('overlayImg').src = overlayUrl;
            document.getElementById('overlayDownload').href = overlayUrl;

            document.getElementById('loading').style.display = 'none';
            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>
""")


# ── GET /health — JSON Health Check for monitoring ─────────────────────────
@app.get("/health")
def health_check():
    return JSONResponse({
        "status"    : "running",
        "model"     : MODEL_NAME,
        "encoder"   : "efficientnet-b4",
        "image_size": IMAGE_SIZE,
        "threshold" : THRESHOLD,
        "device"    : str(device),
    })


# ── POST /predict — Run Inference ──────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ── Validate file type ──
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid file type '{file_ext}'. Allowed: jpg, jpeg, png"
        )

    # ── Unique base name to avoid conflicts ──
    base_name  = uuid.uuid4().hex
    temp_input = os.path.join(TEMP_DIR, f"{base_name}_input{file_ext}")

    try:
        # ── Save uploaded file temporarily ──
        contents = await file.read()
        with open(temp_input, "wb") as f:
            f.write(contents)

        # ── Preprocess ──
        tensor, original = preprocess(temp_input)
        orig_h, orig_w   = original.shape[:2]

        # ── Inference ──
        mask, elapsed = run_inference(tensor)
        water_pct     = float(mask.mean() * 100)

        # ── Save outputs ──
        mask_path, overlay_path = save_outputs(
            original, mask, base_name, file_ext
        )

        # ── Log request ──
        logger.info(
            f"file={file.filename} | "
            f"size={orig_w}x{orig_h} | "
            f"time={elapsed*1000:.1f}ms | "
            f"water={water_pct:.2f}%"
        )

        # ── Return mask as download ──
        # Overlay available via /overlay/{base_name}_overlay{ext}
        return FileResponse(
            path       = mask_path,
            media_type = f"image/{file_ext.strip('.')}",
            filename   = f"{os.path.splitext(file.filename)[0]}_mask{file_ext}",
            headers    = {
                "X-Water-Percentage"  : f"{water_pct:.2f}",
                "X-Inference-Time-Ms" : f"{elapsed*1000:.1f}",
                "X-Overlay-Available" : f"/overlay/{base_name}_overlay{file_ext}",
            }
        )

    except ValueError as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        # ── Always clean up input temp file ──
        if os.path.exists(temp_input):
            os.remove(temp_input)


# ── GET /overlay/{filename} — Retrieve Overlay Image ──────────────────────
@app.get("/overlay/{filename}")
def get_overlay(filename: str):
    # Security: prevent path traversal
    filename     = os.path.basename(filename)
    overlay_path = os.path.join(TEMP_DIR, filename)

    if not os.path.exists(overlay_path):
        raise HTTPException(
            status_code = 404,
            detail      = "Overlay not found. It may have expired or been cleaned up."
        )

    file_ext   = os.path.splitext(filename)[1].lower()
    media_type = f"image/{file_ext.strip('.')}"

    return FileResponse(
        path       = overlay_path,
        media_type = media_type,
        filename   = filename
    )


# ── Run Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Server starting...")
    print("🌊 Web UI     : http://localhost:8000/")
    print("📖 API docs   : http://localhost:8000/docs")
    print("❤️  Health     : http://localhost:8000/health")
    print("Press CTRL+C to stop\n")
    uvicorn.run(
        "app.app:app",
        host    = "0.0.0.0",  # bind to all interfaces inside container
        port    = 8000,
        reload  = False,
        workers = 1
    )


# import os
# import sys
# import time
# import uuid
# import logging
# import numpy as np
# import cv2
# import torch
# import segmentation_models_pytorch as smp
# from contextlib import asynccontextmanager
# from albumentations import Compose, Resize, Normalize
# from albumentations.pytorch import ToTensorV2
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse, JSONResponse
# import uvicorn

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # ── Change these before running ────────────────────────────────────────────
# # MODEL_PATH   = r"outputs\results_UNet_retrain\outputs\checkpoints\best_model.pth"
# MODEL_PATH   = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\outputs\results_UNet++\outputs\checkpoints\best_model.pth"
# # MODEL_NAME   = "unet"   # "unet" or "unetplusplus"
# MODEL_NAME   = "unetplusplus"
# # ──────────────────────────────────────────────────────────────────────────

# # ── Constants ──────────────────────────────────────────────────────────────
# IMAGE_SIZE = 256
# MEAN       = (0.485, 0.456, 0.406)
# STD        = (0.229, 0.224, 0.225)
# THRESHOLD  = 0.5
# TEMP_DIR   = "outputs/temp"
# LOG_DIR    = "logs"
# LOG_FILE   = os.path.join(LOG_DIR, "app.log")

# # ── Setup Directories ──────────────────────────────────────────────────────
# os.makedirs(TEMP_DIR, exist_ok=True)
# os.makedirs(LOG_DIR,  exist_ok=True)

# # ── Setup Logging ──────────────────────────────────────────────────────────
# logging.basicConfig(
#     level    = logging.INFO,
#     format   = "%(asctime)s | %(levelname)s | %(message)s",
#     handlers = [
#         logging.FileHandler(LOG_FILE),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # ── Global model variables ─────────────────────────────────────────────────
# model  = None
# device = None


# # ── Build Model ────────────────────────────────────────────────────────────
# def build_model(model_name):
#     attention = "scse" if model_name == "unetplusplus" else None

#     if model_name == "unet":
#         m = smp.Unet(
#             encoder_name    = "efficientnet-b4",
#             encoder_weights = "imagenet",
#             in_channels     = 3,
#             classes         = 1,
#             activation      = None,
#         )
#     elif model_name == "unetplusplus":
#         m = smp.UnetPlusPlus(
#             encoder_name           = "efficientnet-b4",
#             encoder_weights        = "imagenet",
#             in_channels            = 3,
#             classes                = 1,
#             activation             = None,
#             decoder_attention_type = attention,
#         )
#     else:
#         raise ValueError(f"Unknown model: {model_name}. Use 'unet' or 'unetplusplus'.")
#     return m


# # ── Preprocess Image ───────────────────────────────────────────────────────
# def preprocess(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image: {image_path}")
#     image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     original = image.copy()

#     transform = Compose([
#         Resize(IMAGE_SIZE, IMAGE_SIZE),
#         Normalize(mean=MEAN, std=STD),
#         ToTensorV2(),
#     ])
#     tensor = transform(image=image)["image"].unsqueeze(0)  # (1, 3, 256, 256)
#     return tensor, original


# # ── Run Inference ──────────────────────────────────────────────────────────
# def run_inference(tensor):
#     global model, device
#     tensor = tensor.to(device)

#     with torch.no_grad():
#         t_start = time.time()
#         logits  = model(tensor)
#         elapsed = time.time() - t_start

#     probs = torch.sigmoid(logits)
#     mask  = (probs > THRESHOLD).float()
#     mask  = mask[0, 0].cpu().numpy()   # (256, 256)

#     return mask, elapsed


# # ── Save Outputs ───────────────────────────────────────────────────────────
# def save_outputs(original, mask, base_name, input_ext):
#     # Resize original to 256×256 for consistent output
#     original_resized = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

#     # ── Binary mask ──
#     mask_uint8 = (mask * 255).astype(np.uint8)
#     mask_path  = os.path.join(TEMP_DIR, f"{base_name}_mask{input_ext}")
#     cv2.imwrite(mask_path, mask_uint8)

#     # ── Overlay — water highlighted in blue ──
#     overlay      = original_resized.copy().astype(np.float32)
#     water_pixels = mask.astype(bool)
#     overlay[water_pixels, 0] = overlay[water_pixels, 0] * 0.4
#     overlay[water_pixels, 1] = overlay[water_pixels, 1] * 0.4
#     overlay[water_pixels, 2] = np.clip(
#         overlay[water_pixels, 2] * 0.4 + 180, 0, 255
#     )
#     overlay      = overlay.astype(np.uint8)
#     overlay_path = os.path.join(TEMP_DIR, f"{base_name}_overlay{input_ext}")
#     cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

#     return mask_path, overlay_path


# # ── Lifespan — Load model once at startup ─────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global model, device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Device       : {device}")
#     logger.info(f"Model        : {MODEL_NAME}")
#     logger.info(f"Loading from : {MODEL_PATH}")

#     model = build_model(MODEL_NAME)
#     model.load_state_dict(torch.load(
#         MODEL_PATH,
#         map_location = device,
#         weights_only = True
#     ))
#     model = model.to(device)
#     model.eval()
#     logger.info("Model loaded successfully. Server ready.")
#     yield
#     logger.info("Server shutting down.")


# # ── FastAPI App ────────────────────────────────────────────────────────────
# app = FastAPI(
#     title       = "Water Body Segmentation API",
#     description = "Segment water bodies from satellite images using U-Net++ with EfficientNet-B4",
#     version     = "1.0.0",
#     lifespan    = lifespan
# )


# # ── GET / — Health Check ───────────────────────────────────────────────────
# @app.get("/")
# def health_check():
#     return JSONResponse({
#         "status"    : "running",
#         "model"     : MODEL_NAME,
#         "encoder"   : "efficientnet-b4",
#         "image_size": IMAGE_SIZE,
#         "threshold" : THRESHOLD,
#         "device"    : str(device),
#     })


# # ── POST /predict — Run Inference ──────────────────────────────────────────
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):

#     # ── Validate file type ──
#     allowed_extensions = {".jpg", ".jpeg", ".png"}
#     file_ext = os.path.splitext(file.filename)[1].lower()
#     if file_ext not in allowed_extensions:
#         logger.warning(f"Invalid file type: {file.filename}")
#         raise HTTPException(
#             status_code = 400,
#             detail      = f"Invalid file type '{file_ext}'. Allowed: jpg, jpeg, png"
#         )

#     # ── Unique base name to avoid conflicts ──
#     base_name  = uuid.uuid4().hex
#     temp_input = os.path.join(TEMP_DIR, f"{base_name}_input{file_ext}")

#     try:
#         # ── Save uploaded file temporarily ──
#         contents = await file.read()
#         with open(temp_input, "wb") as f:
#             f.write(contents)

#         # ── Preprocess ──
#         tensor, original = preprocess(temp_input)
#         orig_h, orig_w   = original.shape[:2]

#         # ── Inference ──
#         mask, elapsed = run_inference(tensor)
#         water_pct     = float(mask.mean() * 100)

#         # ── Save outputs ──
#         mask_path, overlay_path = save_outputs(
#             original, mask, base_name, file_ext
#         )

#         # ── Log request ──
#         logger.info(
#             f"file={file.filename} | "
#             f"size={orig_w}x{orig_h} | "
#             f"time={elapsed*1000:.1f}ms | "
#             f"water={water_pct:.2f}%"
#         )

#         # ── Return mask as download ──
#         # Overlay available via /overlay/{base_name}_overlay{ext}
#         return FileResponse(
#             path       = mask_path,
#             media_type = f"image/{file_ext.strip('.')}",
#             filename   = f"{os.path.splitext(file.filename)[0]}_mask{file_ext}",
#             headers    = {
#                 "X-Water-Percentage"  : f"{water_pct:.2f}",
#                 "X-Inference-Time-Ms" : f"{elapsed*1000:.1f}",
#                 "X-Overlay-Available" : f"/overlay/{base_name}_overlay{file_ext}",
#             }
#         )

#     except ValueError as e:
#         logger.error(f"Preprocessing error: {str(e)}")
#         raise HTTPException(status_code=422, detail=str(e))

#     except Exception as e:
#         logger.error(f"Inference error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

#     finally:
#         # ── Always clean up input temp file ──
#         if os.path.exists(temp_input):
#             os.remove(temp_input)


# # ── GET /overlay/{filename} — Retrieve Overlay Image ──────────────────────
# @app.get("/overlay/{filename}")
# def get_overlay(filename: str):
#     # Security: prevent path traversal
#     filename     = os.path.basename(filename)
#     overlay_path = os.path.join(TEMP_DIR, filename)

#     if not os.path.exists(overlay_path):
#         raise HTTPException(
#             status_code = 404,
#             detail      = "Overlay not found. It may have expired or been cleaned up."
#         )

#     file_ext   = os.path.splitext(filename)[1].lower()
#     media_type = f"image/{file_ext.strip('.')}"

#     return FileResponse(
#         path       = overlay_path,
#         media_type = media_type,
#         filename   = filename
#     )


# # ── Run Server ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("\nServer starting...")
#     print("API docs : http://localhost:8000/docs")
#     print("Health   : http://localhost:8000/")
#     print("Press CTRL+C to stop\n")
#     uvicorn.run(
#         "app.app:app",
#         host    = "127.0.0.1",
#         port    = 8000,
#         reload  = False,
#         workers = 1
#     )