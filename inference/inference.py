import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Change these before running ────────────────────────────────────────────
# MODEL_PATH   = r"outputs\results_UNet_retrain\outputs\checkpoints\best_model.pth"
MODEL_PATH   = r"outputs\results_UNet++\outputs\checkpoints\best_model.pth"
# MODEL_NAME   = "unet"   # "unet" or "unetplusplus"
MODEL_NAME   = "unetplusplus"
INPUT_IMAGE  = r"Water Bodies Dataset\Images\water_body_896.jpg"
OUTPUT_DIR   = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\outputs\predictions"
# ──────────────────────────────────────────────────────────────────────────

# ── Constants ──────────────────────────────────────────────────────────────
IMAGE_SIZE = 256
MEAN       = (0.485, 0.456, 0.406)
STD        = (0.229, 0.224, 0.225)
THRESHOLD  = 0.5


# ── Build Model ────────────────────────────────────────────────────────────
def build_model(model_name):
    attention = "scse" if model_name == "unetplusplus" else None

    if model_name == "unet":
        model = smp.Unet(
            encoder_name    = "efficientnet-b4",
            encoder_weights = "imagenet",
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name           = "efficientnet-b4",
            encoder_weights        = "imagenet",
            in_channels            = 3,
            classes                = 1,
            activation             = None,
            decoder_attention_type = attention,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'unet' or 'unetplusplus'.")
    return model


# ── Preprocess Image ───────────────────────────────────────────────────────
def preprocess(image_path):
    # Load image BGR → RGB
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original for visualization (before resize/normalize)
    original = image.copy()

    # Apply same transforms as training val/test pipeline
    transform = Compose([
        Resize(IMAGE_SIZE, IMAGE_SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0)  # (1, 3, 256, 256)

    return tensor, original


# ── Run Inference ──────────────────────────────────────────────────────────
def predict(model, tensor, device):
    model.eval()
    tensor = tensor.to(device)

    with torch.no_grad():
        t_start = time.time()
        logits  = model(tensor)                          # (1, 1, 256, 256)
        elapsed = time.time() - t_start

    probs = torch.sigmoid(logits)                        # (1, 1, 256, 256)
    mask  = (probs > THRESHOLD).float()
    mask  = mask[0, 0].cpu().numpy()                     # (256, 256) — explicit indexing

    return mask, elapsed


# ── Save Outputs ───────────────────────────────────────────────────────────
def save_outputs(original, mask, output_dir, image_path):
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ── Resize original to 256×256 for consistent visualization ──
    original_resized = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

    # ── 1. Save original image ──
    orig_path = os.path.join(output_dir, f"{base_name}_original.png")
    cv2.imwrite(orig_path, cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR))

    # ── 2. Save binary mask ──
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_path  = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask_uint8)

    # ── 3. Save overlay — water highlighted in blue ──
    overlay = original_resized.copy().astype(np.float32)
    water_pixels = mask.astype(bool)

    # Blue tint on water pixels
    overlay[water_pixels, 0] = overlay[water_pixels, 0] * 0.4          # R
    overlay[water_pixels, 1] = overlay[water_pixels, 1] * 0.4          # G
    overlay[water_pixels, 2] = np.clip(
        overlay[water_pixels, 2] * 0.4 + 180, 0, 255                   # B boost
    )
    overlay = overlay.astype(np.uint8)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # ── 4. Save combined figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Water Overlay (Blue)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        f"Inference — {MODEL_NAME.upper()} | EfficientNet-B4",
        fontsize=13
    )
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{base_name}_result.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return orig_path, mask_path, overlay_path, fig_path


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("INFERENCE")
    print("=" * 55)
    print(f"Device       : {device}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Input image  : {INPUT_IMAGE}")

    # ── Build and load model ──
    model = build_model(MODEL_NAME)
    model.load_state_dict(torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    ))
    model = model.to(device)
    model.eval()
    print(f"Weights loaded: {MODEL_PATH}")

    # ── Preprocess ──
    tensor, original = preprocess(INPUT_IMAGE)
    print(f"Image shape  : {original.shape} → resized to {IMAGE_SIZE}x{IMAGE_SIZE}")

    # ── Predict ──
    mask, elapsed = predict(model, tensor, device)
    water_pct = mask.mean() * 100
    print(f"Inference time     : {elapsed*1000:.1f} ms")
    print(f"Water pixels       : {water_pct:.2f}%")

    # ── Save outputs ──
    orig_path, mask_path, overlay_path, fig_path = save_outputs(
        original, mask, OUTPUT_DIR, INPUT_IMAGE
    )

    # ── Summary ──
    print("=" * 55)
    print("OUTPUT FILES")
    print("=" * 55)
    print(f"Original   → {orig_path}")
    print(f"Mask       → {mask_path}")
    print(f"Overlay    → {overlay_path}")
    print(f"Result fig → {fig_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()