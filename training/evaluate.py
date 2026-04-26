import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_dataloaders
from training.metrics import compute_metrics

# ── Change these before running ────────────────────────────────────────────
MODEL_PATH = r"outputs\results_UNet++\outputs\checkpoints\best_model.pth"
MODEL_NAME = "unetplusplus"  # "unet" or "unetplusplus"
# MODEL_NAME = "unet"
# ──────────────────────────────────────────────────────────────────────────


# ── Load Config ────────────────────────────────────────────────────────────
def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Build Model ────────────────────────────────────────────────────────────
def build_model(cfg, model_name):
    m = cfg["model"]

    # Attention is tied to model type — do not read from config
    # config.yaml may have been changed between runs
    attention = "scse" if model_name == "unetplusplus" else None

    if model_name == "unet":
        model = smp.Unet(
            encoder_name    = m["encoder"],
            encoder_weights = m["encoder_weights"],
            in_channels     = m["in_channels"],
            classes         = m["classes"],
            activation      = None,
        )
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name           = m["encoder"],
            encoder_weights        = m["encoder_weights"],
            in_channels            = m["in_channels"],
            classes                = m["classes"],
            activation             = None,
            decoder_attention_type = attention,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ── Evaluate on Test Set ───────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images)
            m = compute_metrics(preds, masks)
            for k in total_metrics:
                total_metrics[k] += m[k]

    n = len(loader)
    return {k: v / n for k, v in total_metrics.items()}


# ── Save Test Predictions ──────────────────────────────────────────────────
def save_predictions(model, loader, device, save_path, n_samples=8):
    model.eval()
    images_shown = 0
    collected = {"images": [], "true_masks": [], "pred_masks": []}

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds  = torch.sigmoid(model(images))
            preds  = (preds > 0.5).float()

            for i in range(images.size(0)):
                if images_shown >= n_samples:
                    break

                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * std + mean).clip(0, 1)

                collected["images"].append(img)
                collected["true_masks"].append(masks[i, 0].cpu().numpy())
                collected["pred_masks"].append(preds[i, 0].cpu().numpy())
                images_shown += 1

            if images_shown >= n_samples:
                break

    # Use actual number collected — avoids empty subplots
    n_actual = len(collected["images"])
    fig, axes = plt.subplots(n_actual, 3, figsize=(12, n_actual * 4))

    # Handle case where n_actual == 1 (axes won't be 2D)
    if n_actual == 1:
        axes = np.expand_dims(axes, 0)

    axes[0, 0].set_title("Image",     fontsize=12)
    axes[0, 1].set_title("True Mask", fontsize=12)
    axes[0, 2].set_title("Pred Mask", fontsize=12)

    for idx in range(n_actual):
        axes[idx, 0].imshow(collected["images"][idx])
        axes[idx, 1].imshow(collected["true_masks"][idx], cmap="gray")
        axes[idx, 2].imshow(collected["pred_masks"][idx], cmap="gray")
        for j in range(3):
            axes[idx, j].axis("off")

    plt.suptitle(
        f"Test Set Predictions — {MODEL_NAME.upper()} | {n_actual} samples",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Test predictions saved → {save_path}")


# ── Save Results to Text File ──────────────────────────────────────────────
def save_results(metrics, save_path, model_name, model_path):
    with open(save_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"TEST SET EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model      : {model_name}\n")
        f.write(f"Weights    : {model_path}\n")
        f.write("-" * 50 + "\n")
        f.write(f"IoU        : {metrics['iou']:.4f}\n")
        f.write(f"Dice       : {metrics['dice']:.4f}\n")
        f.write(f"Precision  : {metrics['precision']:.4f}\n")
        f.write(f"Recall     : {metrics['recall']:.4f}\n")
        f.write("=" * 50 + "\n")
    print(f"Results saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    p   = cfg["paths"]

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Test DataLoader only ──
    # num_workers=0 on CPU (Windows safe), use config value on GPU
    num_workers = cfg["train"]["num_workers"] if torch.cuda.is_available() else 0

    _, _, test_loader = get_dataloaders(
        valid_files_path = p["valid_files"],
        image_dir        = p["image_dir"],
        mask_dir         = p["mask_dir"],
        batch_size       = cfg["train"]["batch_size"],
        num_workers      = num_workers,
        seed             = cfg["train"]["seed"],
    )
    print(f"Test samples : {len(test_loader.dataset)}")

    # ── Build and load model ──
    model = build_model(cfg, MODEL_NAME)
    model.load_state_dict(torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    ))
    model = model.to(device)
    model.eval()
    print(f"Model loaded : {MODEL_NAME} from {MODEL_PATH}")

    # ── Run evaluation ──
    print("\nRunning evaluation on test set...")
    metrics = evaluate(model, test_loader, device)

    # ── Print results ──
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Model      : {MODEL_NAME}")
    print(f"Weights    : {MODEL_PATH}")
    print("-" * 50)
    print(f"IoU        : {metrics['iou']:.4f}")
    print(f"Dice       : {metrics['dice']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print("=" * 50)

    # ── Save predictions and results ──
    os.makedirs(p["checkpoint_dir"], exist_ok=True)

    pred_path   = os.path.join(
        p["checkpoint_dir"],
        f"{MODEL_NAME}_test_predictions.png"
    )
    result_path = os.path.join(
        p["checkpoint_dir"],
        f"{MODEL_NAME}_test_results.txt"
    )

    save_predictions(model, test_loader, device, pred_path)
    save_results(metrics, result_path, MODEL_NAME, MODEL_PATH)


if __name__ == "__main__":
    main()