import os
import sys
import time
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_dataloaders
from training.losses import BCEDiceLoss
from training.metrics import compute_metrics


# ── Load Config ────────────────────────────────────────────────────────────
def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Set Seeds ──────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Build Model ────────────────────────────────────────────────────────────
def build_model(cfg):
    m = cfg["model"]
    if m["name"] == "unet":
        model = smp.Unet(
            encoder_name    = m["encoder"],
            encoder_weights = m["encoder_weights"],
            in_channels     = m["in_channels"],
            classes         = m["classes"],
            activation      = None,
        )
    elif m["name"] == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name           = m["encoder"],
            encoder_weights        = m["encoder_weights"],
            in_channels            = m["in_channels"],
            classes                = m["classes"],
            activation             = None,
            decoder_attention_type = m["attention"],
        )
    else:
        raise ValueError(f"Unknown model: {m['name']}")
    return model


# ── Train One Epoch ────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss    = 0
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0}

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        m = compute_metrics(preds.detach(), masks.detach())
        for k in total_metrics:
            total_metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


# ── Validate One Epoch ─────────────────────────────────────────────────────
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss    = 0
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)

            total_loss += loss.item()
            m = compute_metrics(preds, masks)
            for k in total_metrics:
                total_metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


# ── Save Training Curves ───────────────────────────────────────────────────
def save_curves(history, cfg, best_iou, checkpoint_dir):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve — train and val on same graph
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss",
                 color="steelblue", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",
                 color="coral",     linewidth=2)
    axes[0].set_title("Loss Curve (Train vs Val)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # IoU curve — train and val on same graph
    axes[1].plot(epochs_range, history["train_iou"], label="Train IoU",
                 color="steelblue", linewidth=2)
    axes[1].plot(epochs_range, history["val_iou"],   label="Val IoU",
                 color="coral",     linewidth=2)
    axes[1].set_title("IoU Curve (Train vs Val)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(
        f"{cfg['model']['name'].upper()} | {cfg['model']['encoder']} "
        f"| Best Val IoU: {best_iou:.4f}",
        fontsize=13
    )
    plt.tight_layout()

    curve_path = os.path.join(
        checkpoint_dir,
        f"{cfg['wandb']['run_name']}_curves.png"
    )
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {curve_path}")
    return curve_path


# ── Save Sample Predictions ────────────────────────────────────────────────
def save_predictions(model, loader, device, cfg, checkpoint_dir, n_samples=4):
    model.eval()
    images_shown = 0

    # ImageNet denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    axes[0, 0].set_title("Image",      fontsize=12)
    axes[0, 1].set_title("True Mask",  fontsize=12)
    axes[0, 2].set_title("Pred Mask",  fontsize=12)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds  = torch.sigmoid(model(images))
            preds  = (preds > 0.5).float()

            for i in range(images.size(0)):
                if images_shown >= n_samples:
                    break

                # Denormalize image for display
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * std + mean).clip(0, 1)

                true_mask = masks[i, 0].cpu().numpy()
                pred_mask = preds[i, 0].cpu().numpy()

                axes[images_shown, 0].imshow(img)
                axes[images_shown, 1].imshow(true_mask, cmap="gray")
                axes[images_shown, 2].imshow(pred_mask, cmap="gray")

                for j in range(3):
                    axes[images_shown, j].axis("off")

                images_shown += 1

            if images_shown >= n_samples:
                break

    plt.suptitle(
        f"Sample Predictions — {cfg['model']['name'].upper()} | {cfg['model']['encoder']}",
        fontsize=13
    )
    plt.tight_layout()

    pred_path = os.path.join(
        checkpoint_dir,
        f"{cfg['wandb']['run_name']}_predictions.png"
    )
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print(f"Sample predictions saved → {pred_path}")
    return pred_path


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()

    # ── Seeds ──
    set_seed(cfg["train"]["seed"])

    # ── Debug mode overrides ──
    debug = cfg["debug"]["enabled"]
    if debug:
        print("⚠️  DEBUG MODE ON — small dataset, CPU, 2 epochs")
        cfg["train"]["epochs"]     = cfg["debug"]["epochs"]
        cfg["train"]["batch_size"] = cfg["debug"]["batch_size"]

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() and not debug else "cpu")
    print(f"Device : {device}")

    # ── Paths ──
    p = cfg["paths"]
    os.makedirs(p["checkpoint_dir"], exist_ok=True)

    # ── DataLoaders ──
    train_loader, val_loader, test_loader = get_dataloaders(
        valid_files_path = p["valid_files"],
        image_dir        = p["image_dir"],
        mask_dir         = p["mask_dir"],
        batch_size       = cfg["train"]["batch_size"],
        num_workers      = cfg["train"]["num_workers"],
        seed             = cfg["train"]["seed"],
        max_samples      = cfg["debug"]["max_samples"] if debug else None,
    )

    # ── Model ──
    model = build_model(cfg).to(device)
    print(f"Model  : {cfg['model']['name']} | Encoder: {cfg['model']['encoder']}")

    # ── Loss, Optimizer, Scheduler ──
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = cfg["scheduler"]["mode"],
        factor   = cfg["scheduler"]["factor"],
        patience = cfg["scheduler"]["patience"],
        min_lr   = cfg["scheduler"]["min_lr"],
    )

    # ── Resume from checkpoint ──
    start_epoch = 0
    best_iou    = 0.0
    ckpt_cfg    = cfg["checkpoint"]

    if ckpt_cfg["resume"] and os.path.exists(ckpt_cfg["resume_path"]):
        ckpt = torch.load(ckpt_cfg["resume_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt["best_iou"]
        print(f"Resumed from epoch {start_epoch} | Best IoU so far: {best_iou:.4f}")

    # ── W&B Init ──
    if not debug:
        wandb.init(
            project = cfg["wandb"]["project"],
            name    = cfg["wandb"]["run_name"],
            entity  = cfg["wandb"]["entity"],
            config  = {
                "model"              : cfg["model"]["name"],
                "encoder"            : cfg["model"]["encoder"],
                "epochs"             : cfg["train"]["epochs"],
                "batch_size"         : cfg["train"]["batch_size"],
                "lr"                 : cfg["train"]["lr"],
                "image_size"         : cfg["train"]["image_size"],
                "early_stop_patience": cfg["early_stopping"]["patience"],
            }
        )

    # ── History tracking ──
    history = {
        "train_loss": [], "val_loss": [],
        "train_iou":  [], "val_iou":  [],
    }

    # ── Early Stopping State ──
    es_cfg      = cfg["early_stopping"]
    es_counter  = 0
    es_best_iou = best_iou

    # ── Training Loop ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Starting training for {cfg['train']['epochs']} epochs")
    print("=" * 60)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t_start = time.time()

        train_loss, train_m = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_m   = val_epoch(model, val_loader, criterion, device)

        scheduler.step(val_m["iou"])
        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Track history ──
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_m["iou"])
        history["val_iou"].append(val_m["iou"])

        # ── Console log ──
        print(
            f"Epoch [{epoch+1:03d}/{cfg['train']['epochs']}] "
            f"| Time: {elapsed:.1f}s "
            f"| LR: {current_lr:.6f} "
            f"| Train Loss: {train_loss:.4f}  IoU: {train_m['iou']:.4f} "
            f"| Val Loss: {val_loss:.4f}  IoU: {val_m['iou']:.4f} "
            f"| Dice: {val_m['dice']:.4f} "
            f"| Prec: {val_m['precision']:.4f} "
            f"| Rec: {val_m['recall']:.4f}"
        )

        # ── W&B log ──
        if not debug:
            wandb.log({
                "epoch"           : epoch + 1,
                "lr"              : current_lr,
                "train/loss"      : train_loss,
                "train/iou"       : train_m["iou"],
                "train/dice"      : train_m["dice"],
                "train/precision" : train_m["precision"],
                "train/recall"    : train_m["recall"],
                "val/loss"        : val_loss,
                "val/iou"         : val_m["iou"],
                "val/dice"        : val_m["dice"],
                "val/precision"   : val_m["precision"],
                "val/recall"      : val_m["recall"],
            })

        # ── Save best model ──
        if ckpt_cfg["save_best"] and val_m["iou"] > best_iou + es_cfg["min_delta"]:
            best_iou = val_m["iou"]
            best_path = os.path.join(p["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), best_path)
            if not debug:
                wandb.save(best_path)
            print(f"  ✅ Best model saved — Val IoU: {best_iou:.4f}")

        # ── Save last checkpoint ──
        if ckpt_cfg["save_last"]:
            last_path = os.path.join(p["checkpoint_dir"], "last_checkpoint.pth")
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_iou"       : best_iou,
            }, last_path)

        # ── Early stopping ──
        if val_m["iou"] > es_best_iou + es_cfg["min_delta"]:
            es_best_iou = val_m["iou"]
            es_counter  = 0
        else:
            es_counter += 1
            print(f"  Early stopping counter: {es_counter}/{es_cfg['patience']}")
            if es_counter >= es_cfg["patience"]:
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break

    # ── Save curves and predictions ────────────────────────────────────────
    curve_path = save_curves(history, cfg, best_iou, p["checkpoint_dir"])
    pred_path  = save_predictions(model, val_loader, device, cfg, p["checkpoint_dir"])

    # ── Upload to W&B ──
    if not debug:
        wandb.log({
            "training_curves"     : wandb.Image(curve_path),
            "sample_predictions"  : wandb.Image(pred_path),
        })

    # ── Final Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Training complete. Best Val IoU: {best_iou:.4f}")
    print(f"Best model   → {os.path.join(p['checkpoint_dir'], 'best_model.pth')}")
    print(f"Curves saved → {curve_path}")
    print(f"Predictions  → {pred_path}")
    print("=" * 60)

    if not debug:
        wandb.finish()


if __name__ == "__main__":
    main()