import torch
SMOOTH = 1e-6
def iou_score(pred, target):
    """
    Intersection over Union (IoU)
    pred   : raw logits  (batch, 1, H, W)
    target : binary mask (batch, 1, H, W) values 0.0 or 1.0
    returns: scalar IoU value
    """
    pred   = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection

    return (intersection + SMOOTH) / (union + SMOOTH)

def dice_score(pred, target):
    """
    Dice Coefficient
    pred   : raw logits  (batch, 1, H, W)
    target : binary mask (batch, 1, H, W) values 0.0 or 1.0
    returns: scalar Dice value
    """
    pred   = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    return (2.0 * intersection + SMOOTH) / \
           (pred.sum() + target.sum() + SMOOTH)

def precision_score(pred, target):
    """
    Precision = TP / (TP + FP)
    pred   : raw logits  (batch, 1, H, W)
    target : binary mask (batch, 1, H, W) values 0.0 or 1.0
    returns: scalar Precision value
    """
    pred   = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    pred   = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()

    return (tp + SMOOTH) / (tp + fp + SMOOTH)

def recall_score(pred, target):
    """
    Recall = TP / (TP + FN)
    pred   : raw logits  (batch, 1, H, W)
    target : binary mask (batch, 1, H, W) values 0.0 or 1.0
    returns: scalar Recall value
    """
    pred   = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    pred   = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()

    return (tp + SMOOTH) / (tp + fn + SMOOTH)

def compute_metrics(pred, target):
    """
    Compute all 4 metrics at once.
    Called by train.py and evaluate.py

    Returns dict:
        {
            "iou"      : float,
            "dice"     : float,
            "precision": float,
            "recall"   : float
        }
    """
    return {
        "iou"      : iou_score(pred, target).item(),
        "dice"     : dice_score(pred, target).item(),
        "precision": precision_score(pred, target).item(),
        "recall"   : recall_score(pred, target).item(),
    }

# ── Sanity Check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1 — perfect prediction (pred == target)
    target   = torch.randint(0, 2, (4, 1, 256, 256)).float()
    pred_perfect = target * 10.0 - 5.0  # logits that map exactly to target

    metrics = compute_metrics(pred_perfect, target)
    print("=" * 45)
    print("TEST 1 — Perfect Prediction")
    print("=" * 45)
    print(f"IoU       : {metrics['iou']:.4f}  (expected ~1.0)")
    print(f"Dice      : {metrics['dice']:.4f}  (expected ~1.0)")
    print(f"Precision : {metrics['precision']:.4f}  (expected ~1.0)")
    print(f"Recall    : {metrics['recall']:.4f}  (expected ~1.0)")

    # Test 2 — random prediction
    pred_random = torch.randn(4, 1, 256, 256)
    metrics = compute_metrics(pred_random, target)
    print("\n" + "=" * 45)
    print("TEST 2 — Random Prediction")
    print("=" * 45)
    print(f"IoU       : {metrics['iou']:.4f}  (expected ~0.3-0.5)")
    print(f"Dice      : {metrics['dice']:.4f}  (expected ~0.4-0.6)")
    print(f"Precision : {metrics['precision']:.4f}  (expected ~0.4-0.6)")
    print(f"Recall    : {metrics['recall']:.4f}  (expected ~0.4-0.6)")

    print("\nmetrics.py sanity check passed.")