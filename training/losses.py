import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for binary segmentation.

    BCE  : stabilizes early training (pixel-wise)
    Dice : improves region overlap (pushes IoU higher)

    Input:
        pred   : raw logits  (batch, 1, 256, 256) - no sigmoid applied yet
        target : binary mask (batch, 1, 256, 256) - values 0.0 or 1.0

    Output:
        scalar loss value = BCE Loss + Dice Loss
    """

    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.smooth   = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()  # applies sigmoid internally

    def dice_loss(self, pred, target):
        # Apply sigmoid to convert logits → probabilities [0, 1]
        pred = torch.sigmoid(pred)

        # Flatten spatial dims for computation
        pred   = pred.view(-1)
        target = target.view(-1)

        # Dice Score = (2 * intersection) / (sum of pred + sum of target)
        intersection = (pred * target).sum()
        dice_score   = (2.0 * intersection + self.smooth) / \
                       (pred.sum() + target.sum() + self.smooth)

        # Dice Loss = 1 - Dice Score
        return 1.0 - dice_score

    def forward(self, pred, target):
        bce  = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return bce + dice


# ── Sanity Check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    criterion = BCEDiceLoss()

    # Simulate model output (raw logits) and binary mask
    pred   = torch.randn(4, 1, 256, 256)   # fake logits
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()  # fake binary mask

    loss = criterion(pred, target)
    print(f"Combined BCE+Dice Loss : {loss.item():.4f}")
    print("losses.py sanity check passed.")