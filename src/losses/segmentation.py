import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_smooth = dice_smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


def build_default_criterion():
    return nn.BCEWithLogitsLoss()
