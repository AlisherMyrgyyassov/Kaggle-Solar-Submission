import os
import torch
import torch.nn as nn
from models.ultraunet import UltraUNet
from config import Config


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def get_model(config):
    model = UltraUNet(
        img_ch=config.IN_CHANNELS,
        output_ch=config.NUM_CLASSES,
        n_channels=config.ULTRA_N_CHANNELS,
    )
    return model.to(config.DEVICE)


def checkpoint_path(config, which=None):
    """Resolve a trained checkpoint. which: 'ema', 'raw', or None (use Config.INFERENCE_CHECKPOINT)."""
    if which is None:
        which = getattr(config, "INFERENCE_CHECKPOINT", "ema")
    if which == "raw":
        path = os.path.join(config.MODEL_DIR, "best_model.pth")
        if os.path.exists(path):
            return path
    ema_path = os.path.join(config.MODEL_DIR, "best_model_ema.pth")
    raw_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    if which == "ema" and os.path.exists(ema_path):
        return ema_path
    if os.path.exists(raw_path):
        return raw_path
    if os.path.exists(ema_path):
        return ema_path
    raise FileNotFoundError(f"No checkpoint in {config.MODEL_DIR}")


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, smooth=1e-6):
    logits = logits.float()
    targets = targets.float()
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def get_loss_function(alpha=0.3, beta=0.7):
    bce = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return 0.5 * bce(pred, target) + 0.5 * tversky_loss(pred, target, alpha, beta)

    return combined_loss


def get_metrics():
    def compute_metrics(pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        pred_binary = (pred > 0.5).float()

        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        fn = ((1 - pred_binary) * target).sum()

        iou = tp / (tp + fp + fn + 1e-7)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)

        return {"iou": iou.item(), "dice": dice.item()}

    return compute_metrics
