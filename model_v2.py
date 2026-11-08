"""ConvNeXt-tiny U-Net (pretrained timm encoder + smp decoder) + soft-clDice losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from config import Config


class ConvNeXtUNet(nn.Module):
    ENCODER_CHANNELS = {
        "convnext_tiny": [3, 96, 192, 384, 768],
        "convnext_small": [3, 96, 192, 384, 768],
        "convnext_base": [3, 128, 256, 512, 1024],
        "convnext_nano": [3, 80, 160, 320, 640],
        "convnext_pico": [3, 64, 128, 256, 512],
        "convnext_femto": [3, 48, 96, 192, 384],
        "convnext_atto": [3, 40, 80, 160, 320],
        "convnext_atto_ols": [3, 40, 80, 160, 320],
    }
    DECODER_CHANNELS = (256, 128, 64, 32)

    def __init__(self, encoder_name="convnext_tiny", in_channels=3, classes=1,
                 pretrained=True, decoder_channels=None, aux_spine=False,
                 head_upsample=1):
        super().__init__()
        self.decoder_channels = tuple(decoder_channels) if decoder_channels else self.DECODER_CHANNELS
        self.aux_spine = aux_spine
        self.in_channels = in_channels
        self.head_upsample = int(head_upsample)
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, features_only=True,
            in_chans=in_channels,
        )
        self._n_drop = 0
        encoder_channels = self._resolve_encoder_channels(encoder_name)
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=len(self.decoder_channels),
            use_norm="batchnorm",
            attention_type=None,
            add_center_block=False,
            interpolation_mode="bilinear",
        )
        self.head = nn.Conv2d(self.decoder_channels[-1], classes, 1)
        self.spine_head = nn.Conv2d(self.decoder_channels[-1], 1, 1) if aux_spine else None

    def _resolve_encoder_channels(self, encoder_name):
        """Return [input_channels, s4, s8, s16, s32] for the 4-stage decoder.

        ConvNeXt is hardcoded; any other timm features_only encoder is handled
        generically by dropping stride-2 stages so the strides match (4/8/16/32).
        """
        if encoder_name in self.ENCODER_CHANNELS:
            return self.ENCODER_CHANNELS[encoder_name]
        ch = list(self.encoder.feature_info.channels())
        red = list(self.encoder.feature_info.reduction())
        n = len(self.decoder_channels)
        # Keep the n deepest stages. A 5-stage encoder (ResNet/EfficientNetV2,
        # reductions 2/4/8/16/32) therefore keeps its stride-2 skip when the
        # decoder has 5 blocks; a 4-block decoder drops it, as before.
        drop = max(0, len(ch) - n)
        self._n_drop = drop
        kept = ch[drop:drop + n]
        if len(kept) != len(self.decoder_channels):
            raise ValueError(
                f"Encoder {encoder_name} yields {len(kept)} usable stages "
                f"(reductions {red}), expected {len(self.decoder_channels)}."
            )
        return [self.in_channels] + kept

    def forward(self, x, return_spine=False):
        h, w = x.shape[-2:]
        skips = self.encoder(x)
        n = len(self.decoder_channels)
        if self._n_drop:
            skips = skips[self._n_drop:self._n_drop + n]
        # smp's UnetDecoder takes the target size from features[0] (it drops that
        # entry before concatenating, so only .shape is read). Passing a strided
        # half-resolution view therefore makes the last block run at stride 2
        # instead of stride 1: -22% MACs at 2048, and the logit is upsampled 2x.
        feat0 = (x[:, :, ::self.head_upsample, ::self.head_upsample]
                 if self.head_upsample > 1 else x)
        features = [feat0] + list(skips)
        out = self.decoder(features)
        mask = self.head(out)
        if self.head_upsample > 1:
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        if return_spine and self.spine_head is not None:
            spine = self.spine_head(out)
            if self.head_upsample > 1:
                spine = F.interpolate(spine, size=(h, w), mode="bilinear", align_corners=False)
            return mask, spine
        return mask


def get_convnext_unet(config, encoder_name="convnext_tiny", pretrained=True,
                      decoder_channels=None, aux_spine=False, head_upsample=1):
    model = ConvNeXtUNet(
        encoder_name=encoder_name,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        aux_spine=aux_spine,
        head_upsample=head_upsample,
    )
    return model.to(config.DEVICE)


class DualBranchUNet(nn.Module):
    """HookNet-style dual-magnification: local 1024px branch + global (whole-disk
    downsampled to 512) branch, fused at the bottleneck."""

    def __init__(self, encoder_name="convnext_tiny", global_size=512, classes=1, pretrained=True):
        super().__init__()
        self.local_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        self.global_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        # convnext channels: [96,192,384,768]
        self.fuse = nn.Conv2d(768 * 2, 768, 1)
        self.decoder = UnetDecoder(
            encoder_channels=[3, 96, 192, 384, 768],
            decoder_channels=(256, 128, 64, 32),
            n_blocks=4,
            use_norm="batchnorm",
            attention_type=None,
            add_center_block=False,
            interpolation_mode="bilinear",
        )
        self.head = nn.Conv2d(32, classes, 1)
        self.global_size = global_size

    def forward(self, local, global_disk):
        g_skips = self.global_encoder(global_disk)   # [128,64,32,16] for 512 input
        l_skips = self.local_encoder(local)          # [256,128,64,32] for 1024 input
        g_b = g_skips[-1]
        l_b = l_skips[-1]
        g_up = F.interpolate(g_b, size=l_b.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([l_b, g_up], dim=1))
        features = [local, l_skips[0], l_skips[1], l_skips[2], fused]
        x = self.decoder(features)
        return self.head(x)


def get_dual_branch_unet(config, encoder_name="convnext_tiny", pretrained=True):
    model = DualBranchUNet(encoder_name=encoder_name, global_size=512, classes=config.NUM_CLASSES, pretrained=pretrained)
    return model.to(config.DEVICE)


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, smooth=1e-6):
    probs = torch.sigmoid(logits).float().view(logits.size(0), -1)
    targets = targets.float().view(targets.size(0), -1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - t).mean()


def soft_skeletonize(x, iters=10):
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(F.max_pool2d(x, kernel_size=3, stride=1, padding=1) - x)
        x = F.relu(min_pool - contour)
    return x


def soft_dice(pred, target, smooth=1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (denom + smooth)).mean()


def kd_mse_loss(logits, teacher):
    """MSE between student sigmoid and teacher soft prob (ensemble)."""
    probs = torch.sigmoid(logits)
    return F.mse_loss(probs, teacher)


def kd_kl_loss(logits, teacher, T=2.0):
    """Temperature-scaled forward KL: student sigmoid toward teacher prob."""
    eps = 1e-7
    p_t = teacher.clamp(eps, 1 - eps)
    logit_t = torch.log(p_t / (1 - p_t)) / T
    p_s = torch.sigmoid(logits / T)
    p_tt = torch.sigmoid(logit_t)
    kl = p_tt * (torch.log(p_tt + eps) - torch.log(p_s + eps)) + \
        (1 - p_tt) * (torch.log(1 - p_tt + eps) - torch.log(1 - p_s + eps))
    return (T * T) * kl.mean()


def spine_heatmap_loss(logits, target, pos_weight=50.0):
    """Auxiliary spine loss: positive-weighted BCE + soft Dice on the soft heatmap.

    Spine polylines are ~1% of pixels, so plain BCE collapses to all-zero; the
    pos_weight + Dice keep the head from doing that.
    """
    target = target.float()
    pw = torch.full((), pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    dice = 1.0 - soft_dice(torch.sigmoid(logits), target)
    return bce + dice


def soft_cl_dice(logits, targets, iters=5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    vp = soft_skeletonize(probs, iters=iters)
    vl = soft_skeletonize(targets, iters=iters)
    tprec = ((vp * targets).sum(dim=(1, 2, 3)) + smooth) / (vp.sum(dim=(1, 2, 3)) + smooth)
    tsens = ((vl * probs).sum(dim=(1, 2, 3)) + smooth) / (vl.sum(dim=(1, 2, 3)) + smooth)
    cl_dice = 2 * tprec * tsens / (tprec + tsens + smooth)
    return (1 - cl_dice).mean()


def get_loss_function(alpha=0.3, beta=0.7, cldice_w=0.0):
    bce = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        loss = 0.5 * bce(pred, target) + 0.5 * tversky_loss(pred, target, alpha, beta)
        if cldice_w > 0:
            loss = loss + cldice_w * soft_cl_dice(pred, target)
        return loss

    return combined_loss
