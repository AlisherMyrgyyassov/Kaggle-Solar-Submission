"""Single-image probability prediction with optional TTA (flip/rotate averaging).

Reuses the corrected inference scale (2x upscale -> 512px patches) and
3-channel ImageNet normalization, matching training for pretrained backbones.
"""
import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from preprocessing import generate_patches, stitch_patches


def detect_solar_mask(image, solar_center=(1024, 1024), solar_radius=1000, erode=12):
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        m = np.zeros(image.shape, np.uint8)
        cv2.circle(m, solar_center, solar_radius, 255, -1)
        return m
    ll = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    dm = (labels == ll).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        m = np.zeros(image.shape, np.uint8)
        cv2.circle(m, solar_center, solar_radius, 255, -1)
        return m
    hull = cv2.convexHull(contours[0])
    filled = np.zeros_like(dm)
    cv2.fillPoly(filled, [hull], 255)
    if erode > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        filled = cv2.erode(filled, ek)
    return filled


def _run_patches(patches, model, config):
    patch_size = config.MODEL_INPUT_SIZE
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    half = getattr(config, "HALF", False)
    xs = []
    for p in patches:
        if config.IN_CHANNELS == 3:
            p3 = np.repeat(p[..., None], 3, axis=2).astype(np.float32) / 255.0
            t = (torch.from_numpy(p3).permute(2, 0, 1) - mean) / std
        else:
            t = torch.from_numpy(p.astype(np.float32) / 255.0).unsqueeze(0)
        if half:
            t = t.half()
        xs.append(t)
    xs = torch.stack(xs)
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), config.BATCH_SIZE):
            out = model(xs[i:i + config.BATCH_SIZE].to(config.DEVICE))
            probs = torch.sigmoid(out).float().cpu().numpy()
            preds.extend([q.squeeze() for q in probs])
    return preds


def predict_prob_map(image, model, config, tta=False):
    sf = config.MODEL_INPUT_SIZE / config.CROP_SIZE
    h0, w0 = image.shape
    solar0 = detect_solar_mask(image, config.SOLAR_DISK_CENTER, config.SOLAR_DISK_RADIUS, config.SOLAR_DISK_ERODE)
    img2 = cv2.resize(image, (int(w0 * sf), int(h0 * sf)), interpolation=cv2.INTER_CUBIC)
    solar2 = cv2.resize(solar0, (int(w0 * sf), int(h0 * sf)), interpolation=cv2.INTER_NEAREST)
    H2, W2 = img2.shape
    patch_size = config.MODEL_INPUT_SIZE
    stride = patch_size - 64

    transforms = [
        ("id", lambda x: x),
        ("hflip", lambda x: x[:, ::-1]),
        ("vflip", lambda x: x[::-1, :]),
        ("rot180", lambda x: x[::-1, ::-1]),
    ]
    if not tta:
        transforms = transforms[:1]

    accum = np.zeros((H2, W2), dtype=np.float32)
    for name, tf in transforms:
        img_t = tf(img2)
        info = [(None, None, int(x), int(y)) for _, _, x, y in generate_patches(img_t, None, patch_size, stride)]
        patches = [img_t[y:y + patch_size, x:x + patch_size] for _, _, x, y in info]
        preds = _run_patches(patches, model, config)
        full = stitch_patches(info, preds, full_h=H2, full_w=W2, patch_size=patch_size, stride=stride)
        if name == "hflip":
            full = full[:, ::-1]
        elif name == "vflip":
            full = full[::-1, :]
        elif name == "rot180":
            full = full[::-1, ::-1]
        accum += full
    prob2 = accum / len(transforms)
    prob2[solar2 == 0] = 0
    prob = cv2.resize(prob2, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return prob


def predict_prob_map_dual(image, model, config, local_size=1024, global_size=512):
    """Dual-branch inference: local 1024px native patches + global full-disk (512)."""
    h0, w0 = image.shape
    solar0 = detect_solar_mask(image, config.SOLAR_DISK_CENTER, config.SOLAR_DISK_RADIUS, config.SOLAR_DISK_ERODE)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    g = cv2.resize(image, (global_size, global_size), interpolation=cv2.INTER_AREA)
    g3 = np.repeat(g[..., None], 3, axis=2).astype(np.float32) / 255.0
    g3 = (torch.from_numpy(g3).permute(2, 0, 1) - mean) / std
    g3 = g3.unsqueeze(0).to(config.DEVICE)

    patch_size = local_size
    stride = patch_size - 64
    info = [(None, None, int(x), int(y)) for _, _, x, y in generate_patches(image, None, patch_size, stride)]
    preds = []
    with torch.no_grad():
        for i in range(0, len(info), config.BATCH_SIZE):
            chunk = info[i:i + config.BATCH_SIZE]
            xs = []
            for _, _, x, y in chunk:
                p = image[y:y + patch_size, x:x + patch_size]
                p3 = np.repeat(p[..., None], 3, axis=2).astype(np.float32) / 255.0
                xs.append((torch.from_numpy(p3).permute(2, 0, 1) - mean) / std)
            xb = torch.stack(xs).to(config.DEVICE)
            out = model(xb, g3.expand(xb.size(0), -1, -1, -1))
            probs = torch.sigmoid(out).cpu().numpy()
            preds.extend([q.squeeze() for q in probs])

    full = stitch_patches(info, preds, full_h=h0, full_w=w0, patch_size=patch_size, stride=stride)
    full[solar0 == 0] = 0
    return full

# ---------------------------------------------------------------------------
# Single-pass full-disk inference (2026-11-03)
#
# ConvNeXtUNet is fully convolutional (7x7 depthwise + 1x1 pointwise convs,
# bilinear upsampling, no positional embeddings), so a 1024-trained model can be
# evaluated on the whole 2048px image in one forward. Measured on
# convnext_atto_1024_kd5_dec128_full over all 107 val files, both FP16:
#   patch 8x1024 : PQ 0.4321, 278 ms/img, 214.9 GMAC/img
#   single 2048  : PQ 0.4314, 103 ms/img, 107.4 GMAC/img   (delta -0.0007, ns)
# i.e. -50% MACs and 2.7x faster for no measurable PQ change. It also removes
# the 64px overlap seam and the per-patch numpy normalisation (146 ms, 101 MB
# host->device) by normalising the single uint8 image on the GPU.
# ---------------------------------------------------------------------------

D4_VIEWS = [(0, False), (0, True), (2, False), (2, True),
            (1, False), (1, True), (3, False), (3, True)]  # (rot90 k, flip)


def _d4(t, k, flip):
    if flip:
        t = torch.flip(t, dims=[3])
    return torch.rot90(t, k, dims=[2, 3]) if k else t


def _d4_inv(t, k, flip):
    if k:
        t = torch.rot90(t, -k, dims=[2, 3])
    return torch.flip(t, dims=[3]) if flip else t


def detect_solar_mask_fast(image, solar_center=(1024, 1024), solar_radius=1000,
                           erode=12, factor=4):
    """Disk mask computed on a factor-x downsampled image (~16x cheaper).

    Only the limb is affected, and the mask is used purely as an outside-disk
    gate, so the 1-2px difference is irrelevant (erosion already truncates 0%
    of GT). Measured 31.8 ms -> 5.2 ms at factor 4.
    """
    h, w = image.shape
    small = cv2.resize(image, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    m = detect_solar_mask(
        small,
        (solar_center[0] // factor, solar_center[1] // factor),
        max(1, solar_radius // factor),
        max(1, erode // factor),
    )
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def predict_prob_map_full(image, model, config, tta=1, fast_mask=True, scale=1.0):
    """Whole-image probability map in one forward pass (+ optional D4 TTA).

    ``tta`` in {1, 2, 4, 8} selects the first N dihedral views. The image is
    normalised once on the GPU; each view is a stride-view rotation/flip of the
    same tensor, so the cost is a pure forward pass per view.

    ``scale`` != 1 resizes the image before the forward and maps the probability
    back, i.e. it changes the effective filament scale (a multi-scale view).
    """
    h0, w0 = image.shape
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img_r = cv2.resize(image, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                           interpolation=interp)
    else:
        img_r = image
    hr, wr = img_r.shape
    cx, cy = config.SOLAR_DISK_CENTER
    ctr = (cx * scale, cy * scale)
    rad = config.SOLAR_DISK_RADIUS * scale
    ero = max(1, int(config.SOLAR_DISK_ERODE * scale))
    if fast_mask:
        solar0 = detect_solar_mask_fast(img_r, ctr, rad, ero)
    else:
        solar0 = detect_solar_mask(img_r, ctr, rad, ero)

    mean = torch.tensor(IMAGENET_MEAN, device=config.DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=config.DEVICE).view(1, 3, 1, 1)
    x = torch.from_numpy(np.ascontiguousarray(img_r)).to(config.DEVICE).float().div_(255.0)
    x = x.view(1, 1, hr, wr)
    if config.IN_CHANNELS == 1:
        x = (x - mean.view(1, 1, 1, 1)[..., 0:1]) / std.view(1, 1, 1, 1)[..., 0:1]
    else:
        x = (x - mean) / std          # (1,1,H,W) - (1,3,1,1) -> (1,3,H,W)

    half = getattr(config, "HALF", False)
    acc = torch.zeros((hr, wr), dtype=torch.float32, device=config.DEVICE)
    views = D4_VIEWS[:max(1, min(int(tta), 8))]
    with torch.no_grad():
        for k, flip in views:
            t = _d4(x, k, flip)
            if half:
                t = t.half()
            out = model(t)
            if isinstance(out, tuple):
                out = out[0]
            p = torch.sigmoid(out.float())
            acc += _d4_inv(p, k, flip)[0, 0]
    prob = (acc / len(views)).cpu().numpy()
    prob[solar0 == 0] = 0.0
    if scale != 1.0:
        prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return prob


def stitch_batch(batch, model, config):
    """Run a batch of patches and stitch them back into one full-image probability map."""
    patches = batch["patches"].squeeze(0)
    coords = batch["coords"]
    if isinstance(coords[0], list) and len(coords) == 1:
        coords = coords[0]
    filename = batch["filename"][0]
    orig_h = int(batch["original_shape"][0][0])
    orig_w = int(batch["original_shape"][1][0])
    solar_mask = batch["solar_mask"][0].numpy()
    scaled_h = int(batch["scaled_shape"][0][0])
    scaled_w = int(batch["scaled_shape"][1][0])

    patch_size = config.MODEL_INPUT_SIZE
    stride = patch_size - 64
    patch_preds = []
    for i in range(0, len(patches), config.BATCH_SIZE):
        batch_patches = patches[i:i + config.BATCH_SIZE].to(config.DEVICE)
        out = model(batch_patches)
        probs = torch.sigmoid(out).cpu().numpy()
        patch_preds.extend([p.squeeze() for p in probs])

    patch_info = [(None, None, int(x), int(y)) for [x, y] in coords]
    full_prob = stitch_patches(
        patch_info, patch_preds,
        full_h=scaled_h, full_w=scaled_w,
        patch_size=patch_size, stride=stride,
    )
    full_prob[solar_mask == 0] = 0
    if (scaled_h, scaled_w) != (orig_h, orig_w):
        full_prob = cv2.resize(full_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return full_prob, filename, orig_h, orig_w
