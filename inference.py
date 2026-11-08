import os
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils
from config import Config
from dataset import get_test_dataloader
from model import get_model, checkpoint_path
from preprocessing import stitch_patches, label_instances, filter_labeled_instances


def stitch_batch(batch, model, config):
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
        full_prob = cv2.resize(
            full_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
    return full_prob, filename, orig_h, orig_w


def instances_to_rows(name, instances):
    rows = []
    for i, inst_mask in enumerate(instances, start=1):
        rle = mask_utils.encode(np.asfortranarray(inst_mask))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        rows.append({
            "filament_id": f"{name}_{i}",
            "segmentation_rle": counts,
        })
    return rows


def generate_predictions(config=None, threshold=None, min_area=None, close_kernel=None):
    if config is None:
        config = Config

    config.create_dirs()
    print(f"Device: {config.DEVICE}")
    print(f"Model input: {config.MODEL_INPUT_SIZE}, Crop: {config.CROP_SIZE}")

    test_loader, _ = get_test_dataloader(config)

    model = get_model(config)
    ckpt_path = checkpoint_path(config)
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {os.path.basename(ckpt_path)} (epoch {ckpt.get('epoch', '?')}, "
          f"Dice {ckpt.get('best_dice', '?'):.4f})")

    if threshold is None:
        threshold = config.INFERENCE_THRESHOLD
    if min_area is None:
        min_area = config.INFERENCE_MIN_AREA
    if close_kernel is None:
        close_kernel = config.INFERENCE_CLOSE_KERNEL

    csv_rows = []
    mask_dir = os.path.join(config.OUTPUT_DIR, "test_masks")
    os.makedirs(mask_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            full_prob, filename, _, _ = stitch_batch(batch, model, config)
            name = os.path.splitext(filename)[0]

            labels, stats = label_instances(full_prob, threshold, close_kernel)
            instances = filter_labeled_instances(labels, stats, min_area)

            binary = (labels > 0).astype(np.uint8)
            Image.fromarray(binary * 255, mode="L").save(
                os.path.join(mask_dir, f"{name}.png")
            )
            csv_rows.extend(instances_to_rows(name, instances))

    csv_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(csv_rows)

    n_images = len(set(r["filament_id"].rsplit("_", 1)[0] for r in csv_rows)) if csv_rows else 0
    print(f"\nCheckpoint: {os.path.basename(ckpt_path)}")
    print(f"Threshold: {threshold}, Min area: {min_area}, Close kernel: {close_kernel}")
    print(f"Predictions: {len(csv_rows)} instances across {n_images} images")
    print(f"Submission: {csv_path}")
    print(f"Masks: {mask_dir}")

    return csv_rows


if __name__ == "__main__":
    generate_predictions()
