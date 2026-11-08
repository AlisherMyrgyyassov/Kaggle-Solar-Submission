import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils
from preprocessing import create_solar_mask, generate_patches


class ScaleAwareDataset(Dataset):
    def __init__(self, image_dir, annotation_data, image_ids,
                 crop_size=256, model_size=512, augment=False,
                 pos_bias=0.75, crops_per_image=8, seed=42):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.model_size = model_size
        self.augment = augment
        self.pos_bias = pos_bias
        self.crops_per_image = crops_per_image
        self.rng = random.Random(seed)

        self.image_info = {}
        for img in annotation_data["images"]:
            if img["id"] in image_ids:
                self.image_info[img["id"]] = img

        self.annotations_by_image = {}
        self.bboxes_by_image = {}
        for ann in annotation_data["annotations"]:
            img_id = ann["image_id"]
            if img_id in image_ids:
                if img_id not in self.annotations_by_image:
                    self.annotations_by_image[img_id] = []
                    self.bboxes_by_image[img_id] = []
                self.annotations_by_image[img_id].append(ann)
                self.bboxes_by_image[img_id].append(ann.get("bbox", None))

        self.image_ids = image_ids
        self._mask_cache = {}

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101,
                         mask_interpolation=cv2.INTER_NEAREST, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.ElasticTransform(alpha=120, sigma=50, p=0.3),
                A.GridDistortion(distort_limit=0.3, p=0.2),
                A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def _create_mask(self, img_id):
        if img_id in self._mask_cache:
            return self._mask_cache[img_id]

        info = self.image_info[img_id]
        h, w = info["height"], info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        if img_id in self.annotations_by_image:
            for ann in self.annotations_by_image[img_id]:
                for seg in ann["segmentation"]:
                    if len(seg) < 6:
                        continue
                    pts = np.array([(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)

        self._mask_cache[img_id] = mask
        return mask

    def __len__(self):
        return len(self.image_ids) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        img_id = self.image_ids[img_idx]

        info = self.image_info[img_id]
        img_path = os.path.join(self.image_dir, info["file_name"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = self._create_mask(img_id)

        h, w = image.shape
        s = self.crop_size

        bboxes = self.bboxes_by_image.get(img_id, [])
        valid_bboxes = [b for b in bboxes if b is not None]

        if self.augment and valid_bboxes and self.rng.random() < self.pos_bias:
            bx, by, bw, bh = self.rng.choice(valid_bboxes)
            cx = bx + bw / 2 + self.rng.uniform(-s / 4, s / 4)
            cy = by + bh / 2 + self.rng.uniform(-s / 4, s / 4)
            x0 = int(np.clip(cx - s / 2, 0, w - s))
            y0 = int(np.clip(cy - s / 2, 0, h - s))
        else:
            x0 = self.rng.randint(0, max(1, w - s))
            y0 = self.rng.randint(0, max(1, h - s))

        img_crop = image[y0:y0 + s, x0:x0 + s]
        mask_crop = mask[y0:y0 + s, x0:x0 + s]

        if s != self.model_size:
            img_crop = cv2.resize(img_crop, (self.model_size, self.model_size), interpolation=cv2.INTER_CUBIC)
            mask_crop = cv2.resize(mask_crop, (self.model_size, self.model_size), interpolation=cv2.INTER_LINEAR)

        transformed = self.transform(image=img_crop, mask=mask_crop)

        img_tensor = transformed["image"].float() / 255.0
        mask_tensor = (transformed["mask"].unsqueeze(0).float() / 255.0).clamp(0, 1)

        return {"image": img_tensor, "mask": mask_tensor}


class InferenceDataset(Dataset):
    def __init__(self, image_dir, patch_size=512, stride=448,
                 solar_radius=1000, solar_center=(1024, 1024),
                 solar_disk_erode=12, filenames=None, scale_factor=1,
                 in_channels=1, normalize=False):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.stride = stride
        self.solar_radius = solar_radius
        self.solar_center = solar_center
        self.solar_disk_erode = solar_disk_erode
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.normalize = normalize
        if filenames is not None:
            self.image_files = sorted(filenames)
        else:
            self.image_files = sorted(os.listdir(image_dir))

        self.transform = A.Compose([ToTensorV2()])

    def _detect_solar_disk(self, image):
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels <= 1:
            return create_solar_mask(image.shape, self.solar_center, self.solar_radius)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        disk_mask = (labels == largest_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(disk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return create_solar_mask(image.shape, self.solar_center, self.solar_radius)
        hull = cv2.convexHull(contours[0])
        filled = np.zeros_like(disk_mask)
        cv2.fillPoly(filled, [hull], 255)
        if self.solar_disk_erode > 0:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.solar_disk_erode, self.solar_disk_erode))
            filled = cv2.erode(filled, erode_kernel)
        return filled

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        orig_shape = image.shape

        solar_mask = self._detect_solar_disk(image)

        if self.scale_factor != 1:
            h, w = image.shape
            image = cv2.resize(
                image, (int(w * self.scale_factor), int(h * self.scale_factor)),
                interpolation=cv2.INTER_CUBIC,
            )
            solar_mask = cv2.resize(
                solar_mask,
                (int(w * self.scale_factor), int(h * self.scale_factor)),
                interpolation=cv2.INTER_NEAREST,
            )

        patches_info = generate_patches(image, None, self.patch_size, self.stride)
        patch_tensors = []
        for patch, _, px, py in patches_info:
            if self.in_channels == 3:
                patch = np.repeat(patch[..., None], 3, axis=2)
            transformed = self.transform(image=patch)
            t = transformed["image"].float() / 255.0
            if self.normalize:
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                t = (t - mean) / std
            patch_tensors.append(t)

        patch_batch = torch.stack(patch_tensors)
        patch_coords = [(px, py) for _, _, px, py in patches_info]

        return {
            "patches": patch_batch,
            "coords": patch_coords,
            "filename": filename,
            "original_shape": orig_shape,
            "scaled_shape": image.shape,
            "solar_mask": solar_mask,
        }


def load_and_parse_annotations(json_path):
    with open(json_path, "r") as f:
        content = f.read()

    last_complete = content.rfind('"segmentation"')
    search_end = last_complete + 50
    next_close = content.find('}]', search_end)
    if next_close > 0:
        valid_prefix = content[:next_close + 2]
    else:
        pos = content.rfind('"id":')
        end = content.find('}', pos)
        valid_prefix = content[:end + 1] + "\n]}"

    trimmed = valid_prefix.rstrip()
    if not trimmed.endswith("]}"):
        last_entry_end = trimmed.rfind('"}')
        if last_entry_end > 0:
            trimmed = trimmed[:last_entry_end + 2] + "\n]\n}"

    try:
        data = json.loads(trimmed)
    except json.JSONDecodeError:
        content_fixed = content[:content.rfind('"segmentation"')]
        content_fixed = content_fixed[:content_fixed.rfind('}')]
        content_fixed += "\n]\n}"
        data = json.loads(content_fixed)

    print(f"Loaded {len(data['images'])} images, {len(data['annotations'])} annotations")
    return data


def create_data_splits(annotation_data, val_split=0.15, seed=42):
    annotated_ids = set()
    for ann in annotation_data["annotations"]:
        annotated_ids.add(ann["image_id"])

    available_ids = [img["id"] for img in annotation_data["images"] if img["id"] in annotated_ids]
    print(f"Images with annotations: {len(available_ids)}")

    train_ids, val_ids = train_test_split(available_ids, test_size=val_split, random_state=seed)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    return train_ids, val_ids


def get_dataloaders(config):
    data = load_and_parse_annotations(config.ANNOTATION_FILE)
    train_ids, val_ids = create_data_splits(data, config.VAL_SPLIT, config.SEED)

    train_dataset = ScaleAwareDataset(
        image_dir=config.TRAIN_IMG_DIR,
        annotation_data=data,
        image_ids=train_ids,
        crop_size=config.CROP_SIZE,
        model_size=config.MODEL_INPUT_SIZE,
        augment=True,
        pos_bias=config.POS_BIAS,
        crops_per_image=config.CROPS_PER_IMAGE,
        seed=config.SEED,
    )

    val_dataset = ScaleAwareDataset(
        image_dir=config.TRAIN_IMG_DIR,
        annotation_data=data,
        image_ids=val_ids,
        crop_size=config.CROP_SIZE,
        model_size=config.MODEL_INPUT_SIZE,
        augment=False,
        pos_bias=0.0,
        crops_per_image=8,
        seed=config.SEED + 1,
    )

    loader_kwargs = dict(
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    if config.NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def rasterize_instance_rles(annotations, height, width):
    """One COCO RLE per filament annotation (class-agnostic). Empty polygons skipped."""
    rles = []
    for ann in annotations:
        mask = np.zeros((height, width), dtype=np.uint8)
        segs = ann.get("segmentation") or []
        for seg in segs:
            if len(seg) < 6:
                continue
            pts = np.array([(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        if mask.any():
            rles.append(mask_utils.encode(np.asfortranarray(mask)))
    return rles


def filenames_for_ids(annotation_data, image_ids):
    id_to_name = {img["id"]: img["file_name"] for img in annotation_data["images"]}
    names = []
    missing = 0
    for img_id in image_ids:
        name = id_to_name.get(img_id)
        if name is None:
            missing += 1
            continue
        names.append(name)
    if missing:
        print(f"Warning: {missing} split ids had no file_name")
    return names


def get_inference_dataloader(config, image_dir, filenames=None):
    dataset = InferenceDataset(
        image_dir=image_dir,
        patch_size=config.MODEL_INPUT_SIZE,
        stride=config.MODEL_INPUT_SIZE - 64,
        solar_radius=config.SOLAR_DISK_RADIUS,
        solar_center=config.SOLAR_DISK_CENTER,
        solar_disk_erode=config.SOLAR_DISK_ERODE,
        filenames=filenames,
        scale_factor=config.MODEL_INPUT_SIZE / config.CROP_SIZE,
        in_channels=config.IN_CHANNELS,
        normalize=config.NORMALIZE,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return loader, dataset


def get_test_dataloader(config):
    return get_inference_dataloader(config, config.TEST_IMG_DIR)


# ---------------------------------------------------------------------------
# File-level (leak-free) split + union-mask dataset for pretrained backbones
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_file_splits(annotation_data, val_split=0.15, seed=42):
    """Split unique file_names (not annotator records) to avoid train/val leakage."""
    files = sorted({img["file_name"] for img in annotation_data["images"]})
    train_files, val_files = train_test_split(files, test_size=val_split, random_state=seed)
    print(f"File split: train {len(train_files)}, val {len(val_files)}")
    return train_files, val_files


def rasterize_binary_mask(anns, h, w, fill=255):
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        for seg in ann.get("segmentation") or []:
            if len(seg) < 6:
                continue
            pts = np.array([(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], fill)
    return mask


def collect_annotator_polygons(annotation_data):
    """{file_name: [{'image_id', 'h', 'w', 'anns'}, ...]} in JSON order."""
    anns_by = {}
    for a in annotation_data["annotations"]:
        anns_by.setdefault(a["image_id"], []).append(a)
    by_file = {}
    for img in annotation_data["images"]:
        fname = img["file_name"]
        by_file.setdefault(fname, []).append({
            "image_id": img["id"],
            "h": img["height"],
            "w": img["width"],
            "anns": anns_by.get(img["id"], []),
        })
    return by_file


def build_annotator_mask_map(annotation_data):
    """{file_name: [uint8 0/255 mask per annotator record]}."""
    by_file = collect_annotator_polygons(annotation_data)
    out = {}
    for fname, recs in by_file.items():
        out[fname] = [rasterize_binary_mask(r["anns"], r["h"], r["w"]) for r in recs]
    return out


def build_file_mask_map(annotation_data, mode="single"):
    """Return {file_name: mask} built from annotator records.

    mode:
      single / union / majority -> uint8 0/255
      staple / staple_hard -> uint8 0/255 STAPLE thresholded at 0.5
      staple_soft -> float32 [0, 1] STAPLE posterior
    Files with no annotations get an all-zero mask.
    """
    if mode in ("staple", "staple_hard", "staple_soft"):
        from staple import build_staple_mask_map
        kind = "global" if mode.endswith("_global") else "file"
        return build_staple_mask_map(annotation_data, mode=mode, kind=kind)

    by_file = collect_annotator_polygons(annotation_data)
    out = {}
    for img in annotation_data["images"]:
        fname = img["file_name"]
        if fname in out:
            continue
        recs = by_file.get(fname, [])
        h, w = img["height"], img["width"]
        if not recs:
            out[fname] = np.zeros((h, w), dtype=np.uint8)
            continue
        if mode == "single":
            labeled = [r for r in recs if r["anns"]] or recs[:1]
            out[fname] = rasterize_binary_mask(labeled[0]["anns"], h, w)
            continue
        masks = [rasterize_binary_mask(r["anns"], h, w, fill=1) for r in recs]
        vote = np.sum(masks, axis=0) if masks else np.zeros((h, w), dtype=np.uint8)
        if mode == "majority":
            n = len(masks)
            thr = n // 2 + 1 if n else 1
            out[fname] = (vote >= thr).astype(np.uint8) * 255
        else:
            out[fname] = (vote >= 1).astype(np.uint8) * 255
    return out


def build_file_spine_map(annotation_data, filenames=None, thickness=3, blur_sigma=2.0):
    """Return {file_name: uint8 soft spine heatmap (0..255)}.

    Mirrors `build_file_mask_map(mode="single")`: uses the first labeled annotator
    record's `spine` polylines (MAGFiLO metadata, allowed). Drawn as polylines then
    Gaussian-blurred into a soft target; empty / unlabeled files are all-zero.
    """
    by_file = collect_annotator_polygons(annotation_data)
    keep = set(filenames) if filenames is not None else None
    out = {}
    for img in annotation_data["images"]:
        fname = img["file_name"]
        if fname in out:
            continue
        if keep is not None and fname not in keep:
            continue
        h, w = img["height"], img["width"]
        recs = by_file.get(fname, [])
        labeled = [r for r in recs if r["anns"]] or recs[:1]
        canvas = np.zeros((h, w), dtype=np.uint8)
        if labeled:
            for a in labeled[0]["anns"]:
                sp = a.get("spine")
                if not sp:
                    continue
                pts = np.array(sp, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(canvas, [pts], False, 255, thickness)
        m = canvas.astype(np.float32) / 255.0
        if blur_sigma > 0:
            m = np.clip(cv2.GaussianBlur(m, (0, 0), blur_sigma), 0.0, 1.0)
        out[fname] = (m * 255.0).astype(np.uint8)
    return out


MAX_ANNOTATORS = 3


def _mask_to_float01(mask):
    if mask.dtype == np.uint8:
        return mask.astype(np.float32) / 255.0
    return np.clip(mask.astype(np.float32), 0.0, 1.0)


def _mask_binary_u8(mask):
    if mask.dtype == np.uint8:
        return (mask > 0).astype(np.uint8)
    return (mask >= 0.5).astype(np.uint8)


def _has_foreground(mask):
    if isinstance(mask, (list, tuple)):
        return any(_has_foreground(m) for m in mask)
    return bool(_mask_binary_u8(mask).any())


class FileScaleAwareDataset(Dataset):
    """Scale-aware crops from file-level union masks, 3-channel pretrained input."""

    def __init__(self, image_dir, filenames, mask_map, crop_size=256, model_size=512,
                 augment=False, pos_bias=0.75, crops_per_image=8, seed=42,
                 neg_filenames=None, neg_ratio=0.0, global_size=None,
                 annotator_map=None, sample_annotator=False, return_all_annotators=False,
                 teacher_map=None, return_teacher=False, spine_map=None, return_spine=False,
                 copy_paste=False, copy_paste_p=0.5):
        self.image_dir = image_dir
        self.filenames = filenames
        self.mask_map = mask_map
        self.crop_size = crop_size
        self.model_size = model_size
        self.augment = augment
        self.pos_bias = pos_bias
        self.crops_per_image = crops_per_image
        self.rng = random.Random(seed)
        self.neg_filenames = neg_filenames or []
        self.neg_ratio = neg_ratio
        self.copy_paste = copy_paste
        self.copy_paste_p = copy_paste_p
        self.global_size = global_size
        self.annotator_map = annotator_map
        self.sample_annotator = sample_annotator
        self.return_all_annotators = return_all_annotators
        self.teacher_map = teacher_map
        self.return_teacher = return_teacher and teacher_map is not None
        self.spine_map = spine_map
        self.return_spine = return_spine and spine_map is not None

        sample = next(iter(mask_map.values())) if mask_map else np.zeros((1, 1), np.uint8)
        self.soft_mask = isinstance(sample, str) or (
            not isinstance(sample, list) and getattr(sample, "dtype", None) != np.uint8
        )

        self._bbox_cache = {}
        self._global_cache = {}
        self._teacher_cache = {}
        self._spine_cache = {}

        mask_interp = cv2.INTER_LINEAR if self.soft_mask else cv2.INTER_NEAREST
        geo = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101,
                     mask_interpolation=mask_interp, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ElasticTransform(alpha=120, sigma=50, p=0.3),
            A.GridDistortion(distort_limit=0.3, p=0.2),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
        ]
        if augment:
            if return_all_annotators or self.return_teacher or self.return_spine:
                self.transform = A.ReplayCompose(geo + [ToTensorV2()])
            else:
                self.transform = A.Compose(geo + [ToTensorV2()])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def _load_mask_array(self, fname):
        m = self.mask_map[fname]
        if isinstance(m, str):
            return np.load(m).astype(np.float32)
        return m

    def _resolve_mask(self, fname):
        if self.sample_annotator and self.annotator_map is not None:
            masks = [m for m in self.annotator_map.get(fname, []) if _has_foreground(m)]
            if not masks:
                masks = self.annotator_map.get(fname) or [self._load_mask_array(fname)]
            return masks[self.rng.randrange(len(masks))]
        return self._load_mask_array(fname)

    def _bboxes(self, fname):
        if fname not in self._bbox_cache:
            if self.annotator_map is not None:
                src = np.zeros_like(_mask_binary_u8(self.annotator_map[fname][0]))
                for m in self.annotator_map[fname]:
                    src |= _mask_binary_u8(m)
            else:
                src = self._load_mask_array(fname)
            bin_m = _mask_binary_u8(src)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
            boxes = []
            for lid in range(1, n):
                if stats[lid, cv2.CC_STAT_AREA] < 20:
                    continue
                x, y = stats[lid, cv2.CC_STAT_LEFT], stats[lid, cv2.CC_STAT_TOP]
                w, h = stats[lid, cv2.CC_STAT_WIDTH], stats[lid, cv2.CC_STAT_HEIGHT]
                boxes.append((x, y, w, h))
            self._bbox_cache[fname] = boxes
        return self._bbox_cache[fname]

    def _global_disk(self, fname):
        if fname not in self._global_cache:
            img = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(img, (self.global_size, self.global_size), interpolation=cv2.INTER_AREA)
            g3 = np.repeat(g[..., None], 3, axis=2).astype(np.float32) / 255.0
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            g3 = (torch.from_numpy(g3).permute(2, 0, 1) - mean) / std
            self._global_cache[fname] = g3
        return self._global_cache[fname]

    def _load_teacher(self, fname):
        if fname not in self._teacher_cache:
            t = self.teacher_map[fname]
            if isinstance(t, str):
                t = np.load(t)
            self._teacher_cache[fname] = t
        return self._teacher_cache[fname]

    def _crop_resize_teacher(self, fname, x0, y0, s):
        t = self._load_teacher(fname)
        crop = t[y0:y0 + s, x0:x0 + s].astype(np.float32)
        if s != self.model_size:
            crop = cv2.resize(crop, (self.model_size, self.model_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _crop_resize_spine(self, fname, x0, y0, s):
        if fname not in self._spine_cache:
            self._spine_cache[fname] = self.spine_map[fname]
        sp = self._spine_cache[fname]
        crop = sp[y0:y0 + s, x0:x0 + s].astype(np.float32)
        if s != self.model_size:
            crop = cv2.resize(crop, (self.model_size, self.model_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _copy_paste_crop(self, bg_img, out_size, bg_teacher=None):
        """Paste 1-3 annotated filament patches onto an (empty) background crop.

        Gives the model exposure to "a marked filament in a mostly-empty scene",
        targeting the empty-file FPs without the risk profile of pseudo-labeling.
        Returns (composited image, binary mask[, teacher]) at out_size; the teacher
        is composited from the source files' ensemble maps so KD stays aligned.
        """
        bg = bg_img.copy()
        bg_mask = np.zeros((out_size, out_size), dtype=np.uint8)
        bg_t = bg_teacher.copy() if bg_teacher is not None else None
        n = self.rng.randint(1, 3)
        for _ in range(n):
            fname = self.rng.choice(self.filenames)
            boxes = self._bboxes(fname)
            if not boxes:
                continue
            bx, by, bw, bh = self.rng.choice(boxes)
            img = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_GRAYSCALE)
            bin_m = _mask_binary_u8(self._load_mask_array(fname))
            ps = int(max(bw, bh) * self.rng.uniform(1.2, 1.8))
            ps = int(np.clip(ps, 48, out_size))
            cx, cy = bx + bw / 2, by + bh / 2
            x0 = int(np.clip(cx - ps / 2, 0, img.shape[1] - ps))
            y0 = int(np.clip(cy - ps / 2, 0, img.shape[0] - ps))
            pimg = img[y0:y0 + ps, x0:x0 + ps]
            pmask = bin_m[y0:y0 + ps, x0:x0 + ps]
            pt = None
            if bg_t is not None:
                pt = self._load_teacher(fname)[y0:y0 + ps, x0:x0 + ps].astype(np.float32)
            if pimg.shape != (ps, ps) or not pmask.any():
                continue
            if self.rng.random() < 0.5:
                pimg, pmask = pimg[:, ::-1], pmask[:, ::-1]
                if pt is not None:
                    pt = pt[:, ::-1]
            if self.rng.random() < 0.5:
                pimg, pmask = pimg[::-1, :], pmask[::-1, :]
                if pt is not None:
                    pt = pt[::-1, :]
            py = self.rng.randint(0, out_size - ps)
            px = self.rng.randint(0, out_size - ps)
            region = bg[py:py + ps, px:px + ps]
            m = pmask > 0
            region[m] = pimg[m]
            bg[py:py + ps, px:px + ps] = region
            bg_mask[py:py + ps, px:px + ps] = np.maximum(
                bg_mask[py:py + ps, px:px + ps], pmask
            )
            if pt is not None:
                region_t = bg_t[py:py + ps, px:px + ps]
                region_t[m] = pt[m]
                bg_t[py:py + ps, px:px + ps] = region_t
        if bg_t is not None:
            return bg, bg_mask, bg_t
        return bg, bg_mask

    def __len__(self):
        return len(self.filenames) * self.crops_per_image

    def __getitem__(self, idx):
        fi = idx // self.crops_per_image
        fname = self.filenames[fi]
        image = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = self.mask_map[fname]
        h, w = image.shape
        s = self.crop_size
        boxes = self._bboxes(fname)

        if self.neg_filenames and self.rng.random() < self.neg_ratio:
            fname = self.rng.choice(self.neg_filenames)
            image = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_GRAYSCALE)
            mask = self.mask_map[fname]
            boxes = []

        if self.augment and boxes and self.rng.random() < self.pos_bias:
            bx, by, bw, bh = self.rng.choice(boxes)
            cx = bx + bw / 2 + self.rng.uniform(-s / 4, s / 4)
            cy = by + bh / 2 + self.rng.uniform(-s / 4, s / 4)
            x0 = int(np.clip(cx - s / 2, 0, w - s))
            y0 = int(np.clip(cy - s / 2, 0, h - s))
        else:
            x0 = self.rng.randint(0, max(1, w - s))
            y0 = self.rng.randint(0, max(1, h - s))

        img_crop = image[y0:y0 + s, x0:x0 + s]
        if s != self.model_size:
            img_crop = cv2.resize(img_crop, (self.model_size, self.model_size), interpolation=cv2.INTER_CUBIC)

        cp_mask = None
        cp_teacher = None
        if self.copy_paste and self.neg_filenames and self.rng.random() < self.copy_paste_p:
            bg_fname = self.rng.choice(self.neg_filenames)
            bg = cv2.imread(os.path.join(self.image_dir, bg_fname), cv2.IMREAD_GRAYSCALE)
            bgh, bgw = bg.shape
            bx0 = self.rng.randint(0, max(1, bgw - s))
            by0 = self.rng.randint(0, max(1, bgh - s))
            bg_crop = bg[by0:by0 + s, bx0:bx0 + s]
            if s != self.model_size:
                bg_crop = cv2.resize(bg_crop, (self.model_size, self.model_size), interpolation=cv2.INTER_CUBIC)
            bg_teacher = self._crop_resize_teacher(bg_fname, bx0, by0, s) if self.return_teacher else None
            res = self._copy_paste_crop(bg_crop, self.model_size, bg_teacher)
            if self.return_teacher:
                img_crop, cp_mask, cp_teacher = res
            else:
                img_crop, cp_mask = res

        img_crop = np.repeat(img_crop[..., None], 3, axis=2)

        if self.return_all_annotators and self.annotator_map is not None:
            raw_masks = self.annotator_map.get(fname) or [np.zeros((h, w), dtype=np.uint8)]
            img_tensor, stacked, n_k = self._multi_annotator_item(img_crop, raw_masks, x0, y0, s)
            out = {"image": img_tensor, "mask": stacked[:1], "masks": stacked, "n_masks": n_k}
        else:
            if cp_mask is not None:
                mask_crop = cp_mask
            else:
                mask = self._resolve_mask(fname)
                mask_crop = self._crop_resize_mask(mask, x0, y0, s)
            transformed = self.transform(image=img_crop, mask=mask_crop)
            img_tensor = self._normalize_image(transformed["image"])
            mask_tensor = self._mask_tensor(transformed["mask"])
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            out = {"image": img_tensor, "mask": mask_tensor}
            if self.return_teacher:
                teacher_crop = cp_teacher if cp_teacher is not None else self._crop_resize_teacher(fname, x0, y0, s)
                if self.augment:
                    r2 = A.ReplayCompose.replay(transformed["replay"], image=img_crop, mask=teacher_crop)
                    teacher_tensor = self._mask_tensor(r2["mask"])
                else:
                    teacher_tensor = self._mask_tensor(teacher_crop)
                if teacher_tensor.ndim == 2:
                    teacher_tensor = teacher_tensor.unsqueeze(0)
                out["teacher"] = teacher_tensor
            if self.return_spine:
                spine_crop = self._crop_resize_spine(fname, x0, y0, s)
                if self.augment:
                    r3 = A.ReplayCompose.replay(transformed["replay"], image=img_crop, mask=spine_crop)
                    spine_tensor = self._mask_tensor(r3["mask"])
                else:
                    spine_tensor = self._mask_tensor(spine_crop)
                if spine_tensor.ndim == 2:
                    spine_tensor = spine_tensor.unsqueeze(0)
                out["spine"] = spine_tensor
        if self.global_size:
            out["global"] = self._global_disk(fname)
        return out

    def _crop_resize_mask(self, mask, x0, y0, s):
        crop = mask[y0:y0 + s, x0:x0 + s]
        if s != self.model_size:
            interp = cv2.INTER_LINEAR if (self.soft_mask or crop.dtype != np.uint8) else cv2.INTER_NEAREST
            crop = cv2.resize(crop, (self.model_size, self.model_size), interpolation=interp)
        return crop

    def _mask_tensor(self, t):
        if not torch.is_tensor(t):
            t = torch.from_numpy(np.ascontiguousarray(t))
        t = t.float()
        if t.numel() and t.max() > 1.0:
            t = t / 255.0
        return t.clamp(0, 1)

    def _normalize_image(self, img_t):
        img_tensor = img_t.float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (img_tensor - mean) / std

    def _multi_annotator_item(self, img_crop, raw_masks, x0, y0, s):
        crops = [self._crop_resize_mask(m, x0, y0, s) for m in raw_masks[:MAX_ANNOTATORS]]
        n_k = len(crops)
        while len(crops) < MAX_ANNOTATORS:
            crops.append(np.zeros_like(crops[0]))
        if self.augment:
            transformed = self.transform(image=img_crop, mask=crops[0])
            img_tensor = self._normalize_image(transformed["image"])
            tensors = [self._mask_tensor(transformed["mask"])]
            replay = transformed.get("replay")
            if replay is not None:
                for extra in crops[1:n_k]:
                    r = A.ReplayCompose.replay(replay, image=img_crop, mask=extra)
                    tensors.append(self._mask_tensor(r["mask"]))
            else:
                for extra in crops[1:n_k]:
                    tensors.append(self._mask_tensor(self.transform(image=img_crop, mask=extra)["mask"]))
            for _ in range(MAX_ANNOTATORS - n_k):
                tensors.append(torch.zeros_like(tensors[0]))
        else:
            transformed = self.transform(image=img_crop, mask=crops[0])
            img_tensor = self._normalize_image(transformed["image"])
            tensors = [self._mask_tensor(self.transform(image=img_crop, mask=c)["mask"]) for c in crops]
        stacked = torch.stack([t if t.ndim == 2 else t.squeeze(0) for t in tensors], dim=0)
        return img_tensor, stacked, n_k


def _entry_has_fg(mask_or_path):
    if isinstance(mask_or_path, str):
        m = np.load(mask_or_path)
        return bool((np.asarray(m) >= 0.5).any())
    return _has_foreground(mask_or_path)


def get_file_dataloaders(config, train_files, val_files, mask_map, neg_ratio=0.0, global_size=None,
                         annotator_map=None, sample_annotator=False, return_all_annotators=False,
                         teacher_map=None, return_teacher=False, spine_map=None, return_spine=False,
                         copy_paste=False, copy_paste_p=0.5):
    def is_pos(f):
        if annotator_map is not None:
            return any(_has_foreground(m) for m in annotator_map.get(f, []))
        return _entry_has_fg(mask_map[f])

    pos_files = [f for f in train_files if is_pos(f)]
    neg_files = [f for f in train_files if not is_pos(f)]
    extra = dict(
        annotator_map=annotator_map,
        sample_annotator=sample_annotator,
        return_all_annotators=return_all_annotators,
    )
    train_dataset = FileScaleAwareDataset(
        image_dir=config.TRAIN_IMG_DIR,
        filenames=pos_files,
        mask_map=mask_map,
        crop_size=config.CROP_SIZE,
        model_size=config.MODEL_INPUT_SIZE,
        augment=True,
        pos_bias=config.POS_BIAS,
        crops_per_image=config.CROPS_PER_IMAGE,
        seed=config.SEED,
        neg_filenames=neg_files,
        neg_ratio=neg_ratio,
        global_size=global_size,
        teacher_map=teacher_map,
        return_teacher=return_teacher,
        spine_map=spine_map,
        return_spine=return_spine,
        copy_paste=copy_paste,
        copy_paste_p=copy_paste_p,
        **extra,
    )
    val_dataset = FileScaleAwareDataset(
        image_dir=config.TRAIN_IMG_DIR,
        filenames=val_files,
        mask_map=mask_map,
        crop_size=config.CROP_SIZE,
        model_size=config.MODEL_INPUT_SIZE,
        augment=False,
        pos_bias=0.0,
        crops_per_image=8,
        seed=config.SEED + 1,
        global_size=global_size,
        **extra,
    )
    # 2026-11-03: with 63 GB RAM the per-worker copy of the annotator map is
    # affordable and the serial path is 2-3x slower per epoch. Respect the
    # explicit override; fall back to the config value.
    n_workers = config.NUM_WORKERS
    loader_kwargs = dict(
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )
    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
