"""Train a pretrained ConvNeXt U-Net with a leak-free file split.

Checkpoint selection uses the official multi-annotator all-pairs PQ
(eval_pq_v2.exact_pq), not greedy first-annotator matching.
"""
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from dataset import (
    build_annotator_mask_map,
    build_file_mask_map,
    build_file_spine_map,
    create_file_splits,
    get_file_dataloaders,
    get_inference_dataloader,
    load_and_parse_annotations,
)
from eval_pq_v2 import build_multi_gt, exact_pq, instances_to_rles
from eval_pq import build_gt_maps
from inference import stitch_batch
from model import EMA
from model_v2 import get_convnext_unet, get_loss_function, kd_mse_loss, kd_kl_loss, spine_heatmap_loss
from model_edge import get_edgeatt_unet
from preprocessing import label_instances, component_stats, keep_mask_from_stats, kept_area_rles


def build_val_gt(data, val_files):
    """First-annotator GT label maps (legacy greedy evaluator / classifiers)."""
    val_set = set(val_files)
    ann_ids = {a["image_id"] for a in data["annotations"]}
    name_to_first = {}
    for img in data["images"]:
        fname = img["file_name"]
        if fname in val_set and img["id"] in ann_ids and fname not in name_to_first:
            name_to_first[fname] = img["id"]
    first_ids = [name_to_first[f] for f in val_files if f in name_to_first]
    maps = build_gt_maps(data, first_ids) if first_ids else {}
    for f in val_files:
        stem = os.path.splitext(f)[0]
        if stem not in maps:
            img0 = next(img for img in data["images"] if img["file_name"] == f)
            maps[stem] = (np.zeros((img0["height"], img0["width"]), dtype=np.int32), [])
    return maps


@torch.no_grad()
def compute_val_pq(model, config, val_files, multi_gt, threshold, min_area, close,
                   t_hi=0.0, min_meanp=0.0):
    loader, _ = get_inference_dataloader(config, config.TRAIN_IMG_DIR, filenames=val_files)
    model.eval()
    pred_rles_by_stem = {}
    for batch in loader:
        full_prob, filename, _, _ = stitch_batch(batch, model, config)
        stem = os.path.splitext(filename)[0]
        labels, stats, area, meanp, pmax = component_stats(full_prob, threshold, close)
        keep = (keep_mask_from_stats(area, meanp, pmax, t_hi, min_area, min_meanp)
                if area.size else np.zeros(0, bool))
        pred_rles_by_stem[stem] = kept_area_rles(labels, stats, keep, min_area=min_area)
    return exact_pq(multi_gt, pred_rles_by_stem)


def recalibrate_bn(model, loader, config, n_batches=120):
    """Re-estimate BatchNorm running stats after weight averaging (SWA).

    Averaging weights leaves the decoder's BN statistics mismatched; a short
    forward-only pass in train mode with reset stats fixes it.
    """
    import itertools
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None           # cumulative moving average
    model.train()
    with torch.no_grad():
        for batch in itertools.islice(loader, n_batches):
            model(batch["image"].to(config.DEVICE, non_blocking=True))
    model.eval()


def pq_sparse_from_counts(sum_iou, tp, fp, fn, r):
    """Expected PQ if each labelled filament survives to the test set with prob r.

    The test annotations are sparser than the training ones (r ~ 0.63), so this
    re-weights false positives; selecting the epoch on it is a cheap proxy for
    selecting on the leaderboard.
    """
    denom = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return (r * sum_iou / denom) if denom > 0 else 0.0


def loss_from_batch(criterion, logits, batch, multi_loss):
    if not multi_loss or "masks" not in batch:
        return criterion(logits, batch["mask"].to(logits.device, non_blocking=True))
    masks = batch["masks"].to(logits.device, non_blocking=True)
    n_masks = batch["n_masks"]
    parts = []
    bsz = logits.size(0)
    for b in range(bsz):
        k = int(n_masks[b].item() if torch.is_tensor(n_masks) else n_masks[b])
        k = max(k, 1)
        for i in range(k):
            parts.append(criterion(logits[b:b + 1], masks[b:b + 1, i:i + 1]))
    return torch.stack(parts).mean()


def combined_loss(criterion, out, batch, multi_loss, kd_fn, kd_weight, spine_weight=0.0):
    if spine_weight > 0 and isinstance(out, tuple):
        mask_logits, spine_logits = out
    else:
        mask_logits, spine_logits = out, None
    loss = loss_from_batch(criterion, mask_logits, batch, multi_loss)
    if "teacher" in batch and kd_fn is not None:
        teacher = batch["teacher"].to(mask_logits.device, non_blocking=True)
        loss = loss + kd_weight * kd_fn(mask_logits, teacher)
    if spine_logits is not None and "spine" in batch:
        target = batch["spine"].to(spine_logits.device, non_blocking=True).float()
        loss = loss + spine_weight * spine_heatmap_loss(spine_logits, target)
    return loss


def save_checkpoint(path, model, optimizer, scheduler, scaler, ema, epoch, best_pq):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "ema_shadow": ema.shadow if ema else None,
        "best_pq": best_pq,
    }, path)


def get_metrics():
    def compute_metrics(pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        pred_binary = (pred > 0.5).float()
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        fn = ((1 - pred_binary) * target).sum()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        return {"iou": iou.item(), "dice": dice.item()}
    return compute_metrics


def train(args):
    config = Config
    config.IN_CHANNELS = 3
    config.NORMALIZE = True
    config.LEARNING_RATE = args.lr
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch
    config.CROP_SIZE = args.crop_size
    config.MODEL_INPUT_SIZE = args.model_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    config.create_dirs()
    sf = config.MODEL_INPUT_SIZE / config.CROP_SIZE
    print(f"Device: {config.DEVICE} | encoder={args.encoder} | lr={args.lr} "
          f"| epochs={args.epochs} | batch={args.batch} | cldice_w={args.cldice_w} "
          f"| mask={args.mask_mode} | neg_ratio={args.neg_ratio} "
          f"| crop={args.crop_size} model_size={args.model_size} scale_factor={sf:.3f}")

    data = load_and_parse_annotations(config.ANNOTATION_FILE)
    if args.use_all_data:
        all_files = sorted({img["file_name"] for img in data["images"]})
        train_files = all_files
        val_files = []
        print(f"Train on all data: {len(train_files)} files, no val")
    elif args.fold >= 0:
        from sklearn.model_selection import KFold
        all_files = sorted({img["file_name"] for img in data["images"]})
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=config.SEED)
        splits = list(kf.split(all_files))
        tr_idx, va_idx = splits[args.fold]
        train_files = [all_files[i] for i in tr_idx]
        val_files = [all_files[i] for i in va_idx]
        print(f"Fold {args.fold}/{args.n_folds}: train {len(train_files)}, val {len(val_files)}")
    else:
        train_files, val_files = create_file_splits(data, config.VAL_SPLIT, config.SEED)

    annotator_map = None
    sample_annotator = args.mask_mode == "random_annotator"
    return_all = args.mask_mode == "multi_loss"
    if args.mask_mode in ("random_annotator", "multi_loss"):
        mask_map = build_file_mask_map(data, mode="single")
        annotator_map = build_annotator_mask_map(data)
        print(f"Loaded per-annotator masks for {len(annotator_map)} files")
    else:
        mask_map = build_file_mask_map(data, mode=args.mask_mode)

    teacher_map = None
    if args.teacher_dir:
        teacher_map = {}
        for f in train_files:
            path = os.path.join(args.teacher_dir, os.path.splitext(f)[0] + ".npy")
            if os.path.exists(path):
                teacher_map[f] = path
        missing = len(train_files) - len(teacher_map)
        print(f"Teacher maps: {len(teacher_map)}/{len(train_files)} (missing {missing})")

    spine_map = None
    if args.aux_spine:
        spine_map = build_file_spine_map(data, filenames=train_files)
        n_nz = sum(1 for m in spine_map.values() if m.any())
        print(f"Spine maps: {len(spine_map)} files ({n_nz} non-empty)")

    multi_gt = build_multi_gt(data, val_files) if val_files else {}
    print(f"Val multi-annotator GT: {len(multi_gt)} files")

    train_loader, val_loader = get_file_dataloaders(
        config, train_files, val_files, mask_map, neg_ratio=args.neg_ratio,
        annotator_map=annotator_map, sample_annotator=sample_annotator,
        return_all_annotators=return_all,
        teacher_map=teacher_map, return_teacher=bool(teacher_map),
        spine_map=spine_map, return_spine=bool(spine_map),
        copy_paste=args.copy_paste, copy_paste_p=args.copy_paste_p,
    )

    if args.arch == "edgeatt":
        model = get_edgeatt_unet(config, encoder_name=args.encoder, pretrained=True, use_edge=True)
    elif args.arch == "mhsa":
        model = get_edgeatt_unet(config, encoder_name=args.encoder, pretrained=True, use_edge=False)
    else:
        dc = tuple(int(x) for x in args.decoder_channels.split(",")) if args.decoder_channels else None
        model = get_convnext_unet(config, encoder_name=args.encoder, pretrained=True,
                                  decoder_channels=dc, aux_spine=args.aux_spine,
                                  head_upsample=args.head_upsample)
        if dc:
            print(f"Decoder channels: {dc}")
        if args.aux_spine:
            print(f"Aux spine head enabled (weight={args.spine_weight})")
    criterion = get_loss_function(args.alpha, args.beta, cldice_w=args.cldice_w)
    kd_fn = None
    if teacher_map:
        kd_fn = kd_kl_loss if args.kd_loss == "kl" else kd_mse_loss
        print(f"KD loss: {args.kd_loss}, weight={args.kd_weight}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler("cuda") if config.USE_AMP else None
    warm = max(0, args.warmup_epochs)
    if warm > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, config.NUM_EPOCHS - warm), eta_min=config.ETA_MIN),
            ],
            milestones=[warm],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.NUM_EPOCHS), eta_min=config.ETA_MIN,
        )
    ema = EMA(model, decay=config.EMA_DECAY)

    if args.init_from:
        ck0 = torch.load(args.init_from, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ck0["model_state_dict"])
        print(f"initialised from {args.init_from} (epoch {ck0.get('epoch')}, "
              f"best_pq {ck0.get('best_pq', float('nan')):.4f})")

    swa_state, swa_n = None, 0

    name = args.tag if args.tag else args.encoder
    ckpt_path = os.path.join(config.MODEL_DIR, f"{name}_checkpoint.pth")
    best_path = os.path.join(config.MODEL_DIR, f"{name}_best.pth")
    best_ema_path = os.path.join(config.MODEL_DIR, f"{name}_best_ema.pth")

    best_pq = -1.0
    best_epoch = -1
    best_sparse = -1.0
    sparse_path = os.path.join(config.MODEL_DIR, f"{name}_best_sparse.pth")
    compute_metrics = get_metrics()

    for epoch in range(config.NUM_EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        m = {"iou": 0.0, "dice": 0.0}
        for batch in tqdm(train_loader, desc=f"Train {epoch + 1}"):
            images = batch["image"].to(config.DEVICE)
            masks = batch["mask"].to(config.DEVICE)
            optimizer.zero_grad()
            if config.USE_AMP:
                with autocast("cuda"):
                    out = model(images, return_spine=args.aux_spine)
                    loss = combined_loss(criterion, out, batch, return_all, kd_fn, args.kd_weight,
                                         spine_weight=args.spine_weight if args.aux_spine else 0.0)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images, return_spine=args.aux_spine)
                loss = combined_loss(criterion, out, batch, return_all, kd_fn, args.kd_weight,
                                     spine_weight=args.spine_weight if args.aux_spine else 0.0)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ema.update()
            running_loss += loss.item()
            bm = compute_metrics(out[0] if isinstance(out, tuple) else out, masks)
            for k in m:
                m[k] += bm[k]
        n = len(train_loader)
        train_loss = running_loss / n
        for k in m:
            m[k] /= n

        # pixel-dice val (crops)
        model.eval()
        vloss = 0.0
        vm = {"iou": 0.0, "dice": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(config.DEVICE)
                masks = batch["mask"].to(config.DEVICE)
                out = model(images)
                vloss += loss_from_batch(criterion, out, batch, return_all).item()
                bm = compute_metrics(out, masks)
                for k in vm:
                    vm[k] += bm[k]
        nv = max(1, len(val_loader))
        vloss /= nv
        for k in vm:
            vm[k] /= nv

        scheduler.step()

        # full-image val PQ for checkpoint selection
        if val_files:
            pq, sq, rq, tp, fp, fn = compute_val_pq(
                model, config, val_files, multi_gt,
                threshold=args.pq_thr, min_area=args.pq_area, close=args.pq_close,
                t_hi=args.pq_t_hi, min_meanp=args.pq_min_meanp,
            )
        else:
            pq, sq, rq, tp, fp, fn = 0.0, 0.0, 0.0, 0, 0, 0

        sum_iou = pq * (tp + 0.5 * fp + 0.5 * fn)
        pq_r = (pq_sparse_from_counts(sum_iou, tp, fp, fn, args.select_sparse_r)
                if args.select_sparse_r > 0 else 0.0)

        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | train_loss {train_loss:.4f} "
              f"train_dice {m['dice']:.4f} | val_loss {vloss:.4f} val_dice {vm['dice']:.4f} | "
              f"val_PQ {pq:.4f} (SQ {sq:.4f} RQ {rq:.4f} TP {tp} FP {fp} FN {fn})"
              + (f" PQ'@r {pq_r:.4f}" if args.select_sparse_r > 0 else "")
              + f" | {elapsed:.0f}s")

        if args.swa_last > 0 and epoch >= config.NUM_EPOCHS - args.swa_last:
            state = model.state_dict()
            if swa_state is None:
                swa_state = {k: (v.detach().clone().float() if v.is_floating_point() else v.clone())
                             for k, v in state.items()}
            else:
                for k, v in state.items():
                    if v.is_floating_point():
                        swa_state[k].mul_(swa_n / (swa_n + 1)).add_(
                            v.detach().float(), alpha=1.0 / (swa_n + 1))
                    else:
                        swa_state[k] = v.clone()
            swa_n += 1
            print(f"  swa: averaged {swa_n} checkpoint(s)")

        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, ema, epoch, best_pq)

        if val_files:
            if pq > best_pq:
                best_pq = pq
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_pq": pq}, best_path)
                ema.apply_shadow()
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_pq": pq}, best_ema_path)
                ema.restore()
                print(f"  -> new best PQ {pq:.4f} at epoch {epoch + 1}")
            if args.select_sparse_r > 0 and pq_r > best_sparse:
                best_sparse = pq_r
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "best_pq": pq, "pq_sparse": pq_r}, sparse_path)
                print(f"  -> new best PQ'@r {pq_r:.4f} at epoch {epoch + 1}")
        else:
            # no val: save the final (post-cosine-decay) model as best
            if epoch == config.NUM_EPOCHS - 1:
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_pq": 0.0}, best_path)
                ema.apply_shadow()
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_pq": 0.0}, best_ema_path)
                ema.restore()
                print(f"  -> saved final (all-data) model at epoch {epoch + 1}")

    if args.swa_last > 0 and swa_state is not None and val_files:
        model.load_state_dict(swa_state)
        recalibrate_bn(model, train_loader, config)
        pq, sq, rq, tp, fp, fn = compute_val_pq(
            model, config, val_files, multi_gt, threshold=args.pq_thr,
            min_area=args.pq_area, close=args.pq_close,
            t_hi=args.pq_t_hi, min_meanp=args.pq_min_meanp)
        swa_path = os.path.join(config.MODEL_DIR, f"{name}_swa.pth")
        torch.save({"epoch": config.NUM_EPOCHS - 1, "model_state_dict": model.state_dict(),
                    "best_pq": pq, "swa_n": swa_n}, swa_path)
        print(f"SWA ({swa_n} ckpts, BN recalibrated): val_PQ {pq:.4f} "
              f"(SQ {sq:.4f} RQ {rq:.4f} TP {tp} FP {fp} FN {fn}) -> {swa_path}")

    print(f"\nDone. Best val PQ {best_pq:.4f} at epoch {best_epoch + 1}")
    print(f"Best model: {best_path}")
    print(f"Best EMA model: {best_ema_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="convnext_tiny")
    p.add_argument("--decoder_channels", default="",
                   help="Comma list, e.g. 128,64,32,16. Empty = smp default (256,128,64,32).")
    p.add_argument("--arch", default="convnext", choices=["convnext", "edgeatt", "mhsa"])
    p.add_argument("--tag", default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--model_size", type=int, default=512)
    p.add_argument("--cldice_w", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--mask_mode", default="single", choices=[
        "union", "single", "majority", "staple_hard", "staple_soft",
        "random_annotator", "multi_loss",
    ])
    p.add_argument("--neg_ratio", type=float, default=0.0)
    p.add_argument("--pq_thr", type=float, default=0.35)
    p.add_argument("--pq_area", type=int, default=500)
    p.add_argument("--pq_close", type=int, default=3)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--use_all_data", action="store_true")
    p.add_argument("--teacher_dir", default="")
    p.add_argument("--kd_weight", type=float, default=5.0)
    p.add_argument("--kd_loss", default="mse", choices=["mse", "kl"])
    p.add_argument("--aux_spine", action="store_true",
                   help="Add an auxiliary MAGFiLO spine-heatmap head (multi-task).")
    p.add_argument("--spine_weight", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override config.NUM_WORKERS (0 avoids per-worker dataset pickling).")
    p.add_argument("--copy_paste", action="store_true",
                   help="Paste annotated filaments onto empty-file backgrounds during training.")
    p.add_argument("--copy_paste_p", type=float, default=0.5)
    p.add_argument("--init_from", default="",
                   help="Checkpoint to fine-tune from (default: ImageNet init).")
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--swa_last", type=int, default=0,
                   help="Average the weights of the last N epochs (0 = off).")
    p.add_argument("--pq_t_hi", type=float, default=0.0,
                   help="Peak-prob acceptance gate used during epoch selection.")
    p.add_argument("--pq_min_meanp", type=float, default=0.0,
                   help="Mean-prob acceptance gate used during epoch selection.")
    p.add_argument("--select_sparse_r", type=float, default=0.0,
                   help="Also save the epoch maximising PQ'(r) (test sparsity model).")
    p.add_argument("--head_upsample", type=int, default=1,
                   help="1 = full-resolution head (default); 2 = last decoder block at "
                        "stride 2 + 2x bilinear logit upsample (-22%% MACs).")
    args = p.parse_args()
    train(args)
