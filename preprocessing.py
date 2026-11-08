import numpy as np
import cv2


def create_solar_mask(shape=(2048, 2048), center=(1024, 1024), radius=1000):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def apply_solar_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def generate_patches(image, mask, patch_size=512, stride=None, overlap=64):
    if stride is None:
        stride = patch_size - overlap
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size] if mask is not None else None
            patches.append((img_patch, mask_patch, x, y))
    if (w - patch_size) % stride != 0:
        x_edge = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            img_patch = image[y:y + patch_size, x_edge:x_edge + patch_size]
            mask_patch = mask[y:y + patch_size, x_edge:x_edge + patch_size] if mask is not None else None
            patches.append((img_patch, mask_patch, x_edge, y))
    if (h - patch_size) % stride != 0:
        y_edge = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y_edge:y_edge + patch_size, x:x + patch_size]
            mask_patch = mask[y_edge:y_edge + patch_size, x:x + patch_size] if mask is not None else None
            patches.append((img_patch, mask_patch, x, y_edge))
    return patches


def stitch_patches(patches, predictions, full_h=2048, full_w=2048,
                   patch_size=512, stride=448):
    result = np.zeros((full_h, full_w), dtype=np.float32)
    weight = np.zeros((full_h, full_w), dtype=np.float32)

    for (_, _, x, y), pred in zip(patches, predictions):
        p = pred.squeeze()
        if p.shape != (patch_size, patch_size):
            p = cv2.resize(p, (patch_size, patch_size))
        result[y:y + patch_size, x:x + patch_size] += p
        weight[y:y + patch_size, x:x + patch_size] += 1.0

    weight[weight == 0] = 1.0
    return result / weight


def label_instances(prob, threshold, close_kernel=0):
    """Threshold (+ optional ellipse close) then 8-connected labels.

    Returns (labels, stats) from cv2.connectedComponentsWithStats.
    close_kernel=0 skips morphological closing.
    """
    binary = (prob > threshold).astype(np.uint8)
    if close_kernel and close_kernel > 0:
        k = int(close_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    return labels, stats


def _prepare_binary(prob, threshold, close_kernel):
    binary = (prob > threshold).astype(np.uint8)
    if close_kernel and close_kernel > 0:
        k = int(close_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _count_cores(core_roi, min_core_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        core_roi.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0, None
    keep = (st[:, cv2.CC_STAT_AREA] >= min_core_area).astype(np.int32)
    keep[0] = 0
    n_keep = int(keep.sum())
    if n_keep == 0:
        return 0, None
    remap = np.zeros(n, dtype=np.int32)
    remap[keep > 0] = np.arange(1, n_keep + 1)
    return n_keep, remap[lab]


def _split_blobs(prob, binary, sure_fg, min_core_area):
    """8-CC, then watershed only blobs that contain 2+ cores. Thin filaments stay intact."""
    n, bin_lab, bin_st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros(binary.shape, dtype=np.int32)
    out = np.zeros(binary.shape, dtype=np.int32)
    next_id = 1
    for lid in range(1, n):
        x = int(bin_st[lid, cv2.CC_STAT_LEFT])
        y = int(bin_st[lid, cv2.CC_STAT_TOP])
        w = int(bin_st[lid, cv2.CC_STAT_WIDTH])
        h = int(bin_st[lid, cv2.CC_STAT_HEIGHT])
        roi_bin = (bin_lab[y:y + h, x:x + w] == lid).astype(np.uint8)
        roi_fg = sure_fg[y:y + h, x:x + w] & roi_bin
        n_keep, markers = _count_cores(roi_fg, min_core_area)
        if n_keep <= 1:
            out[y:y + h, x:x + w][roi_bin > 0] = next_id
            next_id += 1
            continue
        unknown = (roi_bin > 0) & (markers == 0)
        markers = markers + 1
        markers[unknown] = 0
        roi_prob = np.clip(prob[y:y + h, x:x + w], 0, 1)
        ws_in = cv2.cvtColor((roi_prob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        split = cv2.watershed(ws_in, markers.astype(np.int32))
        split = np.where((split > 1) & (roi_bin > 0), split - 1, 0)
        # any binary pixel left unlabeled keeps the first fragment (don't drop)
        leftover = (roi_bin > 0) & (split == 0)
        if leftover.any():
            split[leftover] = 1
        out[y:y + h, x:x + w][roi_bin > 0] = split[roi_bin > 0] + (next_id - 1)
        next_id += int(split.max())
    return out


def label_instances_watershed(prob, threshold, close_kernel=0, min_peak=0.12):
    """Split touching filaments via watershed from high-confidence cores.

    Cores are pixels with prob > threshold + min_peak (default +0.12). A
    contact neck that dips below the core threshold becomes a watershed cut.
    Blobs with fewer than two cores stay as a single 8-CC instance.
    """
    binary = _prepare_binary(prob, threshold, close_kernel)
    if binary.sum() == 0:
        return np.zeros(prob.shape, dtype=np.int32), np.zeros((1, 5), dtype=np.int32)
    core_t = min(0.95, float(threshold) + float(min_peak))
    sure_fg = ((prob > core_t) & (binary > 0)).astype(np.uint8)
    labels = _split_blobs(prob, binary, sure_fg, min_core_area=40)
    return labels, _labels_to_stats(labels)


def _labels_to_stats(labels):
    n = int(labels.max()) + 1
    if n <= 1:
        return np.zeros((1, 5), dtype=np.int32)
    ys, xs = np.nonzero(labels)
    lids = labels[ys, xs]
    stats = np.zeros((n, 5), dtype=np.int32)
    stats[:, cv2.CC_STAT_AREA] = np.bincount(lids, minlength=n)
    order = np.argsort(lids, kind="mergesort")
    lids_s, xs_s, ys_s = lids[order], xs[order], ys[order]
    starts = np.r_[0, np.flatnonzero(lids_s[1:] != lids_s[:-1]) + 1]
    for i, st in enumerate(starts):
        lid = int(lids_s[st])
        en = starts[i + 1] if i + 1 < len(starts) else len(lids_s)
        stats[lid, cv2.CC_STAT_LEFT] = int(xs_s[st:en].min())
        stats[lid, cv2.CC_STAT_TOP] = int(ys_s[st:en].min())
        stats[lid, cv2.CC_STAT_WIDTH] = int(xs_s[st:en].max() - stats[lid, cv2.CC_STAT_LEFT] + 1)
        stats[lid, cv2.CC_STAT_HEIGHT] = int(ys_s[st:en].max() - stats[lid, cv2.CC_STAT_TOP] + 1)
    return stats


def _skeleton(binary):
    from skimage.morphology import skeletonize
    return skeletonize(binary.astype(bool)).astype(np.uint8)


def label_instances_skeleton(prob, threshold, close_kernel=0, min_peak=0.12):
    """Watershed seeded by the high-prob skeleton (proxy for MAGFiLO spines).

    The medial axis is cut at probability saddles (pixels below
    threshold+min_peak), so two filaments that touch at a faint neck become
    two markers. A uniformly confident filament stays one instance.
    """
    binary = _prepare_binary(prob, threshold, close_kernel)
    if binary.sum() == 0:
        return np.zeros(prob.shape, dtype=np.int32), np.zeros((1, 5), dtype=np.int32)
    skel = _skeleton(binary)
    core_t = min(0.95, float(threshold) + float(min_peak))
    sure_fg = ((skel > 0) & (prob > core_t)).astype(np.uint8)
    labels = _split_blobs(prob, binary, sure_fg, min_core_area=8)
    return labels, _labels_to_stats(labels)


def filter_labeled_instances(labels, stats, min_area):
    """Keep connected components with area >= min_area (label 0 is background)."""
    instances = []
    for lid in range(1, stats.shape[0]):
        if stats[lid, cv2.CC_STAT_AREA] >= min_area:
            instances.append((labels == lid).astype(np.uint8))
    return instances


# ---------------------------------------------------------------------------
# Decoupled "KEEP" instance rule (2026-11-03)
#
# The 1-D sweep (threshold, min_area) couples two conflicting goals: a low
# threshold + small min_area is needed to keep thin barbs and small filaments,
# but it also admits every lukewarm blob. The decoupled rule separates them:
#
#   contour   : 8-connected region of prob > t_lo      (segmentation / SQ)
#   acceptance: peak prob > t_hi AND mean prob >= min_meanp AND area >= min_area
#
# so boundaries stay permissive while instance acceptance is strict. On the
# 107-file val split this is +0.0158 nested-CV PQ (95% CI [+0.0087, +0.0231],
# p < 1e-5) over the old (t, min_area) family, mostly by removing ~27% of FPs
# while keeping TP. Independently reproduced by the public 0.39-LB notebook.
# See long-term-memory/session-2026-11-03.md.
# ---------------------------------------------------------------------------

def component_stats(prob, t_lo, close_kernel=0):
    """8-connected components of (prob > t_lo) plus per-component statistics.

    Returns (labels, stats, area, meanp, pmax) where the last three are indexed
    by label-1 (i.e. component ``lid`` -> index ``lid - 1``).
    """
    binary = _prepare_binary(prob, t_lo, close_kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    n_comp = n - 1
    if n_comp <= 0:
        return labels, stats, np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32)

    flat = labels.ravel()
    nz = np.flatnonzero(flat)
    lv = flat[nz]
    pv = prob.ravel()[nz]
    area = np.bincount(lv, minlength=n)[1:].astype(np.int64)
    psum = np.bincount(lv, weights=pv, minlength=n)[1:]
    pmax = np.zeros(n, dtype=np.float32)
    np.maximum.at(pmax, lv, pv)
    meanp = (psum / np.maximum(area, 1)).astype(np.float32)
    return labels, stats, area, meanp, pmax[1:]


def keep_mask_from_stats(area, meanp, pmax, t_hi=0.0, min_area=0, min_meanp=0.0):
    """Boolean per-component acceptance mask for the decoupled rule."""
    return (pmax > t_hi) & (area >= min_area) & (meanp >= min_meanp)


def kept_area_rles(labels, stats, keep, min_area=0):
    """RLE-encode the components whose ``keep`` flag is True.

    Only the bbox of each component is written into a scratch buffer, so this
    costs O(sum of bbox areas) rather than O(n_instances * H * W).
    """
    from pycocotools import mask as mask_utils

    h, w = labels.shape
    buf = np.zeros((h, w), dtype=np.uint8, order="F")
    out = []
    for idx in np.flatnonzero(keep):
        lid = int(idx) + 1
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lid, cv2.CC_STAT_LEFT])
        y = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        buf[y:y + bh, x:x + bw] = (labels[y:y + bh, x:x + bw] == lid)
        out.append(mask_utils.encode(buf))
        buf[y:y + bh, x:x + bw] = 0
    return out


def keep_instance_rles(prob, t_lo=0.35, close_kernel=0, t_hi=0.0, min_area=300,
                       min_meanp=0.0):
    """Full decoupled post-process: contour at t_lo, accept by (t_hi, area, meanp)."""
    labels, stats, area, meanp, pmax = component_stats(prob, t_lo, close_kernel)
    if area.size == 0:
        return []
    keep = keep_mask_from_stats(area, meanp, pmax, t_hi, min_area, min_meanp)
    return kept_area_rles(labels, stats, keep, min_area=min_area)
