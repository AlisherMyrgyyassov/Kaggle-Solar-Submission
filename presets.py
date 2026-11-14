"""Model and ensemble registry for this competition entry.

Two checkpoints are included in ``weights/``: ``atto_dec128_full_randann_swa``, which
produced the submitted CSV, and ``atto_dec64_full_randann_swa``, the narrow student kept
as a comparison row in the report. Every other tag below is a comparison point too; its
``.pth`` is not in this repository, and nothing in the reproduction path needs it.

One place for (a) how to rebuild each checkpoint, and (b) which checkpoints make
up each candidate ensemble. Used by cache_val_probs.py, keep_sweep.py and
submit_keep.py so the evaluation and the submission can never disagree about
what a preset means.

crop/model_size follow inference (`scale_factor = model_size / crop`), e.g.
small@384 is trained on 384px crops upscaled to a 512px model input.
"""
from model_v2 import ConvNeXtUNet

DEC128 = (128, 64, 32, 16)
DEC64 = (64, 32, 16, 8)

MODEL_SPECS = {
    # --- original ensemble (truncated-GT era) -----------------------------
    "convnext_tiny_1024":        dict(encoder="convnext_tiny",  crop=1024, model_size=1024),
    "convnext_small_1024":       dict(encoder="convnext_small", crop=1024, model_size=1024),
    "convnext_tiny_512":         dict(encoder="convnext_tiny",  crop=512,  model_size=512),
    "convnext_small_384":        dict(encoder="convnext_small", crop=384,  model_size=512),
    # --- complete-GT retrain (2026-09-18/19) ------------------------------
    "convnext_tiny_1024_full":   dict(encoder="convnext_tiny",  crop=1024, model_size=1024),
    "convnext_small_1024_full":  dict(encoder="convnext_small", crop=1024, model_size=1024),
    "convnext_tiny_512_full":    dict(encoder="convnext_tiny",  crop=512,  model_size=512),
    "convnext_small_384_full":   dict(encoder="convnext_small", crop=384,  model_size=512),
    # --- distilled students ----------------------------------------------
    "convnext_atto_1024_kd5":            dict(encoder="convnext_atto", crop=1024, model_size=1024),
    "convnext_atto_1024_kd5_dec128":     dict(encoder="convnext_atto", crop=1024, model_size=1024, decoder_channels=DEC128),
    "convnext_atto_1024_kd5_dec64":      dict(encoder="convnext_atto", crop=1024, model_size=1024, decoder_channels=DEC64),
    "convnext_atto_1024_kd5_full":       dict(encoder="convnext_atto", crop=1024, model_size=1024),
    "convnext_atto_1024_kd5_dec128_full": dict(encoder="convnext_atto", crop=1024, model_size=1024, decoder_channels=DEC128),
    "convnext_atto_1024_kd5_dec64_full": dict(encoder="convnext_atto", crop=1024, model_size=1024, decoder_channels=DEC64),
}

# tag -> val-prob directory template. {pt} = probability-cache tag
# ("raw" = the original 8x1024 patch path, "full2048", "full2048_tta8", ...).
PROB_DIR = {
    "convnext_tiny_1024":        "convnext_tiny_convnext_tiny_1024_file_{pt}",
    "convnext_small_1024":       "convnext_small_convnext_small_1024_file_{pt}",
    "convnext_tiny_512":         "convnext_tiny_convnext_tiny_512_file_{pt}",
    "convnext_small_384":        "convnext_small_convnext_small_384_file_{pt}",
    "convnext_tiny_1024_full":   "convnext_tiny_1024_full_file_{pt}",
    "convnext_small_1024_full":  "convnext_small_1024_full_file_{pt}",
    "convnext_tiny_512_full":    "convnext_tiny_512_full_file_{pt}",
    "convnext_small_384_full":   "convnext_small_384_full_file_{pt}",
    "convnext_atto_1024_kd5":             "convnext_atto_convnext_atto_1024_kd5_file_{pt}",
    "convnext_atto_1024_kd5_dec128":      "convnext_atto_convnext_atto_1024_kd5_dec128_file_{pt}",
    "convnext_atto_1024_kd5_dec64":       "convnext_atto_convnext_atto_1024_kd5_dec64_file_{pt}",
    "convnext_atto_1024_kd5_full":        "convnext_atto_1024_kd5_full_file_{pt}",
    "convnext_atto_1024_kd5_dec128_full": "convnext_atto_1024_kd5_dec128_full_file_{pt}",
    "convnext_atto_1024_kd5_dec64_full":  "convnext_atto_1024_kd5_dec64_full_file_{pt}",
}

# Candidate ensembles: tag -> [(member tag, weight), ...]
PRESETS = {
    # accuracy track
    "ens4_old": [
        ("convnext_tiny_1024", 0.35), ("convnext_small_1024", 0.35),
        ("convnext_tiny_512", 0.15), ("convnext_small_384", 0.15),
    ],
    "ens4_oldfull": [
        ("convnext_tiny_1024", 0.25), ("convnext_small_1024", 0.25),
        ("convnext_tiny_1024_full", 0.25), ("convnext_small_1024_full", 0.25),
    ],
    "ens4_full": [
        ("convnext_tiny_1024_full", 0.35), ("convnext_small_1024_full", 0.35),
        ("convnext_tiny_512_full", 0.15), ("convnext_small_384_full", 0.15),
    ],
    "ens6": [
        ("convnext_tiny_1024", 0.20), ("convnext_small_1024", 0.20),
        ("convnext_tiny_1024_full", 0.20), ("convnext_small_1024_full", 0.20),
        ("convnext_atto_1024_kd5_full", 0.10),
        ("convnext_atto_1024_kd5_dec128_full", 0.10),
    ],
    # efficiency track
    "atto_kd5":        [("convnext_atto_1024_kd5_full", 1.0)],
    "atto_dec128":     [("convnext_atto_1024_kd5_dec128_full", 1.0)],
    "atto_dec64":      [("convnext_atto_1024_kd5_dec64_full", 1.0)],
}


def build_model(tag, pretrained=False):
    """Instantiate the architecture for a tag (weights are loaded by the caller)."""
    import torch  # local import keeps the registry importable without CUDA
    spec = MODEL_SPECS[tag]
    return ConvNeXtUNet(
        encoder_name=spec["encoder"],
        in_channels=spec.get("in_channels", 3),
        classes=1,
        pretrained=pretrained,
        decoder_channels=spec.get("decoder_channels"),
        head_upsample=spec.get("head_upsample", 1),
    ), spec


def load_checkpoint(tag, model_dir, device, half=False):
    """Build a model and load its checkpoint (``ckpt_name`` or ``<tag>_best.pth``)."""
    import os
    import torch
    model, spec = build_model(tag)
    fname = spec.get("ckpt_name", f"{tag}_best.pth")
    path = os.path.join(model_dir, fname)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(device)
    if half:
        model = model.half()
    return model, spec, ck

# --- fine-tuned recall models (2026-11-03 session) ---------------------------
MODEL_SPECS.update({
    "atto_dec128_full_randann":
        dict(encoder="convnext_atto", crop=1024, model_size=1024, decoder_channels=DEC128),
    "atto_dec128_full_randann_swa":
        dict(encoder="convnext_atto", crop=1024, model_size=1024, decoder_channels=DEC128,
             ckpt_name="atto_dec128_full_randann_swa.pth"),
    "convnext_tiny_1024_full_randann":
        dict(encoder="convnext_tiny", crop=1024, model_size=1024),
    "convnext_tiny_1024_full_randann_swa":
        dict(encoder="convnext_tiny", crop=1024, model_size=1024,
             ckpt_name="convnext_tiny_1024_full_randann_swa.pth"),
})
PROB_DIR.update({
    "atto_dec128_full_randann": "atto_dec128_full_randann_file_{pt}",
    "atto_dec128_full_randann_swa": "atto_dec128_full_randann_swa_file_{pt}",
    "convnext_tiny_1024_full_randann": "convnext_tiny_1024_full_randann_file_{pt}",
    "convnext_tiny_1024_full_randann_swa": "convnext_tiny_1024_full_randann_swa_file_{pt}",
})
PRESETS.update({
    "randann_student": [("atto_dec128_full_randann", 1.0)],
    "randann_student_swa": [("atto_dec128_full_randann_swa", 1.0)],
})

PRESETS.update({
    "tiny_full": [("convnext_tiny_1024_full", 1.0)],
    "tiny_randann": [("convnext_tiny_1024_full_randann", 1.0)],
    "tiny_randann_swa": [("convnext_tiny_1024_full_randann_swa", 1.0)],
})

# --- 2026-11-03 (b): 5-stage non-ConvNeXt member + half-resolution head -------
MODEL_SPECS.update({
    "resnet34d_512": dict(encoder="resnet34d", crop=512, model_size=512,
                          decoder_channels=(64, 64, 32, 16, 8)),
    "resnet34d_512_swa": dict(encoder="resnet34d", crop=512, model_size=512,
                              decoder_channels=(64, 64, 32, 16, 8),
                              ckpt_name="resnet34d_512_swa.pth"),
    "atto_dec128_halfhead": dict(encoder="convnext_atto", crop=1024, model_size=1024,
                                 decoder_channels=DEC128, head_upsample=2),
    "atto_dec128_halfhead_swa": dict(encoder="convnext_atto", crop=1024, model_size=1024,
                                     decoder_channels=DEC128, head_upsample=2,
                                     ckpt_name="atto_dec128_halfhead_swa.pth"),
})
PROB_DIR.update({
    "resnet34d_512": "resnet34d_512_file_{pt}",
    "resnet34d_512_swa": "resnet34d_512_swa_file_{pt}",
    "atto_dec128_halfhead": "atto_dec128_halfhead_file_{pt}",
    "atto_dec128_halfhead_swa": "atto_dec128_halfhead_swa_file_{pt}",
})
PRESETS.update({
    "resnet34d_512": [("resnet34d_512", 1.0)],
    "resnet34d_512_swa": [("resnet34d_512_swa", 1.0)],
    "atto_dec128_halfhead": [("atto_dec128_halfhead", 1.0)],
    "atto_dec128_halfhead_swa": [("atto_dec128_halfhead_swa", 1.0)],
})

MODEL_SPECS.update({
    "atto_dec64_full_randann": dict(encoder="convnext_atto", crop=1024, model_size=1024,
                                    decoder_channels=DEC64),
    "atto_dec64_full_randann_swa": dict(encoder="convnext_atto", crop=1024, model_size=1024,
                                        decoder_channels=DEC64,
                                        ckpt_name="atto_dec64_full_randann_swa.pth"),
})
PROB_DIR.update({
    "atto_dec64_full_randann": "atto_dec64_full_randann_file_{pt}",
    "atto_dec64_full_randann_swa": "atto_dec64_full_randann_swa_file_{pt}",
})
PRESETS.update({
    "atto_dec64_randann": [("atto_dec64_full_randann", 1.0)],
    "atto_dec64_randann_swa": [("atto_dec64_full_randann_swa", 1.0)],
})

MODEL_SPECS.update({
    "atto_dec64_halfhead_randann": dict(encoder="convnext_atto", crop=1024, model_size=1024,
                                        decoder_channels=DEC64, head_upsample=2,
                                        ckpt_name="atto_dec64_halfhead_randann_best.pth"),
    "atto_dec64_halfhead_randann_swa": dict(encoder="convnext_atto", crop=1024, model_size=1024,
                                            decoder_channels=DEC64, head_upsample=2,
                                            ckpt_name="atto_dec64_halfhead_randann_swa.pth"),
})
PROB_DIR.update({
    "atto_dec64_halfhead_randann": "atto_dec64_halfhead_randann_file_{pt}",
    "atto_dec64_halfhead_randann_swa": "atto_dec64_halfhead_randann_swa_file_{pt}",
})
PRESETS.update({
    "atto_dec64_halfhead_randann": [("atto_dec64_halfhead_randann", 1.0)],
    "atto_dec64_halfhead_randann_swa": [("atto_dec64_halfhead_randann_swa", 1.0)],
})

# control arm for "is *_old real diversity or just a probability damper?"
PRESETS.update({
    "full_pair": [("convnext_tiny_1024_full", .5), ("convnext_small_1024_full", .5)],
})

# precision-anchor probe (alpha=0.7 Tversky + random_annotator + SWA)
MODEL_SPECS.update({
    "tiny_prec": dict(encoder="convnext_tiny", crop=1024, model_size=1024,
                      ckpt_name="convnext_tiny_1024_prec_best.pth"),
    "tiny_prec_sparse": dict(encoder="convnext_tiny", crop=1024, model_size=1024,
                             ckpt_name="convnext_tiny_1024_prec_best_sparse.pth"),
    "tiny_prec_swa": dict(encoder="convnext_tiny", crop=1024, model_size=1024,
                          ckpt_name="convnext_tiny_1024_prec_swa.pth"),
})
PROB_DIR.update({
    "tiny_prec": "tiny_prec_file_{pt}",
    "tiny_prec_sparse": "tiny_prec_sparse_file_{pt}",
    "tiny_prec_swa": "tiny_prec_swa_file_{pt}",
})
PRESETS.update({
    "tiny_prec": [("tiny_prec", 1.0)],
    "tiny_prec_sparse": [("tiny_prec_sparse", 1.0)],
    "tiny_prec_swa": [("tiny_prec_swa", 1.0)],
})
