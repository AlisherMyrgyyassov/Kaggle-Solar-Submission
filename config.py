import os
import torch

class Config:
    # Paths (derived from this file's location so the repo is drive-letter agnostic)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Override for a data checkout outside the repo: set MAGFILO_DATA_DIR.
    DATA_DIR = os.environ.get(
        "MAGFILO_DATA_DIR",
        os.path.join(BASE_DIR, "data", "MAGFiLO_1.0_Kaggle_2026"))
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train_images")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "test_images")
    ANNOTATION_FILE = os.path.join(DATA_DIR, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")

    # Image settings
    # NOTE (v1-era): the values below are the UltraUNet v1 defaults and are NOT what
    # the live v2 pipeline uses. `eval_v2_sweep.py` and `submit*.py` set CROP_SIZE /
    # MODEL_INPUT_SIZE / IN_CHANNELS / NORMALIZE per run, so treat these as a fallback
    # rather than as the current configuration.
    ORIGINAL_SIZE = 2048
    CROP_SIZE = 256
    MODEL_INPUT_SIZE = 512
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    NORMALIZE = False

    # Scale-aware crop settings
    CROPS_PER_IMAGE = 8
    POS_BIAS = 0.75

    # Circular mask: radius of solar disk (in pixels)
    SOLAR_DISK_RADIUS = 1000
    SOLAR_DISK_CENTER = (ORIGINAL_SIZE // 2, ORIGINAL_SIZE // 2)
    SOLAR_DISK_ERODE = 12

    # Submission / postprocess (v1-era defaults; single source for v1 inference,
    # generate_submission, convert_submission)
    # The v2 winners are passed on the command line instead — see
    # long-term-memory/postprocess-pq.md:
    #   ensemble : submit_ensemble.py --threshold 0.35 --min_area 500 --close 3
    #   student  : submit.py --tag convnext_atto_1024_kd5_dec128_full \
    #              --threshold 0.25 --min_area 400 --close 0
    INFERENCE_CHECKPOINT = "raw"
    INFERENCE_THRESHOLD = 0.44
    INFERENCE_MIN_AREA = 1000
    INFERENCE_CLOSE_KERNEL = 3

    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    VAL_SPLIT = 0.15
    SEED = 42

    # Model
    ULTRA_N_CHANNELS = 32

    # Mixed precision
    USE_AMP = True

    # Scheduler
    SCHEDULER = "cosine"
    WARMUP_EPOCHS = 3
    ETA_MIN = 1e-6

    # Early stopping
    EARLY_STOPPING_PATIENCE = 10

    # EMA
    EMA_DECAY = 0.999

    # Loss
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.SUBMISSION_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("=" * 50)
        print("Configuration (v1 UltraUNet defaults; v2 scripts override)")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Crop size: {cls.CROP_SIZE} -> Model input: {cls.MODEL_INPUT_SIZE}")
        print(f"Crops per image: {cls.CROPS_PER_IMAGE}, Pos bias: {cls.POS_BIAS}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Weight decay: {cls.WEIGHT_DECAY}")
        print(f"Model: UltraUNet (n_channels={cls.ULTRA_N_CHANNELS})")
        print(f"EMA decay: {cls.EMA_DECAY}")
        print(f"Tversky: alpha={cls.TVERSKY_ALPHA}, beta={cls.TVERSKY_BETA}")
        print(f"Scheduler: {cls.SCHEDULER}")
        print(f"Mixed precision: {cls.USE_AMP}")
        print(f"Inference ckpt: {cls.INFERENCE_CHECKPOINT}")
        print(f"Inference threshold: {cls.INFERENCE_THRESHOLD}")
        print(f"Inference min area: {cls.INFERENCE_MIN_AREA}")
        print(f"Inference close kernel: {cls.INFERENCE_CLOSE_KERNEL}")
        print("=" * 50)
