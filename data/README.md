# Data

The competition data is **not** redistributed here. Download it from the Kaggle
competition (`filament-segmentation-2026`) and place it at the default path:

```
data/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/*.jpeg
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/*.jpeg
```

The inference notebook only needs `test/test_images`. Training and validation also
need `train/train_images` and the annotation JSON.

If the data lives elsewhere, set `MAGFILO_DATA_DIR` before running anything:

```powershell
$env:MAGFILO_DATA_DIR = "D:\data\MAGFiLO_1.0_Kaggle_2026"
```

`config.py` reads that variable and falls back to `data/MAGFiLO_1.0_Kaggle_2026`.
