# Weights

| File | System | Params | Val PQ | Rule (low/close/peak/area/mean) |
|---|---|---|---|---|
| `atto_dec128_full_randann_swa.pth` | **submitted student**: ConvNeXt-atto, decoder 128/64/32/16, trained on every annotator with distillation and weight averaging | 4.28 M | 0.4591 | `0.35/0/0.99/100/0.85` |
| `atto_dec64_full_randann_swa.pth` | narrow student: same encoder, decoder 64/32/16/8 | 3.75 M | 0.4599 | `0.30/0/0.99/100/0.85` |

SHA-256:

```
3AD0992AEB808F3892223367EAF84A4D1F307424EB5BAEFED31AF87CF783973D  atto_dec128_full_randann_swa.pth
375FFE74F922E71F0D0B4DA8CA95C6E7477901CCAB5AF6921EB5F59639AB5F61  atto_dec64_full_randann_swa.pth
```

Both are loaded by `presets.load_checkpoint`:

```python
from presets import load_checkpoint
model, spec, ck = load_checkpoint("atto_dec128_full_randann_swa", "weights",
                                  device, half=True)
```

The four-model ensemble (171 M, val 0.4773) needs its four 1024-px checkpoints
(~650 MB). It is reported as a comparison only and is not required to reproduce the
submitted CSV.
