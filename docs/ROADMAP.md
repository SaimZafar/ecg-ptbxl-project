# ECG-PTBXL Project Roadmap

## Results Log

### Baseline v1 — 2026-06-21
- Full PTB-XL train/val split (official strat_fold: 1-8 train, 9 val)
- Model: 1D-CNN, 3 conv blocks (32→64→128 channels)
- 20 epochs, batch size 32, lr 1e-3, Adam optimizer
- Best checkpoint: epoch 19 (val_loss 0.2764)

**Macro-AUC: 0.9264**

| Class | AUC |
|-------|-----|
| NORM  | 0.9463 |
| MI    | 0.9295 |
| STTC  | 0.9301 |
| CD    | 0.9242 |
| HYP   | 0.9019 |

Notes: HYP is the weakest class. Checking if this is due to class imbalance
or genuine signal-detection difficulty before deciding next steps.