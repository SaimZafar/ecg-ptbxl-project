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

### Experiment: Class weighting for HYP imbalance — 2026-06-21
- Hypothesis: HYP underperforms (0.9019 AUC) due to being the rarest class
  (12.2% of training data vs 22-44% for others)
- Applied inverse-frequency pos_weight to BCEWithLogitsLoss
- Result: macro-AUC dropped slightly (0.9264 -> 0.9233)
- HYP AUC barely moved (0.9019 -> 0.9011)
- MI and CD AUC slightly worsened (likely overcorrection on rare class
  hurt the more common ones)

**Conclusion:** Simple inverse-frequency weighting does not help. HYP's
weakness appears to be more about genuine signal-detection difficulty
than pure data scarcity. Reverting to baseline v1 as the reference model.
Next: explore explainability (Grad-CAM equivalent for 1D signals) or
architecture changes as the primary research angle, rather than further
imbalance tuning.

### Explainability: Grad-CAM for 1D ECG signals — 2026-06-21
- Adapted Grad-CAM (originally for images) to work on 1D ECG time-series
- Modified ECGNet.forward() to optionally return intermediate conv features
- Implemented grad_cam_1d() in explain.py: computes per-timestep importance
  by combining gradients and activations from the last conv block
- Tested on 4 validation samples (NORM and MI cases)

**Finding:** The model consistently focuses on QRS complexes (the sharp
heartbeat spikes) rather than the flatter segments between beats. This
matches clinical intuition, since QRS/ST-segment features are exactly
where diagnostic signals like MI typically show up.

**Secondary observation:** For NORM predictions, importance appears spread
fairly evenly across most beats (consistent with judging overall rhythm
regularity). For MI predictions, importance appears more concentrated on
specific individual beats (consistent with localized abnormalities driving
the diagnosis). This is a preliminary observation from a small sample,
worth validating systematically (e.g. averaging importance patterns across
many samples per class).

Next: test whether this pattern holds for weaker-performing classes (CD, HYP).

### Explainability validation across all 5 classes — 2026-06-21
- Tested Grad-CAM on CD and HYP samples (previously only tested NORM/MI)
- QRS-focusing pattern holds across all 5 classes — confirms this is a
  general property of the model, not specific to one class

**Refined observation:** Attention "style" differs subtly by class:
- CD: tight, sharp focus directly on QRS spike (consistent with CD being
  about QRS shape/timing — e.g. bundle branch blocks)
- HYP: slightly broader focus extending toward the T-wave region
  (consistent with HYP often showing amplitude changes beyond just the
  QRS spike)

This is a preliminary pattern from single examples per class — would need
averaging across many samples per class to confirm rigorously, but it's
a promising thread for the explainability angle of the paper.

### Final test-set evaluation (held-out fold 10) — 2026-06-21
Reproducible seeded run (RANDOM_SEED=42). Best checkpoint by val loss,
evaluated ONCE on the held-out test set (never used for tuning).

**Test macro-AUC: 0.9223**

| Class | Test AUC |
|-------|----------|
| NORM  | 0.9442 |
| MI    | 0.9242 |
| STTC  | 0.9327 |
| CD    | 0.9099 |
| HYP   | 0.9006 |

Validation macro-AUC was 0.9231 — the val/test gap is <0.001, indicating
the model generalizes well and is not overfit to the validation set.
This is the headline number for the paper.