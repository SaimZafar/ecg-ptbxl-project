# Research Plan — Cross-Dataset Generalization for ECG Diagnosis

**Goal:** Investigate how a 1D-CNN trained on PTB-XL generalizes to ECGs from
other sources, and what makes it robust. The honest framing throughout: this is
a research system, **not** for clinical use.

**Baseline (v1, on `main`):** 1D-CNN, PTB-XL 5 superclasses,
test macro-AUC 0.9204 ± 0.0016 (patient-disjoint, 3 seeds).

## Phase 1 — Quantify domain shift (weeks 1–2)
- [ ] Add a 2nd dataset (Chapman-Shaoxing).
- [ ] Harmonize its labels to the 5 PTB-XL superclasses (NORM/MI/STTC/CD/HYP).
- [ ] Harmonize format (resample to 100 Hz, lead order, 10 s, amplitude).
- [ ] Run the v1 model on it WITHOUT retraining → measure the AUC drop.
- [ ] Result: quantified domain-shift gap.

## Phase 2 — Robustness (weeks 3–5)
- [ ] Per-signal normalization + augmentation (noise, lead dropout, scaling, drift).
- [ ] Retrain on PTB-XL, re-measure cross-dataset.
- [ ] Result: how much robustness recovers the drop.

## Phase 3 — Multi-source training (weeks 6–9)
- [ ] Train on PTB-XL + dataset 2 combined.
- [ ] Test on a 3rd held-out dataset (CPSC / Georgia).
- [ ] Result: core generalization finding.

## Phase 4 — Write-up + product v2 (weeks 10–12)
- [ ] Paper draft (target: applied-ML / health-ML workshop).
- [ ] Fold robust model + input-handling into the deployed app.

## Experiment log
- (entries added per session)