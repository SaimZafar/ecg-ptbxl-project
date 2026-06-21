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

### Phase 1 result — domain shift to Chapman-Shaoxing (2026-06-21)
Harmonized 19,621 Chapman records (SNOMED -> 5 superclasses; ~43% of 45,152
mapped, rest are rhythm-only with no diagnostic-form equivalent).
Ran the PTB-XL v1 model on Chapman WITHOUT retraining.

| Class | PTB-XL | Chapman | drop | n |
|-------|--------|---------|------|------|
| NORM  | 0.944  | 0.924   | -0.020 | 6991 |
| MI    | 0.924  | 0.885   | -0.039 | 123  |
| STTC  | 0.933  | 0.839   | -0.094 | 10149|
| CD    | 0.910  | 0.805   | -0.105 | 3185 |
| HYP   | 0.901  | 0.930   | +0.029 | 769  |
| **Macro** | **~0.920** | **0.876** | **-0.044** | |

Finding: domain shift is real (~4.4 pts macro) and NON-uniform — STTC and CD
(fine morphology) degrade most; NORM transfers well. MI (n=123) too sparse to
trust. Caveat: drop conflates device/population shift with SCP-vs-SNOMED label
definition differences. Motivates Phase 2 (normalization + augmentation).