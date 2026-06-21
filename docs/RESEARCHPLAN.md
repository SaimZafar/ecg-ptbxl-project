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
### Phase 2 result — normalization + augmentation (2026-06-21)
Retrained with per-lead z-score normalization + augmentation (noise, lead
dropout, baseline wander, amplitude jitter).

| Eval | v1 | v2 (norm+aug) |
|------|------|------|
| PTB-XL test macro | ~0.920 | 0.9222 |
| Chapman macro     | 0.876  | 0.874 |

Finding: in-domain preserved, but cross-dataset macro UNCHANGED. Per-signal
normalization + augmentation did NOT recover the domain-shift gap, indicating
the gap stems from deeper population + label-definition differences (SCP vs
SNOMED), not superficial amplitude/noise variation. Negative result ->
motivates Phase 3 (multi-source training: include Chapman in training).

### Phase 3 result — multi-source training (PTB-XL + Chapman) (2026-06-21)
Split Chapman 70/10/20; trained on PTB-XL-train + Chapman-train (31,176).

| Eval | v1 (PTB-XL only) | v3 (multi-source) |
|------|------|------|
| PTB-XL test macro  | ~0.920 | 0.9161 |
| Chapman test macro | 0.876  | 0.9345 |

STTC (0.84->0.94) and CD (0.80->0.91) collapses fully recovered, confirming
Phase 2 diagnosis: gap needed target-distribution exposure, not input tricks.
Small in-domain cost (-0.006). CAVEAT: Chapman now in training -> this is
multi-domain FITTING, not generalization to unseen source (MI n=22 = noise).
True generalization claim requires a 3rd held-out dataset (Phase 3b capstone).

### Phase 3b capstone — generalization to UNSEEN Georgia (2026-06-21)
Identical norm+aug pipeline; only difference: v3 also trained on Chapman.
Neither saw Georgia. 7,975 mapped Georgia records (MI n=7 excluded).

| Class | v2 PTB-XL-only | v3 multi-source |
|-------|------|------|
| NORM  | 0.633 | 0.611 |
| STTC  | 0.577 | 0.646 |
| CD    | 0.791 | 0.773 |
| HYP   | 0.254 | 0.342 |
| Macro | 0.564 | 0.593 |

1. Multi-source improves unseen-hospital generalization (+0.029 macro, STTC +0.07).
2. But absolute generalization is POOR (~0.59 vs ~0.92 in-domain): two sources
   are not enough for robust cross-hospital transfer.
3. HYP AUC<0.5 = likely label-definition mismatch for hypertrophy; flagged, not claimed.
Conclusion: robust generalization needs more diverse sources + dedicated DG methods.

### Phase 3c ablation — AdaBN vs multi-source (Georgia, unseen)
| Model | Georgia macro |
|-------|------|
| v2 PTB-XL only, original BN | 0.564 |
| v2 PTB-XL only, + AdaBN     | 0.853 |
| v3 multi-source, + AdaBN    | 0.872 |

AdaBN is the dominant lever (+0.29; most of the gap is BN covariate shift,
recoverable label-free). Multi-source adds a smaller consistent increment on
top (+0.019, helps CD/STTC/NORM not HYP). They stack -> 0.87, near in-domain.
Deployment: adapt BN on target hospital's unlabeled ECGs.