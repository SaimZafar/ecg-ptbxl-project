# ECG-Scope: Interpretable ECG Diagnosis

Upload a 12-lead ECG and get an AI-assisted reading across five diagnostic
categories, with a visual explanation of exactly where the model looked.

**Live demo:** https://ecg-ptbxl-project.vercel.app

**API:** https://saim0112-ecg-scope-api.hf.space/docs

**Research and educational tool only. Not for clinical use.** It is not a medical
device and has not been clinically validated. Do not upload real, identifiable
patient data.

---

## What it is

ECG-Scope classifies 12-lead electrocardiograms into five diagnostic superclasses
(NORM normal, MI myocardial infarction, STTC ST/T change, CD conduction
disturbance, HYP hypertrophy) and explains every prediction with a Grad-CAM
attention overlay on the signal.

It pairs a from-scratch deep learning pipeline with a deployed full-stack product:

- Model: a 1D convolutional neural network trained on PTB-XL.
- Research: a cross-dataset generalization study across three public ECG databases.
- Product: a React and FastAPI web app with authentication, file upload, and saved history.

## Screenshots

Add screenshots of the landing page and the analyze page here.

## Key results

### In-domain (PTB-XL), patient-disjoint splits, 3 seeds

Test macro-AUC: 0.920 plus or minus 0.002

| Class | Test AUC |
|-------|----------|
| NORM  | 0.944 |
| MI    | 0.924 |
| STTC  | 0.933 |
| CD    | 0.910 |
| HYP   | 0.901 |

### Cross-dataset generalization

How well does a PTB-XL-trained model transfer to ECGs from different hospitals?

| Setting | Macro-AUC |
|---------|-----------|
| PTB-XL (in-domain) | 0.920 |
| Chapman-Shaoxing (unseen, no adaptation) | 0.876 |
| Georgia (unseen, no adaptation) | about 0.59 |
| Georgia with test-time BatchNorm adaptation (AdaBN) | 0.87 |

Main finding: most of the apparent cross-hospital generalization gap is BatchNorm
covariate shift, recoverable label-free at test time via AdaBN, with no retraining
and no labels required. Multi-source training adds a smaller margin on top. Naive
input normalization and augmentation did not close the gap.

## Tech stack

- ML: PyTorch, scikit-learn, wfdb, NumPy/SciPy
- Backend: FastAPI on Hugging Face Spaces (Docker)
- Frontend: React (Vite) and React Router on Vercel
- Auth and data: Supabase (Postgres with Row Level Security)

## How it works

1. A compact 1D-CNN (3 conv blocks, global average pool, classifier) on 100 Hz, 12-lead signals.
2. Per-signal normalization and augmentation (noise, lead dropout, baseline wander, amplitude jitter).
3. Grad-CAM adapted to 1D time-series for explainability.
4. Test-time BatchNorm adaptation (AdaBN) for cross-source robustness.
5. FastAPI serves predictions; React renders the trace with the attention overlay.

## Repo structure

```
src/         ML pipeline: data, model, training, evaluation, explainability
notebooks/   Kaggle training notebook
backend/     FastAPI inference API and trained model
frontend/    React web app
docs/        Research plan and experiment log
```

## Datasets

All from PhysioNet, harmonized to a shared 5-superclass SNOMED-CT label scheme:

- PTB-XL, the primary training set
- Chapman-Shaoxing 12-lead ECG
- Georgia 12-lead ECG Challenge Database

## Running locally

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# frontend
cd frontend
npm install
npm run dev
```

The frontend needs a `.env.local` with `VITE_SUPABASE_URL` and `VITE_SUPABASE_KEY`.

## License

MIT for the code. Datasets remain under their respective PhysioNet licenses.