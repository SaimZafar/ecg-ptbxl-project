"""ECG-Scope API: bundled samples + user uploads (single + AdaBN batch)."""
import os, io
import numpy as np
from scipy.signal import resample
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ml import analyze, analyze_batch, CLASSES, ASSETS_DIR, DISCLAIMER

app = FastAPI(title="ECG-Scope API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_d = np.load(os.path.join(ASSETS_DIR, "samples.npz"), allow_pickle=True)
SIGNALS, LABELS = _d["signals"], _d["labels"]
LEAD_II, MAX_BYTES = 1, 8_000_000

def true_label(i):
    return ", ".join(CLASSES[c] for c in range(len(CLASSES)) if LABELS[i][c] == 1)

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES, "disclaimer": DISCLAIMER}

@app.get("/samples")
def samples():
    return [{"index": i, "name": f"Sample {i+1}", "true_label": true_label(i)} for i in range(len(SIGNALS))]

@app.get("/analyze/{index}")
def analyze_sample(index: int):
    if not 0 <= index < len(SIGNALS):
        raise HTTPException(404, "sample not found")
    sig = SIGNALS[index]
    r = analyze(sig); r["trace"] = sig[:, LEAD_II].tolist(); r["true_label"] = true_label(index); r["lead"] = "II"
    return r

def _parse(raw, fname):
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 8 MB).")
    n = (fname or "").lower()
    try:
        if n.endswith(".npy"):
            arr = np.load(io.BytesIO(raw))
        elif n.endswith((".csv", ".txt")):
            t = raw.decode("utf-8", errors="ignore")
            try: arr = np.loadtxt(io.StringIO(t), delimiter=",")
            except ValueError: arr = np.loadtxt(io.StringIO(t), delimiter=",", skiprows=1)
        else:
            raise HTTPException(400, "Unsupported file. Upload a .csv or .npy.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not read file as numeric 12-lead ECG data.")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(400, "Expected a 2D array: timesteps x 12 leads.")
    if arr.shape[1] != 12 and arr.shape[0] == 12:
        arr = arr.T
    if arr.shape[1] != 12:
        raise HTTPException(400, f"Expected 12 leads; got shape {tuple(arr.shape)}.")
    if not np.isfinite(arr).all():
        raise HTTPException(400, "File contains NaN or Inf values.")
    if arr.shape[0] != 1000:
        arr = resample(arr, 1000, axis=0).astype(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)

@app.post("/predict-upload")
async def predict_upload(file: UploadFile = File(...)):
    sig = _parse(await file.read(), file.filename)
    r = analyze(sig); r["trace"] = sig[:, LEAD_II].tolist(); r["lead"] = "II"
    return r

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) < 5:
        raise HTTPException(400, "Batch adaptation needs at least 5 files.")
    sigs = [_parse(await f.read(), f.filename) for f in files]
    results = analyze_batch(sigs)
    for r, s in zip(results, sigs):
        r["trace"] = s[:, LEAD_II].tolist(); r["lead"] = "II"
    return {"adapted": True, "count": len(results), "results": results}