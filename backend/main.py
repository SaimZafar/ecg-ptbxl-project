"""FastAPI web layer: serves the bundled sample ECGs and runs the model."""
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml import analyze, CLASSES, ASSETS_DIR

app = FastAPI(title="ECG Diagnosis API")

# let the frontend (a different origin) call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your Vercel URL later
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the bundled samples once at startup
_data = np.load(os.path.join(ASSETS_DIR, "samples.npz"), allow_pickle=True)
SIGNALS = _data["signals"]                 # (5, 1000, 12)
LABELS  = _data["labels"]                   # (5, 5)
NAMES   = [str(n) for n in _data["names"]]
LEAD_II = 1                                 # standard display lead


def true_label(i):
    pos = [CLASSES[c] for c in range(len(CLASSES)) if LABELS[i][c] == 1]
    return ", ".join(pos)


@app.get("/")
def root():
    return {"status": "ok", "samples": len(SIGNALS)}


@app.get("/samples")
def list_samples():
    return [
        {"index": i, "name": f"Sample {i+1}", "true_label": true_label(i)}
        for i in range(len(SIGNALS))
    ]


@app.get("/analyze/{index}")
def analyze_sample(index: int):
    if index < 0 or index >= len(SIGNALS):
        raise HTTPException(status_code=404, detail="sample not found")
    signal = SIGNALS[index]
    result = analyze(signal)
    result["trace"] = signal[:, LEAD_II].tolist()   # lead II, for the chart
    result["true_label"] = true_label(index)
    result["lead"] = "II"
    return result