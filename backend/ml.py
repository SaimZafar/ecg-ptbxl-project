"""ML core: architecture, model loading, prediction, Grad-CAM, batch AdaBN."""
import os, copy
import numpy as np
import torch
import torch.nn as nn

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "best_model.pt")
DEVICE = torch.device("cpu")
DISCLAIMER = "Research/educational model — not for clinical use."
NORMALIZE = True   # this model was trained on normalized signals

class ECGNet(nn.Module):
    def __init__(self, num_leads=12, num_classes=len(CLASSES)):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv1d(num_leads,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,num_classes))
    def forward(self, x, return_features=False):
        x = x.permute(0,2,1)
        x = self.conv_block1(x); x = self.conv_block2(x); f = self.conv_block3(x)
        out = self.classifier(self.global_pool(f).squeeze(-1))
        return (out, f) if return_features else out

def _prep(sig):
    sig = np.asarray(sig, dtype=np.float32)
    if NORMALIZE:
        sig = (sig - sig.mean(0, keepdims=True)) / (sig.std(0, keepdims=True) + 1e-8)
    return np.ascontiguousarray(sig, dtype=np.float32)

_model = None
def load_model():
    global _model
    if _model is None:
        m = ECGNet().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        m.eval(); _model = m
    return _model

def _grad_cam(model, sig, class_idx):
    x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
    out, f = model(x, return_features=True)
    f.retain_grad(); model.zero_grad(); out[0, class_idx].backward()
    w = f.grad[0].mean(1)
    imp = torch.relu((w.unsqueeze(1) * f[0].detach()).sum(0)).numpy()
    imp = np.interp(np.linspace(0, len(imp)-1, sig.shape[0]), np.arange(len(imp)), imp)
    return (imp / imp.max()).tolist() if imp.max() > 0 else imp.tolist()

def _predict_one(model, sig):
    x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].numpy()
    top = int(probs.argmax())
    return {
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
        "predicted_class": CLASSES[top],
        "predicted_index": top,
        "importance": _grad_cam(model, sig, top),
        "disclaimer": DISCLAIMER,
    }

def analyze(sig):
    return _predict_one(load_model(), _prep(sig))

def analyze_batch(signals):
    """Adapt BatchNorm stats to the user's batch (AdaBN), then predict each."""
    m = copy.deepcopy(load_model())
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None
    prepped = [_prep(s) for s in signals]
    m.train()
    with torch.no_grad():
        m(torch.tensor(np.stack(prepped), dtype=torch.float32))
    m.eval()
    return [_predict_one(m, s) for s in prepped]