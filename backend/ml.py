"""Self-contained ML core for serving: model architecture, weight loading,
prediction, and Grad-CAM. The architecture here must match best_model.pt.
"""
import os
import numpy as np
import torch
import torch.nn as nn

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "best_model.pt")
DEVICE = torch.device("cpu")  # one signal at a time — CPU is plenty


class ECGNet(nn.Module):
    def __init__(self, num_leads=12, num_classes=len(CLASSES)):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_leads, 32, 7, padding=3), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(2))
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2))
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(), nn.MaxPool1d(2))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes))

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        features = self.conv_block3(x)
        x = self.global_pool(features).squeeze(-1)
        output = self.classifier(x)
        if return_features:
            return output, features
        return output


_model = None

def load_model():
    """Load the trained model once, then cache it for all later requests."""
    global _model
    if _model is None:
        m = ECGNet().to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        _model = m
    return _model


def grad_cam_1d(model, signal, class_idx):
    """Importance curve (0-1) per timestep for the chosen class."""
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    output, features = model(x, return_features=True)
    features.retain_grad()
    model.zero_grad()
    output[0, class_idx].backward()
    grads = features.grad[0]
    acts = features[0].detach()
    weights = grads.mean(dim=1)
    importance = torch.relu((weights.unsqueeze(1) * acts).sum(dim=0)).cpu().numpy()
    importance = np.interp(
        np.linspace(0, len(importance) - 1, signal.shape[0]),
        np.arange(len(importance)), importance)
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


def analyze(signal):
    """Predict + explain one ECG. signal: numpy array (1000, 12)."""
    model = load_model()
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()
    top = int(probs.argmax())
    importance = grad_cam_1d(model, signal, top)
    return {
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "predicted_class": CLASSES[top],
        "predicted_index": top,
        "importance": importance.tolist(),
    }