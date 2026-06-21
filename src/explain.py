import numpy as np
import torch

from config import CLASSES


def grad_cam_1d(model, signal, class_idx, device):
    """
    Compute a Grad-CAM importance curve for one ECG signal and one class.

    signal: numpy array, shape (1000, 12) — one ECG
    class_idx: which of the 5 classes to explain (0=NORM, 1=MI, etc.)

    Returns: numpy array, shape (1000,) — importance score per original timestep
    """
    model.eval()

    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)

    output, features = model(x, return_features=True)
    features.retain_grad()  # tell PyTorch to keep this intermediate gradient

    target_score = output[0, class_idx]

    model.zero_grad()
    target_score.backward()

    gradients = features.grad[0]      # shape: (128, time')
    activations = features[0].detach()  # shape: (128, time')

    # Grad-CAM recipe: weight each channel by how much it mattered on average,
    # then collapse the 128 channels into one importance curve
    channel_weights = gradients.mean(dim=1)          # shape: (128,)
    importance = (channel_weights.unsqueeze(1) * activations).sum(dim=0)  # shape: (time',)
    importance = torch.relu(importance)              # only positive influence matters

    # upsample back to original 1000 timesteps (since pooling shrank it)
    importance = importance.cpu().numpy()
    importance_upsampled = np.interp(
        np.linspace(0, len(importance) - 1, 1000),
        np.arange(len(importance)),
        importance
    )

    # normalize to 0-1 for easy visualization
    if importance_upsampled.max() > 0:
        importance_upsampled = importance_upsampled / importance_upsampled.max()

    return importance_upsampled