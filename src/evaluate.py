import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config import CLASSES


def get_predictions(model, loader, device):
    """Run the model on a dataset and collect all predictions + true labels."""
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.sigmoid(outputs)  # convert logits to 0-1 probabilities

            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_outputs), np.concatenate(all_labels)


def compute_macro_auc(y_true, y_pred):
    """
    Macro-AUC: compute AUC separately for each of the 5 classes,
    then average them. This is the standard PTB-XL metric.
    """
    aucs = []
    for i, class_name in enumerate(CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
            print(f"  {class_name}: AUC = {auc:.4f}")
        except ValueError:
            print(f"  {class_name}: skipped (only one class present in this split)")

    macro_auc = np.mean(aucs)
    print(f"Macro-AUC: {macro_auc:.4f}")
    return macro_auc