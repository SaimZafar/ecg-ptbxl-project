import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, RESULTS_DIR
from model import ECGNet


class ECGDataset(Dataset):
    """Wraps signals + labels so PyTorch's DataLoader can batch them."""

    def __init__(self, signals, labels):
        self.signals = signals  # shape: (N, 1000, 12)
        self.labels = labels    # shape: (N, 5)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # tells layers like BatchNorm/Dropout we're training
    total_loss = 0.0

    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)

        optimizer.zero_grad()           # clear old gradients
        outputs = model(signals)        # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # compute new gradients
        optimizer.step()                # update weights

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()  # tells BatchNorm/Dropout we're evaluating, not training
    total_loss = 0.0

    with torch.no_grad():  # no gradients needed, saves memory and time
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)

def compute_pos_weights(labels):
    """
    Calculate per-class weights for imbalanced data.
    Rare classes get a higher weight so mistakes on them cost more.
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    weights = neg_counts / (pos_counts + 1e-6)  # +1e-6 avoids divide-by-zero
    return torch.tensor(weights, dtype=torch.float32)

def run_training(train_signals, train_labels, val_signals, val_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ECGDataset(train_signals, train_labels)
    val_ds = ECGDataset(val_signals, val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGNet().to(device)
    pos_weights = compute_pos_weights(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model.pt")
            print(f"  -> saved new best model (val_loss: {val_loss:.4f})")

    return model