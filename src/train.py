import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, RESULTS_DIR, RANDOM_SEED
from model import ECGNet


def set_seed(seed):
    """Make runs reproducible across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_signal(sig):
    """Per-lead z-score — removes amplitude/offset domain differences."""
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True) + 1e-8
    return (sig - mean) / std


def augment_signal(sig):
    """Random augmentations, training only. sig: (1000, 12), already normalized."""
    sig = sig.copy()
    if random.random() < 0.5:                                   # gaussian noise
        sig = sig + np.random.normal(0, 0.1, sig.shape).astype(np.float32)
    if random.random() < 0.3:                                   # lead dropout
        sig[:, random.randint(0, sig.shape[1] - 1)] = 0.0
    if random.random() < 0.3:                                   # baseline wander
        t = np.linspace(0, 2 * np.pi * random.uniform(0.5, 2.0), sig.shape[0]).astype(np.float32)
        sig = sig + (0.2 * np.sin(t))[:, None]
    if random.random() < 0.3:                                   # amplitude jitter
        sig = sig * random.uniform(0.85, 1.15)
    return sig.astype(np.float32)


class ECGDataset(Dataset):
    """Wraps signals + labels. Always per-lead normalized; augments if augment=True."""

    def __init__(self, signals, labels, augment=False):
        self.signals = signals
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = normalize_signal(self.signals[idx].astype(np.float32))
        if self.augment:
            sig = augment_signal(sig)
        signal = torch.tensor(sig, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


def compute_pos_weights(labels):
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    weights = neg_counts / (pos_counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def run_training(train_signals, train_labels, val_signals, val_labels):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ECGDataset(train_signals, train_labels, augment=True)
    val_ds = ECGDataset(val_signals, val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
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