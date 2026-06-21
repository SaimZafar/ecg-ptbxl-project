import torch
import torch.nn as nn

from config import NUM_CLASSES


class ECGNet(nn.Module):
    def __init__(self, num_leads=12, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # global average pool collapses the time dimension to 1,
        # regardless of how long the signal was — makes the model robust
        # to small length differences
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x comes in as (batch, time, leads) — Conv1d wants (batch, leads, time)
        x = x.permute(0, 2, 1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.global_pool(x)        # shape: (batch, 128, 1)
        x = x.squeeze(-1)              # shape: (batch, 128)

        return self.classifier(x)      # shape: (batch, num_classes) — raw scores, not probabilities yet