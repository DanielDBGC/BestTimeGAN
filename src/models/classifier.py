import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, C, T, n_classes):
        super().__init__()

        F1 = 8   # temporal filters
        D  = 2   # depth multiplier

        # Temporal conv (frequency extraction)
        self.temporal = nn.Conv2d(
            1, F1, (1, 64), padding=(0, 32), bias=False
        )

        # Depthwise spatial conv
        self.spatial = nn.Conv2d(
            F1, F1 * D, (C, 1), groups=F1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(F1 * D)

        self.pool1 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(0.25)

        # Separable conv
        self.sep = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )

        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d((1, 8))

        self.fc = nn.Linear((T // 64) * F1 * D, n_classes)

    def forward(self, x):
        # x: [B, T, C] → [B, 1, C, T]
        x = x.permute(0, 2, 1).unsqueeze(1)

        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.sep(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.flatten(start_dim=1)
        return self.fc(x)
