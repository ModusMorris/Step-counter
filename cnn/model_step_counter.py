import torch.nn as nn
import torch.nn.functional as F

class StepCounterCNN(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        # Everything as before, except the last layer:
        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

        self.resblock1 = self._make_resblock(32, 64)
        self.resblock2 = self._make_resblock(64, 128, stride=2)

        final_length = window_size // 4
        self.fc1 = nn.Linear(128 * final_length, 64)

        # IMPORTANT: instead of 1 now 7 outputs (1 for step + 6 for gait types)
        self.fc2 = nn.Linear(64, 7)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def _make_resblock(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        """
        x shape: (batch_size, 2, window_size)
        we return shape: (batch_size, 7)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.resblock1(x)
        x = self.resblock2(x)

        x = x.flatten(1)  # (batch_size, 128*final_length)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)