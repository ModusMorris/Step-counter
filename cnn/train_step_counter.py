import torch.nn as nn


class StepCounterCNN(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, padding=2)  # Change 2 → 4 channels
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm1d(64)

        fc1_input_size = (window_size // 4) * 64  
        self.fc1 = nn.Linear(fc1_input_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
