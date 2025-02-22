import torch.nn as nn


class StepCounterCNN(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, padding=2)  # 2 statt 6 Kanäle
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((window_size // 4) * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # Statt window_size eine einzelne Wahrsch.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x