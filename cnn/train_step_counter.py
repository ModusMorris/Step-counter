import torch.nn as nn


class StepCounterCNN(nn.Module):
    def __init__(self, window_size, num_features):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        # After 2x pooling => window_size / 4
        self.fc1 = nn.Linear((window_size // 4) * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, window_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x shape: (Batch, 6, window_size)
        """
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
