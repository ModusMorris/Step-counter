import torch.nn as nn
import torch.nn.functional as F


class StepCounterCNN(nn.Module):
    def __init__(self, window_size):
        """
        Initializes the StepCounterCNN model.

        Args:
            window_size (int): The size of the input window for the time series data.
        """
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, padding=3)  # Input channels: 2, Output channels: 32
        self.bn1 = nn.BatchNorm1d(32)  # Batch normalization for the first layer
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)  # Max pooling layer

        # Residual blocks
        self.resblock1 = self._make_resblock(32, 64)  # First residual block
        self.resblock2 = self._make_resblock(64, 128, stride=2)  # Second residual block with stride 2

        # Fully Connected Layers
        final_length = window_size // 4  # Calculate the final length after pooling and residual blocks
        self.fc1 = nn.Linear(128 * final_length, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Second fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def _make_resblock(self, in_channels, out_channels, stride=1):
        """
        Creates a residual block with two convolutional layers, batch normalization, and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolutional layer.

        Returns:
            nn.Sequential: A sequential container representing the residual block.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1),  # First convolutional layer
            nn.BatchNorm1d(out_channels),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.Conv1d(out_channels, out_channels, 3, padding=1),  # Second convolutional layer
            nn.BatchNorm1d(out_channels),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.5),  # Dropout for regularization
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, window_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) after applying the sigmoid function.
        """
        # Initial layer
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Apply convolution, batch norm, ReLU, and pooling

        # Residual blocks
        x = self.resblock1(x)  # Apply first residual block
        x = self.resblock2(x)  # Apply second residual block

        # Classification
        x = x.flatten(1)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)  # Apply first fully connected layer
        x = F.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply second fully connected layer
        return self.sigmoid(x)  # Apply sigmoid activation for binary classification
