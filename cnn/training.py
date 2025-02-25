import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_generator import load_datasets
from model_step_counter import StepCounterCNN
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from prediction import main as prediction


def load_all_datasets(root_folder, window_size, batch_size):
    """
    Loads all datasets from subfolders in the given root folder.

    Parameters:
    root_folder (str): Path to the root directory containing dataset folders.
    window_size (int): Number of samples per window for the model.
    batch_size (int): Number of samples per batch.

    Returns:
    ConcatDataset: Combined dataset from all subfolders.
    """
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print("No folders found in", root_folder)
        return None

    # Load datasets from each subfolder
    all_data_loaders = [
        load_datasets(sf, window_size, batch_size).dataset
        for sf in subfolders
        if load_datasets(sf, window_size, batch_size) is not None
    ]
    if not all_data_loaders:
        print("No datasets available!")
        return None

    # Combine all datasets into one
    combined = ConcatDataset(all_data_loaders)
    print(f"{len(all_data_loaders)} datasets, total: {len(combined)} samples.")
    return combined


def split_dataset(dataset, ratio=0.2):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    dataset (ConcatDataset): The dataset to be split.
    ratio (float): The proportion of data to be used for testing. Default is 0.2 (20%).

    Returns:
    Tuple[Subset, Subset]: Training and testing dataset subsets.
    """
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=ratio, random_state=42)
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.005, path="best_model.pth"):
        """
        Initializes the EarlyStopping mechanism.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum improvement in validation loss to be considered significant.
            path (str): File path where the best model will be saved.
        """
        self.patience = patience  # Number of epochs with no improvement before stopping
        self.min_delta = min_delta  # Minimum required change in validation loss
        self.path = path  # Path to save the best model
        self.best_loss = float("inf")  # Initialize best validation loss as infinity
        self.counter = 0  # Counter to track epochs without improvement
        self.best_epoch = 0  # Stores the epoch with the best validation loss
        self.best_train_loss = float("inf")  # Stores the training loss of the best model

    def check(self, train_loss, val_loss, model, epoch):
        """
        Checks whether training should stop early based on validation loss.

        Args:
            train_loss (float): Current training loss.
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The PyTorch model being trained.
            epoch (int): The current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """

        # If the validation loss improves significantly, save the model
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss  # Update best validation loss
            self.best_train_loss = train_loss  # Store corresponding training loss
            self.best_epoch = epoch  # Store epoch number of the best model
            self.counter = 0  # Reset counter since there was an improvement
            torch.save(model.state_dict(), self.path)  # Save the best model checkpoint

        # Check if overfitting occurs (training loss is much lower than validation loss)
        elif abs(train_loss - val_loss) > self.min_delta and val_loss >= self.best_loss:
            print("Early stopping triggered due to overfitting!")  # Print warning
            model.load_state_dict(torch.load(self.path))  # Load the best saved model
            return True  # Stop training

        else:
            # No significant improvement, increment counter
            self.counter += 1

            # If patience limit is reached, stop training
            if self.counter >= self.patience:
                print(f"Early stopping triggered! Best model from epoch {self.best_epoch + 1} loaded from {self.path}")
                model.load_state_dict(torch.load(self.path))  # Load the best model
                return True  # Stop training

        return False  # Continue training


def train_step_counter(root_folder, window_size=256, batch_size=32, epochs=5, lr=0.001, patience=4):
    combined_dataset = load_all_datasets(root_folder, window_size, batch_size)
    if combined_dataset is None:
        return None, None, None

    train_ds, test_ds = split_dataset(combined_dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StepCounterCNN(window_size).to(device).float()

    criterion = nn.BCELoss(weight=torch.tensor([5.0], device=device).float())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    early_stopping = EarlyStopping(patience=patience)

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {ep+1}/{epochs}") as pbar:
            for X, Y in train_loader:
                X, Y = X.float().to(device).permute(0, 2, 1), Y.float().to(device).max(dim=1, keepdim=True)[0]
                optimizer.zero_grad()
                loss = criterion(model(X), Y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                pbar.update(1)

        train_losses.append(ep_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_loss = sum(
                criterion(
                    model(X.float().to(device).permute(0, 2, 1)), Y.float().to(device).max(dim=1, keepdim=True)[0]
                ).item()
                for X, Y in test_loader
            ) / len(test_loader)
        test_losses.append(test_loss)
        print(f"\U0001F535 Epoch {ep+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

        if early_stopping.check(train_losses[-1], test_losses[-1], model, ep):
            break

    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"The model was saved at epoch {early_stopping.best_epoch + 1}.")
    return model, test_loader, device


def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model using a test dataset and generates performance metrics.

    Parameters:
    model (torch.nn.Module): Trained model to be evaluated.
    test_loader (DataLoader): DataLoader for the test dataset.
    device (torch.device): The device (CPU/GPU) on which evaluation is performed.

    Outputs:
    - Prints a classification report.
    - Displays a confusion matrix.
    - Roc Plot
    """
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.float().to(device).permute(0, 2, 1), Y.float().to(device).max(dim=1, keepdim=True)[0]
            outputs = model(X).cpu().numpy()
            predictions = (outputs > 0.5).astype(int)
            y_true.extend(Y.cpu().numpy().flatten())
            y_pred.extend(predictions.flatten())
            y_scores.extend(outputs.flatten())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Step", "Step"], yticklabels=["No Step", "Step"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="dashed")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def main():
    root_folder = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/real_output"
    window_size = 64
    batch_size = 128
    epochs = 20

    model, test_loader, device = train_step_counter(root_folder, window_size, batch_size, epochs, 1e-3)
    if model is not None and test_loader is not None:
        evaluate_model(model, test_loader, device)

    model_path = "best_model.pth"
    left_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/test/005/005_left_acceleration_data.csv"
    right_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/test/005/005_right_acceleration_data.csv"
    stepcount_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/test/005/scaled_step_counts.csv"

    prediction(model_path, left_csv, right_csv, stepcount_csv)


if __name__ == "__main__":
    main()
