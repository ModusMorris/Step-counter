from prediction import main as prediction
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    multilabel_confusion_matrix
)
import pandas as pd

# ==========================================
# Helper Classes & Functions
# ==========================================
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.005, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.best_train_loss = float("inf")

    def check(self, train_loss, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_train_loss = train_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        elif abs(train_loss - val_loss) > self.min_delta and val_loss >= self.best_loss:
            print("Early stopping triggered due to overfitting!")
            model.load_state_dict(torch.load(self.path))
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered! Best model from epoch {self.best_epoch + 1} loaded.")
                model.load_state_dict(torch.load(self.path))
                return True
        return False

def split_dataset(dataset, ratio=0.2):
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=ratio, random_state=42)
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, test_idx)

def load_all_datasets(root_folder, window_size, batch_size, gait_info_df):
    """Extended loading of all folders, passing gait_info_df to load_datasets()."""
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print("No folders found in", root_folder)
        return None

    all_datasets = []
    for sf in subfolders:
        dl = load_datasets(sf, window_size, batch_size, gait_info_df)  # <-- adjusted in data_generator
        if dl is not None:
            all_datasets.append(dl.dataset)

    if not all_datasets:
        print("No datasets available!")
        return None

    combined = ConcatDataset(all_datasets)
    print(f"{len(all_datasets)} datasets, total: {len(combined)} samples.")
    return combined

# ==========================================
# Main Training Function
# ==========================================
def train_step_counter(
    root_folder,
    window_size=256,
    batch_size=32,
    epochs=5,
    lr=0.001,
    patience=4,
    gait_csv="D:/Step-counter/Data/acceleration_metadata.csv"
):
    # 1) Load CSV with gait information
    gait_info_df = pd.read_csv(gait_csv)

    # 2) Load dataset
    combined_dataset = load_all_datasets(root_folder, window_size, batch_size, gait_info_df)
    if combined_dataset is None:
        return None, None, None

    # 3) Split into training/validation
    train_ds, test_ds = split_dataset(combined_dataset, ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4) Define model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StepCounterCNN(window_size).to(device).float()

    # For multi-label: BCE loss without logits, as we already have sigmoid in the model
    # For step, additionally a weight if you have imbalance
    # Here a simplified example (without special weights)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # EarlyStopping & loss lists
    early_stopping = EarlyStopping(patience=patience)
    train_losses, test_losses = [], []

    # 5) Training
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0

        for X, Y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            # X shape: (batch, window_size, 2) => (batch,2,window_size)
            X = X.permute(0,2,1).float().to(device)
            Y = Y.float().to(device)  # shape: (batch, window_size, 7)

            # 1) Step label
            y_step = Y[:,:,0].max(dim=1).values.unsqueeze(1)  # (batch,1)

            # 2) Gait label
            y_gait = Y[:,0,1:]  # (batch,6)

            optimizer.zero_grad()
            out = model(X)  # (batch,7)

            # => pred_step shape (batch,1), pred_gait shape (batch,6)
            pred_step = out[:,0].unsqueeze(1)
            pred_gait = out[:,1:]

            loss_step = criterion(pred_step, y_step)
            loss_gait = criterion(pred_gait, y_gait)
            loss = loss_step + loss_gait
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        train_loss = ep_loss / len(train_loader)
        train_losses.append(train_loss)

        # 6) Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in test_loader:
                X_val = X_val.permute(0,2,1).float().to(device)
                Y_val = Y_val.float().to(device)

                y_step_val = Y_val[:,:,0].max(dim=1).values.unsqueeze(1)
                y_gait_val = Y_val[:,0,1:]

                out_val = model(X_val)
                pred_step_val = out_val[:,0].unsqueeze(1)
                pred_gait_val = out_val[:,1:]

                loss_step_val = criterion(pred_step_val, y_step_val)
                loss_gait_val = criterion(pred_gait_val, y_gait_val)
                val_loss += (loss_step_val + loss_gait_val).item()

        val_loss /= len(test_loader)
        test_losses.append(val_loss)

        print(f"Epoch {ep+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Early Stopping
        if early_stopping.check(train_loss, val_loss, model, ep):
            break

    # 7) Plot training progress
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Validation Loss", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Best model was saved from epoch {early_stopping.best_epoch + 1}.")
    return model, test_loader, device

# ==========================================
# Evaluate Function with Multi-Label
# ==========================================
def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on:
      - Steps (binary)
      - 6 Gait labels (multi-label)
    Then prints classification reports and confusion matrices,
    plus ROC curves for each label.
    """
    model.eval()

    all_step_true, all_step_pred, all_step_prob = [], [], []
    all_gait_true, all_gait_pred, all_gait_prob = [], [], []

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.permute(0,2,1).float().to(device)
            Y = Y.float().to(device)  # (batch, window_size, 7)

            # True labels
            y_step = Y[:,:,0].max(dim=1).values  # (batch,)
            y_gait = Y[:,0,1:]                   # (batch,6)

            out = model(X)       # (batch,7)
            pred_step = out[:,0] # (batch,)
            pred_gait = out[:,1:]# (batch,6)

            # Save for later metrics
            all_step_true.append(y_step.cpu().numpy())           # shape (batch,)
            all_step_prob.append(pred_step.cpu().numpy())        # shape (batch,)
            all_step_pred.append((pred_step>0.5).cpu().numpy())  # shape (batch,)

            all_gait_true.append(y_gait.cpu().numpy())             # shape (batch,6)
            all_gait_prob.append(pred_gait.cpu().numpy())          # shape (batch,6)
            all_gait_pred.append((pred_gait>0.5).cpu().numpy())    # shape (batch,6)

    # Combine into NumPy arrays
    all_step_true = np.concatenate(all_step_true, axis=0)
    all_step_prob = np.concatenate(all_step_prob, axis=0)
    all_step_pred = np.concatenate(all_step_pred, axis=0)

    all_gait_true = np.concatenate(all_gait_true, axis=0)   # (N,6)
    all_gait_prob = np.concatenate(all_gait_prob, axis=0)   # (N,6)
    all_gait_pred = np.concatenate(all_gait_pred, axis=0)   # (N,6)

    # --- 1) Steps (binary classification) ---
    print("\n=== Steps (Binary) ===")
    print(classification_report(all_step_true, all_step_pred, target_names=["No Step","Step"]))

    cm = confusion_matrix(all_step_true, all_step_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Step","Step"], yticklabels=["No Step","Step"])
    plt.title("Confusion Matrix (Steps)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC curve for steps
    fpr, tpr, _ = roc_curve(all_step_true, all_step_prob)
    step_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Step AUC={step_auc:.2f}")
    plt.plot([0,1],[0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Steps)")
    plt.legend()
    plt.grid()
    plt.show()

    # --- 2) Gait types (multi-label) ---
    gait_names = ["langsames_gehen","normales_gehen","laufen",
                  "frei_mitschwingend","links_in_ht","rechts_in_ht"]

    print("\n=== Gait Types (Multi-Label) ===")
    print("-> Classification report per label:")
    print(classification_report(all_gait_true, all_gait_pred, target_names=gait_names))

    # Multi-label confusion matrix (each row = own 2x2)
    ml_cms = multilabel_confusion_matrix(all_gait_true, all_gait_pred)
    for i, label_name in enumerate(gait_names):
        cm_i = ml_cms[i]
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_i, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {label_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # ROC curves for each gait type
    plt.figure()
    for i, label_name in enumerate(gait_names):
        fpr_i, tpr_i, _ = roc_curve(all_gait_true[:,i], all_gait_prob[:,i])
        auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, label=f"{label_name} (AUC={auc_i:.2f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Gaits)")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Example paths
    root_folder = r"D:\Step-counter\Output"
    window_size = 64
    batch_size = 128
    epochs = 20
    lr = 1e-3

    model, test_loader, device = train_step_counter(
        root_folder, window_size, batch_size, epochs, lr
    )
    if model is not None and test_loader is not None:
        evaluate_model(model, test_loader, device)
    model_path = "best_model.pth"
    left_csv = "D:\Step-counter\Output\GX010061\GX010061_left_acceleration_data.csv"
    right_csv = "D:\Step-counter\Output\GX010061\GX010061_right_acceleration_data.csv"
    stepcount_csv = "D:\Step-counter\Output\GX010061\scaled_step_counts.csv"

    prediction(model_path, left_csv, right_csv, stepcount_csv)

if __name__ == "__main__":
    main()