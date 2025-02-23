import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_generator import load_datasets
from model_step_counter import StepCounterCNN
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
from scipy.signal import find_peaks

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
    all_data_loaders = [load_datasets(sf, window_size, batch_size).dataset for sf in subfolders if load_datasets(sf, window_size, batch_size) is not None]
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

def train_step_counter(root_folder, window_size=256, batch_size=32, epochs=5, lr=0.001):
    """
    Loads data, trains the step counter CNN model, and evaluates loss.
    
    Parameters:
    root_folder (str): Path to the dataset directory.
    window_size (int): Number of samples per input window.
    batch_size (int): Number of samples per batch.
    epochs (int): Number of training epochs.
    lr (float): Learning rate for the optimizer.
    
    Returns:
    Tuple[StepCounterCNN, DataLoader, torch.device]: Trained model, test data loader, and device used.
    """
    combined_dataset = load_all_datasets(root_folder, window_size, batch_size)
    if combined_dataset is None:
        return None, None, None
    
    # Split dataset into training and testing sets
    train_ds, test_ds = split_dataset(combined_dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StepCounterCNN(window_size).to(device).float()
    
    # Define loss function and optimizer
    criterion = nn.BCELoss(weight=torch.tensor([5.0], device=device).float())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []

    # Training loop
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
                
        # Store training loss
        train_losses.append(ep_loss / len(train_loader))
        
        # Evaluate test loss
        model.eval()
        with torch.no_grad():
            test_loss = sum(criterion(model(X.float().to(device).permute(0, 2, 1)), Y.float().to(device).max(dim=1, keepdim=True)[0]).item() for X, Y in test_loader) / len(test_loader)
        test_losses.append(test_loss)
        print(f"\U0001F535 Epoch {ep+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
    
    # Plot loss curves
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss", linestyle='dashed')
    plt.legend(), plt.grid(), plt.show()
    
    # Save model
    save_path = os.path.join(root_folder, "step_counter_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Step', 'Step'], yticklabels=['No Step', 'Step'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def evaluate_full_run_6channels_against_groundtruth_plotly(
    model, device, left_csv, right_csv, stepcount_csv, window_size=256, peak_distance=10, peak_height=0.5
):
    """
    Evaluates the full run by comparing the model's step detection against ground truth.
    Generates an interactive Plotly graph with acceleration data and detected steps.
    
    Parameters:
    model (torch.nn.Module): Trained step detection model.
    device (torch.device): The device (CPU/GPU) for processing.
    left_csv (str): Path to CSV file containing left foot acceleration data.
    right_csv (str): Path to CSV file containing right foot acceleration data.
    stepcount_csv (str): Path to CSV file containing ground truth step data.
    window_size (int, optional): Size of the sliding window for processing. Default is 256.
    peak_distance (int, optional): Minimum distance between detected peaks (steps). Default is 10.
    peak_height (float, optional): Minimum height for peak detection. Default is 0.5.
    
    Outputs:
    - Displays an interactive Plotly graph with acceleration data and step detection results.
    - Prints debugging information for CNN predictions.
    """
    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)
    step_counts = pd.read_csv(stepcount_csv)

    def compute_enmo(data):
        """Computes the Euclidean Norm Minus One (ENMO) from accelerometer data."""
        norm = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2) - 1
        return np.maximum(norm, 0)  # Set negative values to 0

    # Compute ENMO for left and right foot acceleration
    left_df["ENMO"] = compute_enmo(left_df)
    right_df["ENMO"] = compute_enmo(right_df)

    # Create a DataFrame with only ENMO values
    combined_df = pd.DataFrame({
        "ENMO_left": left_df["ENMO"],
        "ENMO_right": right_df["ENMO"]
    })

    def extract_peaks(peaks_str):
        """Extracts peak values from a string representation of a list."""
        import ast
        if isinstance(peaks_str, str) and peaks_str.startswith("["):
            return ast.literal_eval(peaks_str)
        return []

    # Extract ground truth step frames
    left_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values[0])
    right_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values[0])
    groundtruth_frames = set(left_peaks + right_peaks)

    # Normalize data using StandardScaler
    sc = StandardScaler()
    arr = sc.fit_transform(combined_df.values)  # shape=(Frames, 2)
    arr = torch.tensor(arr, dtype=torch.float32, device=device)

    model.eval()
    N = len(arr)
    frame_probs = np.zeros(N, dtype=np.float32)
    overlap_cnt = np.zeros(N, dtype=np.float32)

    # Sliding window approach for step detection
    with torch.no_grad():
        for start in tqdm(range(N - window_size), desc="FullRunProcessing"):
            window_ = arr[start : start + window_size, :].permute(1, 0).unsqueeze(0)
            out = model(window_)
            frame_probs[start : start + window_size] += out.squeeze(0).cpu().numpy()
            overlap_cnt[start : start + window_size] += 1

    # Normalize probabilities by overlap count
    valid = overlap_cnt > 0
    frame_probs[valid] /= overlap_cnt[valid]

    # Detect peaks in model output probabilities
    peaks, _ = find_peaks(frame_probs, height=0.02, distance=30, prominence=0.05)
    detected_frames = set(peaks.tolist())

    # Create Plotly figure
    fig = go.Figure()
    time_axis = np.arange(len(combined_df))

    for col in combined_df.columns:
        fig.add_trace(go.Scatter(x=time_axis, y=combined_df[col], mode="lines", name=col))

    fig.add_trace(
        go.Scatter(
            x=list(detected_frames),
            y=[frame_probs[i] for i in detected_frames],
            mode="markers",
            name="Detected Steps",
            marker=dict(color="red", size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(groundtruth_frames),
            y=[frame_probs[i] for i in groundtruth_frames],
            mode="markers",
            name="Ground Truth Steps",
            marker=dict(color="green", symbol="x", size=8),
        )
    )

    fig.update_layout(
        title="Accelerometer Data with Step Detection",
        xaxis_title="Frame",
        yaxis_title="Acceleration / Probability",
        legend_title="Legend",
        template="plotly_white",
    )

    fig.show()
    
    print("\n==== Debugging CNN Predictions ====")
    print("Frame probabilities (first 20 values):", frame_probs[:20])
    print("Max probability from CNN:", np.max(frame_probs))
    print("Mean probability from CNN:", np.mean(frame_probs))



def main():
    root_folder = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/real_output"
    window_size = 64
    batch_size = 32
    epochs = 5

    model, test_loader, device = train_step_counter(root_folder, window_size, batch_size, epochs)
    if model is not None and test_loader is not None:
        evaluate_model(model, test_loader, device)
    
    left_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/real_output/GX010029/GX010029_left_acceleration_data.csv"
    right_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/real_output/GX010029/GX010029_right_acceleration_data.csv"
    stepcount_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/real_output/GX010029/scaled_step_counts.csv"

    evaluate_full_run_6channels_against_groundtruth_plotly(
        model, device, left_csv, right_csv, stepcount_csv, window_size=window_size, peak_distance=10, peak_height=0.4
    )

if __name__ == "__main__":
    main()
