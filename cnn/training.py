import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_generator import load_datasets
from train_step_counter import StepCounterCNN
from torch.utils.data import DataLoader, ConcatDataset, Subset
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import pandas as pd


# 1) Load all folders & combine datasets
def load_all_datasets(root_folder, window_size, batch_size):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print("No folders found in", root_folder)
        return None

    all_data_loaders = []
    for sf in subfolders:
        dl = load_datasets(sf, window_size, batch_size)
        if dl is not None:
            all_data_loaders.append(dl.dataset)

    if not all_data_loaders:
        print("No datasets available!")
        return None

    combined = ConcatDataset(all_data_loaders)
    print(f"{len(all_data_loaders)} datasets, total: {len(combined)} samples.")
    return combined


# 2) Split into Train/Test (80/20)
def split_dataset(dataset, ratio=0.2):
    length = len(dataset)
    idx = np.arange(length)
    train_idx, test_idx = train_test_split(idx, test_size=ratio, random_state=42)
    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    return train_ds, test_ds


# 3) Training
def train_step_counter(root_folder, window_size=256, batch_size=32, epochs=5, lr=0.001):
    # 1) Load datasets
    combined_dataset = load_all_datasets(root_folder, window_size, batch_size)
    if combined_dataset is None:
        return None, None, None

    # 2) 80/20 Split
    train_ds, test_ds = split_dataset(combined_dataset, ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StepCounterCNN(window_size).to(device)

    criterion = nn.BCELoss(weight=torch.tensor([5.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    # 3) Training loop
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {ep+1}/{epochs}") as pbar:
            for X, Y in train_loader:
                X = X.float().to(device).permute(0, 2, 1)  # Shape (Batch, 2, window_size)
                Y = Y.float().to(device).max(dim=1, keepdim=True)[0]  # Mittelwert der Labels pro Fenster
                
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, Y)
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                pbar.update(1)

        avg_loss = ep_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"🔹 Epoch {ep+1}, Loss: {avg_loss:.4f}")

    # Plot loss
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model
    save_path = os.path.join(root_folder, "step_counter_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

    return model, test_loader, device


# 4) Evaluate Full Run - Ground Truth
def evaluate_full_run_6channels_against_groundtruth_plotly(
    model, device, left_csv, right_csv, stepcount_csv, window_size=256, peak_distance=10, peak_height=0.5
):
    """
    Erstellt einen interaktiven Plotly-Graphen mit den Beschleunigungsdaten und den erkannten sowie Ground-Truth-Schritten.
    """

    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)
    step_counts = pd.read_csv(stepcount_csv)

    def compute_enmo(data):
        """Berechnet die Euclidean Norm Minus One (ENMO)."""
        norm = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2) - 1
        return np.maximum(norm, 0)  # Negative Werte auf 0 setzen

    # Berechne ENMO für links und rechts
    left_df["ENMO"] = compute_enmo(left_df)
    right_df["ENMO"] = compute_enmo(right_df)

    # Erstelle DataFrame nur mit ENMO-Werten
    combined_df = pd.DataFrame({
        "ENMO_left": left_df["ENMO"],
        "ENMO_right": right_df["ENMO"]
    })

    def extract_peaks(peaks_str):
        import ast

        if isinstance(peaks_str, str) and peaks_str.startswith("["):
            return ast.literal_eval(peaks_str)
        return []

    left_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values[0])
    right_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values[0])
    groundtruth_frames = set(left_peaks + right_peaks)

    sc = StandardScaler()
    arr = sc.fit_transform(combined_df.values)  # shape=(Frames,6)
    arr = torch.tensor(arr, dtype=torch.float32, device=device)

    model.eval()
    N = len(arr)
    frame_probs = np.zeros(N, dtype=np.float32)
    overlap_cnt = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(N - window_size), desc="FullRunProcessing"):
            window_ = arr[start : start + window_size, :4].permute(1, 0).unsqueeze(0)
            out = model(window_)
            frame_probs[start : start + window_size] += out.squeeze(0).cpu().numpy()
            overlap_cnt[start : start + window_size] += 1

    valid = overlap_cnt > 0
    frame_probs[valid] /= overlap_cnt[valid]

    peaks, _ = find_peaks(frame_probs, height=0.02, distance=30, prominence=0.05)
    detected_frames = set(peaks.tolist())

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
    root_folder = "D:\\Step-counter\\Output"
    window_size = 64
    batch_size = 32
    epochs = 5

    # 1) Training
    model, test_loader, device = train_step_counter(
        root_folder, window_size=window_size, batch_size=batch_size, epochs=epochs
    )

    left_csv = "D:/Step-counter/Output/GX010029/GX010029_left_acceleration_data.csv"
    right_csv = "D:/Step-counter/Output/GX010029/GX010029_right_acceleration_data.csv"
    stepcount_csv = "D:/Step-counter/Output/GX010029/scaled_step_counts.csv"

    evaluate_full_run_6channels_against_groundtruth_plotly(
        model, device, left_csv, right_csv, stepcount_csv, window_size=window_size, peak_distance=10, peak_height=0.4
    )


if __name__ == "__main__":
    main()
