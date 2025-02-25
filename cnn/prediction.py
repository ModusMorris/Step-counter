import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import ast
from model_step_counter import StepCounterCNN


def load_model(model_path, device, window_size=64):
    """Loads the trained model."""
    model = StepCounterCNN(window_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_enmo(data):
    """Computes the Euclidean Norm Minus One (ENMO) from accelerometer data."""
    norm = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2) - 1
    return np.maximum(norm, 0)


def process_data(left_csv, right_csv):
    """Loads and processes acceleration data from left and right foot CSV files."""
    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)

    return pd.DataFrame({"ENMO_left": compute_enmo(left_df), "ENMO_right": compute_enmo(right_df)})


def detect_steps(model, device, data, window_size=64):
    """Runs the step detection model on the given data."""
    data = torch.tensor(StandardScaler().fit_transform(data), dtype=torch.float32, device=device)
    frame_probs = np.zeros(len(data), dtype=np.float32)
    overlap_cnt = np.zeros(len(data), dtype=np.float32)

    with torch.no_grad():
        for start in range(len(data) - window_size):
            window = data[start : start + window_size].T.unsqueeze(0)
            frame_probs[start : start + window_size] += model(window).cpu().numpy().flatten()
            overlap_cnt[start : start + window_size] += 1

    frame_probs[overlap_cnt > 0] /= overlap_cnt[overlap_cnt > 0]
    return find_peaks(frame_probs, height=0.02, distance=30, prominence=0.05)[0]


def parse_groundtruth_steps(groundtruth_csv):
    """Parses the ground truth step data from CSV."""
    groundtruth_df = pd.read_csv(groundtruth_csv, nrows=2)  # Only consider the first two rows
    steps = set()
    for peak_str in groundtruth_df["Peaks"].dropna():
        try:
            steps.update(ast.literal_eval(peak_str))
        except (SyntaxError, ValueError):
            continue
    return steps


def plot_results(data, detected_steps, groundtruth_steps):
    """Generates an interactive Plotly visualization of acceleration data, detected steps, and ground truth."""
    fig = go.Figure()
    time_axis = np.arange(len(data))

    # Plot acceleration data
    for col in data.columns:
        fig.add_trace(go.Scatter(x=time_axis, y=data[col], mode="lines", name=col))

    # Plot detected steps
    fig.add_trace(
        go.Scatter(
            x=list(detected_steps),
            y=[data.iloc[i].mean() for i in detected_steps],
            mode="markers",
            name=f"Detected Steps ({len(detected_steps)})",
            marker=dict(color="red", size=8),
        )
    )

    # Plot ground truth steps
    fig.add_trace(
        go.Scatter(
            x=list(groundtruth_steps),
            y=[data.iloc[i].mean() for i in groundtruth_steps],
            mode="markers",
            name=f"Ground Truth Steps ({len(groundtruth_steps)})",
            marker=dict(color="green", symbol="x", size=8),
        )
    )

    fig.update_layout(
        title="Step Detection Visualization",
        xaxis_title="Frame",
        yaxis_title="Acceleration / Probability",
        legend_title="Legend",
        template="plotly_white",
    )

    fig.show()


def main(model_path, left_csv, right_csv, groundtruth_csv):
    """Runs the full step detection pipeline and visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    data = process_data(left_csv, right_csv)
    detected_steps = detect_steps(model, device, data)
    groundtruth_steps = parse_groundtruth_steps(groundtruth_csv)

    plot_results(data, detected_steps, groundtruth_steps)


if __name__ == "__main__":
    model_path = "D:/Daisy/5. Semester/SmartHealth/Step-counter/cnn/best_model.pth"
    left_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/test/005/005_left_acceleration_data.csv"
    right_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/test/005/005_right_acceleration_data.csv"
    groundtruth_csv = "D:/Daisy/5. Semester/SmartHealth/Step-counter/Output/processed_sliced_and_scaled data/test/005/scaled_step_counts.csv"

    main(model_path, left_csv, right_csv, groundtruth_csv)
