import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import ast
from model_step_counter import StepCounterCNN


def load_model(model_path, device, window_size=64):
    """
    Load a pre-trained StepCounterCNN model.

    Parameters:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model onto.
        window_size (int, optional): Input window size for the model. Defaults to 64.

    Returns:
        StepCounterCNN: The loaded and evaluated model.
    """
    model = StepCounterCNN(window_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_enmo(data):
    """
    Compute the Euclidean Norm Minus One (ENMO) for accelerometer data.

    This function calculates the Euclidean norm of the 'X', 'Y', and 'Z' columns,
    subtracts 1 from the result, and returns the maximum between the computed value and 0.

    Parameters:
        data (DataFrame): Accelerometer data with 'X', 'Y', and 'Z' columns.

    Returns:
        np.ndarray: Array of ENMO values.
    """
    norm = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2) - 1
    return np.maximum(norm, 0)


def process_data(left_csv, right_csv):
    """
    Load and process accelerometer data from left and right CSV files.

    Reads the CSV files for both left and right foot data, computes the ENMO for each,
    and returns a DataFrame combining the results.

    Parameters:
        left_csv (str): Path to the left foot accelerometer CSV.
        right_csv (str): Path to the right foot accelerometer CSV.

    Returns:
        DataFrame: A DataFrame with columns 'ENMO_left' and 'ENMO_right'.
    """
    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)

    return pd.DataFrame({"ENMO_left": compute_enmo(left_df), "ENMO_right": compute_enmo(right_df)})


def detect_steps(model, device, data, window_size=64):
    """
    Run the step detection model on processed accelerometer data.

    The function scales the data using a StandardScaler, applies the model over sliding windows,
    aggregates the output probabilities, and identifies step peaks using a threshold.

    Parameters:
        model (StepCounterCNN): The loaded step detection model.
        device (torch.device): Device for computation.
        data (DataFrame): Processed accelerometer data.
        window_size (int, optional): Size of the sliding window. Defaults to 64.

    Returns:
        ndarray: Indices of detected step peaks.
    """
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
    """
    Parse ground truth step data from a CSV file.

    Reads the ground truth CSV, extracts the 'Peaks' column from the first two rows,
    evaluates the string representations, and returns a set of ground truth step indices.

    Parameters:
        groundtruth_csv (str): Path to the ground truth CSV file.

    Returns:
        set: A set containing ground truth step indices.
    """
    groundtruth_df = pd.read_csv(groundtruth_csv, nrows=2)  # Only consider the first two rows
    steps = set()
    for peak_str in groundtruth_df["Peaks"].dropna():
        try:
            steps.update(ast.literal_eval(peak_str))
        except (SyntaxError, ValueError):
            continue
    return steps


def plot_results(data, detected_steps, groundtruth_steps):
    """
    Create an interactive Plotly visualization of acceleration data and step detections.

    Plots the acceleration signals for each channel, overlays markers for detected steps and
    ground truth steps, and displays the interactive figure.

    Parameters:
        data (DataFrame): Combined accelerometer data (e.g., 'ENMO_left' and 'ENMO_right').
        detected_steps (ndarray): Indices of steps detected by the model.
        groundtruth_steps (set): Set of ground truth step indices.
    """
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
    """
    Execute the full step detection pipeline and visualization.

    Loads the trained model, processes accelerometer data from left and right CSV files,
    runs the step detection, parses ground truth step data, and visualizes the results.

    Parameters:
        model_path (str): Path to the saved model weights.
        left_csv (str): Path to the left foot accelerometer CSV.
        right_csv (str): Path to the right foot accelerometer CSV.
        groundtruth_csv (str): Path to the ground truth CSV file.
    """
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

