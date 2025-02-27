import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import ast
from model_step_counter import StepCounterCNN

def load_model(model_path, device, window_size=64):
    # Load the model from the specified path and set it to evaluation mode
    model = StepCounterCNN(window_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_enmo(data):
    # Compute the Euclidean Norm Minus One (ENMO) for the accelerometer data
    norm = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2) - 1
    return np.maximum(norm, 0)

def process_data(left_csv, right_csv):
    # Load and process the left and right accelerometer data
    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)
    return pd.DataFrame({
        "ENMO_left": compute_enmo(left_df),
        "ENMO_right": compute_enmo(right_df)
    })

def detect_multi_label(model, device, data, window_size=64):
    """
    Returns per frame:
      - step_prob[frame] = model step probability
      - gait_probs[frame,0..5] = model gait probabilities
    We average overlapping windows as in the old detect_steps().
    """
    data_torch = torch.tensor(StandardScaler().fit_transform(data), dtype=torch.float32, device=device)
    n = len(data_torch)

    step_sum = np.zeros(n, dtype=np.float32)
    step_cnt = np.zeros(n, dtype=np.float32)
    gait_sum = np.zeros((n, 6), dtype=np.float32)
    gait_cnt = np.zeros((n, 6), dtype=np.float32)

    with torch.no_grad():
        for start in range(n - window_size):
            window = data_torch[start : start+window_size].T.unsqueeze(0)  # shape (1,2,window_size)
            out = model(window)  # shape (1,7)
            out_np = out[0].cpu().numpy()  # shape (7,)

            step_val = out_np[0]
            gait_vals = out_np[1:]  # shape (6,)

            # Distribute values to all indices of the window
            step_sum[start : start+window_size] += step_val
            step_cnt[start : start+window_size] += 1

            gait_sum[start : start+window_size, :] += gait_vals
            gait_cnt[start : start+window_size, :] += 1

    # Compute averages
    mask_step = step_cnt > 0
    step_sum[mask_step] /= step_cnt[mask_step]

    mask_gait = gait_cnt > 0
    gait_sum[mask_gait] /= gait_cnt[mask_gait]

    return step_sum, gait_sum  # shape(n,) & shape(n,6)

def parse_groundtruth_steps(groundtruth_csv):
    # Parse ground truth steps from the CSV file
    groundtruth_df = pd.read_csv(groundtruth_csv, nrows=2)
    steps = set()
    for peak_str in groundtruth_df["Peaks"].dropna():
        try:
            steps.update(ast.literal_eval(peak_str))
        except:
            pass
    return steps

def plot_results(data, step_probs, gait_probs, detected_steps, groundtruth_steps):
    """
    Plotly visualization:
      - ENMO_left / ENMO_right as lines (left axis).
      - Step probability (0..1) as a red line on the left axis.
      - Detected steps as red markers, ground truth steps as green markers.
      - Predicted gait on a SECOND y-axis (right side), showing category labels instead of 0..5.
    """
    # 1) Determine the single best gait per frame by argmax
    gait_names = ["langsames_gehen","normales_gehen","laufen",
                  "frei_mitschwingend","links_in_ht","rechts_in_ht"]
    predicted_gait_index = np.argmax(gait_probs, axis=1)

    fig = go.Figure()
    time_axis = np.arange(len(data))

    # ENMO signals on the left axis
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=data["ENMO_left"], 
        mode="lines", 
        name="ENMO_left",
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=data["ENMO_right"], 
        mode="lines", 
        name="ENMO_right",
        yaxis="y1"
    ))

    # Step probability on the left axis as well
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=step_probs, 
        mode="lines", 
        name="Step Probability",  # the CNN output for steps
        line=dict(color="red"),
        yaxis="y1"
    ))

    # Detected steps (red markers) on the left axis
    fig.add_trace(go.Scatter(
        x=list(detected_steps),
        y=[step_probs[i] for i in detected_steps],
        mode="markers",
        name=f"Detected Steps ({len(detected_steps)})",
        marker=dict(color="red", size=8),
        yaxis="y1"
    ))

    # Ground truth steps (green markers) on the left axis
    fig.add_trace(go.Scatter(
        x=list(groundtruth_steps),
        y=[step_probs[i] for i in groundtruth_steps],
        mode="markers",
        name=f"Ground Truth Steps ({len(groundtruth_steps)})",
        marker=dict(color="green", symbol="x", size=8),
        yaxis="y1"
    ))

    # Plot predicted gait on a SECOND y-axis, showing index as numeric
    # but we will map them to category labels with tickvals/ticktext below
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=predicted_gait_index,
        mode="markers",
        name="Predicted Gait",
        marker=dict(color="blue", size=6),
        yaxis="y2"  # <--- second axis
    ))

    # Layout with two y-axes
    fig.update_layout(
        title="Steps and Single Best Gait",
        xaxis=dict(title="Frame"),
        yaxis=dict(
            title="Acceleration / Probability",
            side="left",
            range=[0, max(step_probs.max(), data["ENMO_left"].max(), data["ENMO_right"].max())+0.5],
        ),
        # y2: the axis for the gait index
        yaxis2=dict(
            title="Predicted Gait (Categories)",
            overlaying="y",   # shares the same x-axis
            side="right",
            tickmode="array",
            tickvals=[0,1,2,3,4,5],     # indices 0..5
            ticktext=gait_names,       # the actual labels
            range=[-0.5, 5.5]
        ),
        legend_title="Legend",
        template="plotly_white",
    )

    fig.show()

    # Majority vote on the entire sequence
    majority_gait_idx = np.bincount(predicted_gait_index).argmax()
    majority_gait_label = gait_names[majority_gait_idx]
    print(f"Detected {len(detected_steps)} steps in total.")
    print(f"Overall predicted gait (majority): {majority_gait_label}")


def main(model_path, left_csv, right_csv, groundtruth_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Load and process data
    data = process_data(left_csv, right_csv)

    # Step & gait probabilities per frame
    step_probs, gait_probs = detect_multi_label(model, device, data, window_size=64)

    # Detected steps (small peak detector on step_probs)
    idx_peaks = find_peaks(step_probs, height=0.02, distance=30, prominence=0.05)[0]

    # Ground truth
    groundtruth_steps = parse_groundtruth_steps(groundtruth_csv)

    # Final plot: only 1 gait (via argmax) + total steps
    plot_results(data, step_probs, gait_probs, idx_peaks, groundtruth_steps)


if __name__ == "__main__":
    model_path = "best_model.pth"
    left_csv = "path_to_left.csv"
    right_csv = "path_to_right.csv"
    groundtruth_csv = "path_to_step_counts.csv"
    main(model_path, left_csv, right_csv, groundtruth_csv)
