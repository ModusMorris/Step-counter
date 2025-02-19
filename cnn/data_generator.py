import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import ast  # for safely converting string lists to Python lists


class StepCounterDataset(Dataset):
    def __init__(self, left_data, right_data, step_counts, window_size):
        """
        Creates training windows (segments) from accelerometer data.
        Labels are based on step peaks (left_foot_index, right_foot_index).
        """
        # 1) Merge acceleration data (Left + Right)
        self.data = np.hstack((left_data[["X", "Y", "Z"]], right_data[["X", "Y", "Z"]]))
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

        # 2) Extract step peaks
        def extract_peaks(peaks_str):
            if isinstance(peaks_str, str) and peaks_str.startswith("["):
                return ast.literal_eval(peaks_str)
            return []

        left_str = step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values[0]
        right_str = step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values[0]
        left_peaks = extract_peaks(left_str)
        right_peaks = extract_peaks(right_str)

        # 3) Build labels (0/1 for each frame)
        self.step_labels = np.zeros(len(self.data), dtype=np.float32)
        for p in left_peaks + right_peaks:
            start = max(0, p - window_size // 2)
            end = min(len(self.data), p + window_size // 2)
            self.step_labels[start:end] = 1  # Mark this region as "step"

        self.window_size = window_size

    def __len__(self):
        # Number of possible segments = total length - window size
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.step_labels[idx : idx + self.window_size]
        return x, y


def load_datasets(folder_path, window_size, batch_size):
    """
    Reads (foldername)_left_acceleration_data.csv,
          (foldername)_right_acceleration_data.csv,
          scaled_step_counts.csv
    and creates a DataLoader with segments.
    """
    folder_name = os.path.basename(folder_path)
    left_file = os.path.join(folder_path, f"{folder_name}_left_acceleration_data.csv")
    right_file = os.path.join(folder_path, f"{folder_name}_right_acceleration_data.csv")
    step_file = os.path.join(folder_path, "scaled_step_counts.csv")

    if not (os.path.exists(left_file) and os.path.exists(right_file) and os.path.exists(step_file)):
        print(f"Folder {folder_name}: Missing files, skipping.")
        return None

    left_data = pd.read_csv(left_file)
    right_data = pd.read_csv(right_file)
    step_counts = pd.read_csv(step_file)

    if left_data.empty or right_data.empty or step_counts.empty:
        print(f"Folder {folder_name}: Empty data, skipping.")
        return None

    dataset = StepCounterDataset(left_data, right_data, step_counts, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
