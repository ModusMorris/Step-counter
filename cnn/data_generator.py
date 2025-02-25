import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import ast


def compute_enmo(data):
    # Calculate the ENMO value for the data
    norm = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2) - 1
    return np.maximum(norm, 0)  # Set negative values to 0


class StepCounterDataset(Dataset):
    def __init__(self, left_data, right_data, step_counts, window_size):
        self.window_size = window_size  # Ensure window_size is assigned

        # Calculate ENMO for both feet
        left_data["ENMO"] = compute_enmo(left_data)
        right_data["ENMO"] = compute_enmo(right_data)

        # Calculate the difference in ENMO values
        left_data["ENMO_DIFF"] = left_data["ENMO"].diff().fillna(0)
        right_data["ENMO_DIFF"] = right_data["ENMO"].diff().fillna(0)

        # Stack the ENMO differences for both feet
        self.data = np.hstack((left_data[["ENMO_DIFF"]], right_data[["ENMO_DIFF"]]))

        # Normalize the data
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

        # Extract step labels
        def extract_peaks(peaks_str):
            if isinstance(peaks_str, str):
                try:
                    return ast.literal_eval(peaks_str) if peaks_str.startswith("[") else []
                except:
                    return []
            return []

        # Extract peaks for left and right feet
        left_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values[0])
        right_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values[0])

        # Create step labels
        self.step_labels = np.zeros(len(self.data), dtype=np.float32)

        # Shift step labels to improve peak positions for CNN
        for p in left_peaks + right_peaks:
            if 0 <= p < len(self.step_labels) - (window_size // 2):
                self.step_labels[p + (window_size // 2)] = 1

        # Debugging information
        print("\n==== Debugging Step Extraction ====")
        print("Step data (first few rows):")
        print(step_counts.head())

        print("\nExtraction of peaks for the left foot:")
        print("Raw data from CSV:", step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values)
        print("Extracted peaks:", left_peaks)

        print("\nExtraction of peaks for the right foot:")
        print("Raw data from CSV:", step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values)
        print("Extracted peaks:", right_peaks)

        print("\nTotal peaks found: Left =", len(left_peaks), ", Right =", len(right_peaks))
        print("==================================\n")

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.step_labels[idx : idx + self.window_size]

        # Data augmentation: Add slight noise to the data
        noise = np.random.normal(0, 0.02, x.shape)
        x_augmented = x + noise

        return x_augmented, y


def load_datasets(folder_path, window_size, batch_size):
    """
    Reads the following files:
    (Folder name)_left_acceleration_data.csv,
    (Folder name)_right_acceleration_data.csv,
    scaled_step_counts.csv
    and creates a DataLoader with segments.
    """
    folder_name = os.path.basename(folder_path)
    left_file = os.path.join(folder_path, f"{folder_name}_left_acceleration_data.csv")
    right_file = os.path.join(folder_path, f"{folder_name}_right_acceleration_data.csv")
    step_file = os.path.join(folder_path, "scaled_step_counts.csv")

    # Check if all required files exist
    if not (os.path.exists(left_file) and os.path.exists(right_file) and os.path.exists(step_file)):
        print(f"Folder {folder_name}: Missing files, skipping.")
        return None

    # Load data from CSV files
    left_data = pd.read_csv(left_file)
    right_data = pd.read_csv(right_file)
    step_counts = pd.read_csv(step_file)

    # Check if any of the dataframes are empty
    if left_data.empty or right_data.empty or step_counts.empty:
        print(f"Folder {folder_name}: Empty data, skipping.")
        return None

    # Create dataset and DataLoader
    dataset = StepCounterDataset(left_data, right_data, step_counts, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
