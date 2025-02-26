import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import ast


def compute_enmo(data):
    """
    Compute the ENMO value for accelerometer data.

    Parameters:
        data (DataFrame): Data with 'X', 'Y', and 'Z' columns.

    Returns:
        ndarray: ENMO values with negatives set to 0.
    """
    norm = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2) - 1
    return np.maximum(norm, 0)


class StepCounterDataset(Dataset):
    """
    Dataset for step counting using left and right accelerometer data.
    """
    def __init__(self, left_data, right_data, step_counts, window_size):
        """
        Initialize the dataset by computing ENMO, differences, and step labels.

        Parameters:
            left_data (DataFrame): Left foot accelerometer data.
            right_data (DataFrame): Right foot accelerometer data.
            step_counts (DataFrame): CSV data containing step peaks.
            window_size (int): Size of the data window.
        """
        self.window_size = window_size

        left_data["ENMO"] = compute_enmo(left_data)
        right_data["ENMO"] = compute_enmo(right_data)

        left_data["ENMO_DIFF"] = left_data["ENMO"].diff().fillna(0)
        right_data["ENMO_DIFF"] = right_data["ENMO"].diff().fillna(0)

        self.data = np.hstack((left_data[["ENMO_DIFF"]], right_data[["ENMO_DIFF"]]))

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

        def extract_peaks(peaks_str):
            if isinstance(peaks_str, str):
                try:
                    return ast.literal_eval(peaks_str) if peaks_str.startswith("[") else []
                except:
                    return []
            return []

        left_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values[0])
        right_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values[0])

        self.step_labels = np.zeros(len(self.data), dtype=np.float32)
        for p in left_peaks + right_peaks:
            if 0 <= p < len(self.step_labels) - (window_size // 2):
                self.step_labels[p + (window_size // 2)] = 1

        print("\n==== Debugging Step Extraction ====")
        print("Step data (first few rows):")
        print(step_counts.head())
        print("\nExtraction of peaks for the left foot:")
        print("Raw data:", step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values)
        print("Extracted peaks:", left_peaks)
        print("\nExtraction of peaks for the right foot:")
        print("Raw data:", step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values)
        print("Extracted peaks:", right_peaks)
        print("\nTotal peaks found: Left =", len(left_peaks), ", Right =", len(right_peaks))
        print("==================================\n")

    def __len__(self):
        """
        Return the number of data segments.
        """
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        Get a data segment and its step labels with added noise.

        Parameters:
            idx (int): Starting index of the segment.

        Returns:
            tuple: (augmented data segment, corresponding labels).
        """
        x = self.data[idx : idx + self.window_size]
        y = self.step_labels[idx : idx + self.window_size]
        noise = np.random.normal(0, 0.02, x.shape)
        x_augmented = x + noise
        return x_augmented, y


def load_datasets(folder_path, window_size, batch_size):
    """
    Load accelerometer and step count CSV files from a folder and create a DataLoader.

    Parameters:
        folder_path (str): Path to the folder containing the CSV files.
        window_size (int): Size of each data window.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the dataset, or None if files are missing/empty.
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
