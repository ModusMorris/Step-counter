import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import ast

def compute_enmo(data):
    # Compute the Euclidean Norm Minus One (ENMO) for the accelerometer data
    norm = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2) - 1
    return np.maximum(norm, 0)

class StepCounterDataset(Dataset):
    def __init__(self, left_data, right_data, step_counts, window_size, gait_vector):
        """
        Args:
            left_data, right_data: Accelerometer data (DataFrame).
            step_counts: DataFrame with step peaks.
            window_size: Length of the window.
            gait_vector: np.array of length 6, e.g., [0,1,0,1,0,0] for gait labels.
        """
        self.window_size = window_size

        # --- ENMO, differences, normalization as usual ---
        left_data["ENMO"] = compute_enmo(left_data)
        right_data["ENMO"] = compute_enmo(right_data)

        left_data["ENMO_DIFF"] = left_data["ENMO"].diff().fillna(0)
        right_data["ENMO_DIFF"] = right_data["ENMO"].diff().fillna(0)

        self.data = np.hstack((
            left_data[["ENMO_DIFF"]].values,
            right_data[["ENMO_DIFF"]].values
        ))

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

        # --- Extract step labels ---
        def extract_peaks(peaks_str):
            if isinstance(peaks_str, str):
                try:
                    return ast.literal_eval(peaks_str) if peaks_str.startswith("[") else []
                except:
                    return []
            return []

        left_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "left_foot_index", "Peaks"].values[0])
        right_peaks = extract_peaks(step_counts.loc[step_counts["Joint"] == "right_foot_index", "Peaks"].values[0])

        # Step label: 0 or 1 per sample
        self.step_labels = np.zeros(len(self.data), dtype=np.float32)
        for p in left_peaks + right_peaks:
            # Small offset to center the peak in the window
            if 0 <= p < len(self.step_labels) - (window_size // 2):
                self.step_labels[p + (window_size // 2)] = 1

        # --- Gait label: 6-dimensional vector ---
        # Since the gait is the same for the entire dataset, we store it
        # (and will repeat it for each window).
        self.gait_label = gait_vector.astype(np.float32)  # shape (6,)

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # x shape: (window_size, 2)
        x = self.data[idx : idx + self.window_size]

        # Step label per sample in the window
        y_step = self.step_labels[idx : idx + self.window_size]  # shape (window_size,)

        # Gait label is the same for the entire window:
        # We create a shape (window_size, 6) that carries self.gait_label everywhere.
        y_gait = np.tile(self.gait_label, (self.window_size, 1))  # (window_size, 6)

        # We combine everything into one label: (window_size, 7)
        # Column 0 = step label, columns 1..6 = gait
        # => y[i,0] = y_step[i], y[i,1:] = y_gait[i]
        y = np.zeros((self.window_size, 7), dtype=np.float32)
        y[:, 0] = y_step
        y[:, 1:] = y_gait

        # Optional data augmentation
        noise = np.random.normal(0, 0.02, x.shape)
        x_augmented = x + noise

        return x_augmented, y

def load_datasets(folder_path, window_size, batch_size, gait_info_df):
    """
    Loads data from:
      - (Folder name)_left_acceleration_data.csv
      - (Folder name)_right_acceleration_data.csv
      - scaled_step_counts.csv
    Searches the DataFrame `gait_info_df` for the row corresponding to the current folder ID
    and constructs a gait label from it.
    """
    folder_name = os.path.basename(folder_path)
    left_file = os.path.join(folder_path, f"{folder_name}_left_acceleration_data.csv")
    right_file = os.path.join(folder_path, f"{folder_name}_right_acceleration_data.csv")
    step_file = os.path.join(folder_path, "scaled_step_counts.csv")

    if not (os.path.exists(left_file) and os.path.exists(right_file) and os.path.exists(step_file)):
        print(f"Folder {folder_name}: Missing files, skipping.")
        return None

    # Load CSVs
    left_data = pd.read_csv(left_file)
    right_data = pd.read_csv(right_file)
    step_counts = pd.read_csv(step_file)

    if left_data.empty or right_data.empty or step_counts.empty:
        print(f"Folder {folder_name}: Empty data, skipping.")
        return None

    # --- Get gait label from gait_info_df ---
    # e.g., video_id == folder_name
    row = gait_info_df[gait_info_df["video_id"] == folder_name]
    if row.empty:
        print(f"No row found for {folder_name} in gait_info_df, using 0-label.")
        gait_label = np.zeros(6, dtype=np.float32)
    else:
        # Important: Adjust the order here to match the columns from the CSV
        gait_label = row[["langsames_gehen","normales_gehen","laufen",
                          "frei_mitschwingend","links_in_ht","rechts_in_ht"]].values[0]
        # => shape (6,)

    dataset = StepCounterDataset(left_data, right_data, step_counts,
                                 window_size, gait_vector=gait_label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)