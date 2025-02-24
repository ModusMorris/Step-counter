import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load accelerometer data
def load_accelerometer_data(file_path):
    """Loads accelerometer data from CSV."""
    return pd.read_csv(file_path)

# Normalize accelerometer data
def normalize_accelerometer_data(data):
    """Computes ENMO (Euclidean Norm Minus One) for accelerometer data."""
    norm = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2) - 1
    return np.maximum(norm, 0)  # Negative values set to zero

# Detect peaks in normalized data
def detect_peaks(normalized_data, distance=50, prominence=0.5):
    """Detects step peaks in the normalized accelerometer signal."""
    peaks, _ = find_peaks(normalized_data, distance=distance, prominence=prominence)
    return peaks

# Load ground truth data
def load_ground_truth(file_path):
    """Loads total step count from ground truth CSV."""
    ground_truth = pd.read_csv(file_path)
    total_row = ground_truth[ground_truth["Joint"] == "Total"]
    
    if total_row.empty:
        raise ValueError("The ground truth file does not have a 'Total' row.")

    total_steps = int(total_row.iloc[0]["Detected Steps"])
    return total_steps

# Compare detected steps to ground truth
def compare_with_ground_truth(detected_steps, ground_truth_steps, foot="both"):
    """Compares detected peaks with ground truth steps."""
    print(f"👣 {foot.capitalize()} Foot - Detected Steps: {detected_steps}")
    print(f"🎯 Ground Truth Steps: {ground_truth_steps}")
    
    accuracy = (detected_steps / ground_truth_steps) * 100
    print(f"📊 Accuracy: {accuracy:.2f}%")
    
    return accuracy

# Process all folders
def process_folders(root_dir):
    """Processes all video session folders."""
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):  # Skip non-directories
            continue

        print(f"\n📂 Processing folder: {folder_name}")

        # Paths to accelerometer data (left & right) and ground truth
        left_accel_file = os.path.join(folder_path, f"{folder_name}_left_acceleration_data.csv")
        right_accel_file = os.path.join(folder_path, f"{folder_name}_right_acceleration_data.csv")
        ground_truth_file = os.path.join(folder_path, "scaled_step_counts.csv")

        # Check if files exist
        if not os.path.exists(left_accel_file) or not os.path.exists(right_accel_file):
            print(f"❌ Missing accelerometer files in {folder_name}. Skipping.")
            continue
        if not os.path.exists(ground_truth_file):
            print(f"❌ Missing ground truth file in {folder_name}. Skipping.")
            continue

        # Load and process left & right foot accelerometer data
        left_data = load_accelerometer_data(left_accel_file)
        right_data = load_accelerometer_data(right_accel_file)
        left_norm = normalize_accelerometer_data(left_data)
        right_norm = normalize_accelerometer_data(right_data)

        # Detect peaks for both feet
        left_peaks = detect_peaks(left_norm)
        right_peaks = detect_peaks(right_norm)

        detected_steps_left = len(left_peaks)
        detected_steps_right = len(right_peaks)
        detected_steps_total = detected_steps_left + detected_steps_right

        # Load ground truth
        ground_truth_steps = load_ground_truth(ground_truth_file)

        # Compare results
        compare_with_ground_truth(detected_steps_left, ground_truth_steps // 2, "left")
        compare_with_ground_truth(detected_steps_right, ground_truth_steps // 2, "right")
        compare_with_ground_truth(detected_steps_total, ground_truth_steps, "both")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(left_norm, label="Left Foot ENMO", color="blue", alpha=0.7)
        plt.plot(right_norm, label="Right Foot ENMO", color="red", alpha=0.7)
        
        plt.scatter(left_peaks, left_norm[left_peaks], color="green", label="Left Foot Peaks", marker="x")
        plt.scatter(right_peaks, right_norm[right_peaks], color="orange", label="Right Foot Peaks", marker="o")

        plt.title(f"Step Detection for {folder_name}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Acceleration Magnitude (ENMO)")
        plt.legend()
        plt.show()

# Main function
def main():
    root_dir = "D:\\Step-counter\\Output"  # Adjust path to dataset directory
    process_folders(root_dir)

if __name__ == "__main__":
    main()
