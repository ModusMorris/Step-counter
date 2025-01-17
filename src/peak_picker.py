import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load accelerometer data
def load_accelerometer_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Normalize accelerometer data
def normalize_accelerometer_data(data):
    norm = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2) - data['Z']
    return norm.abs()

# Detect peaks in normalized data
def detect_peaks(normalized_data, distance=50, prominence=0.5):
    peaks, _ = find_peaks(normalized_data, distance=distance, prominence=prominence)
    return peaks

# Load ground truth data
def load_ground_truth(file_path):
    """
    Loads the total number of steps from the 'Total' row in the ground truth file.

    Parameters:
        file_path (str): Path to the ground truth CSV file.

    Returns:
        int: Total number of steps from the ground truth file.
    """
    ground_truth = pd.read_csv(file_path)
    print(f"Available columns in {file_path}: {ground_truth.columns}")

    # Extract the 'Total' row
    total_row = ground_truth[ground_truth['Joint'] == 'Total']
    if total_row.empty:
        raise ValueError("The ground truth file does not have a 'Total' row.")
    
    # Get the total steps
    total_steps = int(total_row.iloc[0]['Detected Steps'])
    print(f"Total steps from ground truth: {total_steps}")
    return total_steps


# Compare detected peaks to ground truth
def compare_with_ground_truth(detected_steps, ground_truth_steps):
    print(f"Detected Steps: {detected_steps}")
    print(f"Ground Truth Steps: {ground_truth_steps}")
    accuracy = detected_steps / ground_truth_steps * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Process all folders
def process_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):  # Process only directories
            print(f"\nProcessing folder: {folder_name}")

            # Paths to accelerometer data and ground truth
            accel_file = os.path.join(folder_path, f"{folder_name}_acceleration_data.csv")
            ground_truth_file = os.path.join(folder_path, "step_counts.csv")

            # Check if both files exist
            if not os.path.exists(accel_file):
                print(f"Missing accelerometer data file: {accel_file}")
                continue
            if not os.path.exists(ground_truth_file):
                print(f"Missing ground truth file: {ground_truth_file}")
                continue

            # Load and process data
            accel_data = load_accelerometer_data(accel_file)
            norm_data = normalize_accelerometer_data(accel_data)

            # Detect peaks
            detected_peaks = detect_peaks(norm_data)
            detected_steps = len(detected_peaks)

            # Load ground truth
            ground_truth_steps = load_ground_truth(ground_truth_file)

            # Compare detected steps with ground truth
            compare_with_ground_truth(detected_steps, ground_truth_steps)

            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(norm_data, label='Normalized Accelerometer Data (EMNO)', alpha=0.7)
            plt.scatter(detected_peaks, norm_data[detected_peaks], color='red', label='Detected Peaks')
            plt.title(f"Step Detection for Video {folder_name}")
            plt.xlabel("Time (samples)")
            plt.ylabel("Acceleration Magnitude")
            plt.legend()
            plt.show()

# Main function
def main():
    root_dir = "D:\\Step-counter\\Output"  # Adjust this path to your root directory
    process_folders(root_dir)

if __name__ == "__main__":
    main()
