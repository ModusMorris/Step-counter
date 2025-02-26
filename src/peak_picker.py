import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def load_accelerometer_data(file_path):
    """
    Load accelerometer data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing accelerometer data.

    Returns:
        DataFrame: The loaded accelerometer data.
    """
    return pd.read_csv(file_path)


def normalize_accelerometer_data(data):
    """
    Compute the ENMO (Euclidean Norm Minus One) for accelerometer data.

    This function calculates the Euclidean norm of the X, Y, and Z columns,
    subtracts 1 from the result, and sets any negative values to zero.

    Parameters:
        data (DataFrame): Accelerometer data with columns "X", "Y", and "Z".

    Returns:
        np.ndarray: The normalized acceleration values.
    """
    norm = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2) - 1
    return np.maximum(norm, 0)


def detect_peaks(normalized_data, distance=50, prominence=0.5):
    """
    Detect peaks in the normalized accelerometer data.

    Uses the scipy.signal.find_peaks function to locate peaks based on the
    provided distance and prominence thresholds.

    Parameters:
        normalized_data (array-like): Normalized accelerometer signal.
        distance (int): Minimum distance between peaks (default is 50 samples).
        prominence (float): Required prominence of peaks (default is 0.5).

    Returns:
        ndarray: Indices of the detected peaks.
    """
    peaks, _ = find_peaks(normalized_data, distance=distance, prominence=prominence)
    return peaks


def load_ground_truth(file_path):
    """
    Load the total step count from a ground truth CSV file.

    Expects a row labeled "Total" in the "Joint" column that contains the total
    detected steps.

    Parameters:
        file_path (str): Path to the ground truth CSV file.

    Returns:
        int: The total number of detected steps.

    Raises:
        ValueError: If the "Total" row is not found in the CSV.
    """
    ground_truth = pd.read_csv(file_path)
    total_row = ground_truth[ground_truth["Joint"] == "Total"]

    if total_row.empty:
        raise ValueError("The ground truth file does not have a 'Total' row.")

    total_steps = int(total_row.iloc[0]["Detected Steps"])
    return total_steps


def compare_with_ground_truth(detected_steps, ground_truth_steps, foot="both"):
    """
    Compare the detected step count with the ground truth.

    Prints the number of detected steps, ground truth steps, and calculates the
    accuracy percentage.

    Parameters:
        detected_steps (int): The number of steps detected.
        ground_truth_steps (int): The ground truth step count.
        foot (str): Indicates which foot's data is being compared ("left", "right", or "both").

    Returns:
        float: The calculated accuracy as a percentage.
    """
    print(f"{foot.capitalize()} Foot - Detected Steps: {detected_steps}")
    print(f"Ground Truth Steps: {ground_truth_steps}")

    accuracy = (detected_steps / ground_truth_steps) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def process_folders(root_dir):
    """
    Process all session folders in the root directory for step detection.

    For each folder, this function loads accelerometer data from left and right foot,
    normalizes the data, detects peaks, compares the detected step count with the ground
    truth, and plots the results.

    Parameters:
        root_dir (str): The root directory containing session folders.

    Returns:
        None
    """
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):  # Skip non-directories
            continue

        print(f"\nProcessing folder: {folder_name}")

        # Construct file paths
        left_accel_file = os.path.join(folder_path, f"{folder_name}_left_acceleration_data.csv")
        right_accel_file = os.path.join(folder_path, f"{folder_name}_right_acceleration_data.csv")
        ground_truth_file = os.path.join(folder_path, "scaled_step_counts.csv")

        # Ensure required files exist
        if not os.path.exists(left_accel_file) or not os.path.exists(right_accel_file):
            print(f"Missing accelerometer files in {folder_name}. Skipping.")
            continue
        if not os.path.exists(ground_truth_file):
            print(f"Missing ground truth file in {folder_name}. Skipping.")
            continue

        # Load and process data
        left_data = load_accelerometer_data(left_accel_file)
        right_data = load_accelerometer_data(right_accel_file)
        left_norm = normalize_accelerometer_data(left_data)
        right_norm = normalize_accelerometer_data(right_data)

        left_peaks = detect_peaks(left_norm)
        right_peaks = detect_peaks(right_norm)

        detected_steps_left = len(left_peaks)
        detected_steps_right = len(right_peaks)
        detected_steps_total = detected_steps_left + detected_steps_right

        ground_truth_steps = load_ground_truth(ground_truth_file)

        compare_with_ground_truth(detected_steps_left, ground_truth_steps // 2, "left")
        compare_with_ground_truth(detected_steps_right, ground_truth_steps // 2, "right")
        compare_with_ground_truth(detected_steps_total, ground_truth_steps, "both")

        # Plot the accelerometer signals and detected peaks
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


def main():
    """
    Main function to process session folders for step detection.

    Sets the root directory path for the dataset and initiates processing of all
    session folders.
    """
    root_dir = "D:\\Step-counter\\Output"  # Adjust path to dataset directory
    process_folders(root_dir)


if __name__ == "__main__":
    main()

