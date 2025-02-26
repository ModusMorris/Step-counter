import os
import pandas as pd
import numpy as np
from clap_detection_methods import load_accelerometer_data


def slice_accelerometer_data(metadata_csv, raw_accel_data_dir, output_root):
    """
    Slice accelerometer data for each video and save the slices to CSV files.

    Reads a metadata CSV containing video IDs, filenames for left/right watches,
    and start/end indices. For each watch file, loads the accelerometer data,
    slices it, saves the slice, and records the slice length for later scaling.

    Parameters:
        metadata_csv (str): Path to the metadata CSV.
        raw_accel_data_dir (str): Directory with raw accelerometer CSV files.
        output_root (str): Directory where sliced data will be saved.

    Returns:
        list: Tuples of (length of sliced data, video folder path) for scaling purposes.
    """
    df = pd.read_csv(metadata_csv)
    scaling_data = []

    for _, row in df.iterrows():
        video_id = row["video_id"]
        watch_left = row["watch_left"]
        watch_right = row["watch_right"]
        start_idx = int(row["start_clap_frame"])
        end_idx = int(row["end_clap_frame"])

        for watch, side in zip([watch_left, watch_right], ["left", "right"]):
            acc_data_path = os.path.join(raw_accel_data_dir, f"{watch}.csv")

            if not os.path.exists(acc_data_path):
                print(f"File not found: {acc_data_path}")
                continue

            cleaned_data, _ = load_accelerometer_data(acc_data_path)
            sliced_data = cleaned_data.loc[start_idx:end_idx]
            video_folder = os.path.join(output_root, video_id)
            os.makedirs(video_folder, exist_ok=True)
            output_filename = f"{video_id}_{side}_acceleration_data.csv"
            output_path = os.path.join(video_folder, output_filename)

            scaling_data.append((len(sliced_data), video_folder))
            sliced_data.to_csv(output_path, index=False)
            print(f"Sliced data for {side} watch saved to: {output_path}")

    return scaling_data


def scale_stepcounts_data(scaling_data):
    """
    Scale step counts based on the length of the sliced accelerometer data.

    For each video folder, reads the step count data and raw video data to compute a
    scaling factor. Then updates the "Peaks" values in the step counts by applying this factor,
    and saves the scaled data.

    Parameters:
        scaling_data (list): List of tuples (length of sliced data, video folder path).
    """
    for len_acc_data, video_folder in scaling_data:
        data_path = os.path.join(video_folder, "step_counts.csv")
        data = pd.read_csv(data_path)
        len_frames_video = len(pd.read_csv(os.path.join(video_folder, "raw_data.csv")))
        scaling_factor = len_acc_data / len_frames_video

        for index, row in data.iterrows():
            val = row["Peaks"]
            if pd.isna(val):
                continue
            val_str = str(val)
            if "[" not in val_str:
                continue
            peaks_str = val_str.strip("[]")
            peaks = [int(p.strip()) for p in peaks_str.split(",")]
            scaled_peaks = [int(p * scaling_factor) for p in peaks]
            data.at[index, "Peaks"] = str(scaled_peaks)

        scaled_step_counts_path = os.path.join(video_folder, "scaled_step_counts.csv")
        data.to_csv(scaled_step_counts_path, index=False)


if __name__ == "__main__":
    metadata_csv_path = os.path.join(os.getcwd(), r"Data\acceleration_metadata.csv")
    raw_accel_data_dir = os.path.join(os.getcwd(), r"Data\accelerometer_data")
    output_root = os.path.join(os.getcwd(), r"Data\real_output")

    scaling_data = slice_accelerometer_data(
        metadata_csv=metadata_csv_path, raw_accel_data_dir=raw_accel_data_dir, output_root=output_root
    )
    scale_stepcounts_data(scaling_data)
