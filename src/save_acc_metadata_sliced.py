import os
import pandas as pd
import numpy as np
from clap_detection_methods import load_accelerometer_data
# Function to slice and save accelerometer data

def slice_accelerometer_data(metadata_csv, raw_accel_data_dir, output_root):

    df = pd.read_csv(metadata_csv)

    for _, row in df.iterrows():
        video_id = row["video_id"]
        watch_filename = row["watch"]
        start_idx = int(row["index_start_clap"])
        end_idx = int(row["index_end_clap"])

        # Build the full path to the raw accelerometer file
        acc_data_path = os.path.join(raw_accel_data_dir, f"{watch_filename}.csv")

        if not os.path.exists(acc_data_path):
            print(f"File not found: {acc_data_path}")
            continue

        # Load and clean the full accelerometer data
        cleaned_data, _ = load_accelerometer_data(acc_data_path)

        # Slice the data from start_idx to end_idx
        sliced_data = cleaned_data.loc[start_idx:end_idx]

        # Create output folder for this video
        video_folder = os.path.join(output_root, video_id)
        os.makedirs(video_folder, exist_ok=True)

        # Define the output filename
        output_filename = f"{video_id}_acceleration_data.csv"
        output_path = os.path.join(video_folder, output_filename)

        # Save sliced data
        sliced_data.to_csv(output_path, index=False)
        print(f"Sliced data saved to: {output_path}")






if __name__ == "__main__":
    metadata_csv_path = r"C:\Users\niki\Desktop\Step-counter\Data\acceleration_metadata.csv"
    raw_accel_data_dir = r"C:\Users\niki\Desktop\Step-counter\Data"
    output_root = r"C:\Users\niki\Desktop\Step-counter\Data\output1"
    
    slice_accelerometer_data(metadata_csv=metadata_csv_path, raw_accel_data_dir=raw_accel_data_dir, output_root=output_root)
