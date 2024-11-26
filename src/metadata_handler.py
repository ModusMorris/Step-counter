import os
import csv
from video_processing import process_video


def save_metadata(video_path, metadata_csv):
    """
    Saves metadata of a video file to a single CSV file and prints step counts.

    Parameters:
        video_path (str): Path to the video file.
        metadata_csv (str): Path to the central metadata CSV file.
    """
    # Extract metadata and steps
    result = process_video(video_path, display_video=False)
    if not result:
        print(f"Failed to extract metadata for '{video_path}'.")
        return

    metadata, _, _, step_counts_joint, _ = result  # Unpack the results from process_video
    required_keys = ["resolution", "fps", "duration_seconds", "creation_time", "num_steps"]
    if not all(key in metadata for key in required_keys):
        print(f"Incomplete metadata for '{video_path}'. Skipping.")
        return

    # Create the CSV file if it doesn't exist
    if not os.path.exists(metadata_csv):
        with open(metadata_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "resolution", "fps", "duration_seconds", "creation_time", "num_steps"])

    # Write metadata to CSV
    video_filename = os.path.basename(video_path)
    with open(metadata_csv, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            video_filename,
            metadata["resolution"],
            metadata["fps"],
            metadata["duration_seconds"],
            metadata["creation_time"],
            metadata["num_steps"]
        ])
    print(f"Metadata for '{video_filename}' saved to '{metadata_csv}'.")
