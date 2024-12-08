import os
import csv
import pandas as pd
from video_processing import process_video


def save_metadata(video_path, metadata_csv, annotation_file=None):
    """
    Saves metadata of a video into a CSV file and optionally checks step counts against annotations.

    Parameters:
        video_path (str): Path to the video file.
        metadata_csv (str): Path to the central metadata CSV file.
        annotation_file (str): Path to the Excel file with manual step annotations (optional).
    """
    # Initialize or load the summary
    if not hasattr(save_metadata, "summary"):
        save_metadata.summary = {"matches": 0, "non_matches": [], "no_annotations": []}

    summary = save_metadata.summary

    # Extract metadata and steps
    result = process_video(video_path, display_video=False)
    if not result:
        print(f"Failed to extract metadata for '{video_path}'.")
        return

    metadata, _, _, _, _ = result  # Unpack results from process_video
    required_keys = ["resolution", "fps", "duration_seconds", "creation_time", "num_steps"]
    if not all(key in metadata for key in required_keys):
        print(f"Incomplete metadata for '{video_path}'. Skipping.")
        return

    calculated_steps = metadata["num_steps"]

    # Check annotations if provided
    if annotation_file:
        if not os.path.exists(annotation_file):
            print(f"Annotation file '{annotation_file}' does not exist.")
            return

        annotations = pd.read_excel(annotation_file)
        video_filename = os.path.basename(video_path)

        if video_filename in annotations["filename"].values:
            manual_steps = annotations.loc[annotations["filename"] == video_filename, "manual_steps"].iloc[0]
            if calculated_steps == manual_steps:
                print(f"Steps match for '{video_filename}' (Manual: {manual_steps}, Calculated: {calculated_steps}).")
                summary["matches"] += 1
            else:
                print(
                    f"Steps DO NOT match for '{video_filename}' (Manual: {manual_steps}, Calculated: {calculated_steps})."
                )
                summary["non_matches"].append(
                    {"filename": video_filename, "manual_steps": manual_steps, "calculated_steps": calculated_steps}
                )
                return
        else:
            print(f"No annotation found for '{video_filename}'. Using calculated steps.")

    # Create CSV if it doesn't exist
    if not os.path.exists(metadata_csv):
        with open(metadata_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "resolution", "fps", "duration_seconds", "creation_time", "num_steps"])

    # Save metadata to CSV
    if not is_video_in_csv(metadata_csv, os.path.basename(video_path)):
        with open(metadata_csv, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    os.path.basename(video_path),
                    metadata["resolution"],
                    metadata["fps"],
                    metadata["duration_seconds"],
                    metadata["creation_time"],
                    metadata["num_steps"],
                ]
            )
        print(f"Metadata for '{video_path}' saved in '{metadata_csv}'.")
    else:
        print(f"Metadata for '{video_path}' already exists in '{metadata_csv}'. Skipping.")


def is_video_in_csv(csv_file, video_filename):
    """
    Checks if a video's metadata is already in the CSV file.

    Parameters:
        csv_file (str): Path to the CSV file.
        video_filename (str): Name of the video file.

    Returns:
        bool: True if the video is already in the CSV, False otherwise.
    """
    if not os.path.exists(csv_file):
        return False

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["filename"] == video_filename:
                return True
    return False


def print_summary():
    """
    Prints a summary of matching and non-matching annotations.

    This function should be called at the end of the main processing loop.
    """
    summary = getattr(save_metadata, "summary", {"matches": 0, "non_matches": []})

    print("\n=== Summary ===")
    print(f"Total Matches: {summary['matches']}")
    print(f"Total Non-Matches: {len(summary['non_matches'])}")

    if summary["non_matches"]:
        print("\nNon-Matching Annotations:")
        for item in summary["non_matches"]:
            print(
                f"  - {item['filename']}: Manual Steps = {item['manual_steps']}, Calculated Steps = {item['calculated_steps']}"
            )
