import os
import csv
import pandas as pd
from video_processing import process_video


def save_metadata(metadata, video_path, metadata_csv, annotation_file=None):
    """
    Saves metadata of a video to a CSV file and optionally checks the step count with annotations.

    Parameters:
        video_path (str): Path to the video file.
        metadata_csv (str): Path to the central metadata file.
        annotation_file (str): Path to the Excel file with manual step annotations (optional).
    """
    # Initialize or load summary from a global variable
    if not hasattr(save_metadata, "summary"):
        save_metadata.summary = {"matches": 0, "non_matches": [], "no_annotations": []}

    summary = save_metadata.summary

    # Check if all required metadata is present
    required_keys = ["resolution", "fps", "duration_seconds", "creation_time", "num_steps"]
    if not all(key in metadata for key in required_keys):
        print(f"Incomplete metadata for '{video_path}'. Skipping.")
        return

    # Extract the calculated steps
    calculated_steps = metadata["num_steps"]

    # Check if annotations exist, if an annotation file is provided
    if annotation_file:
        if not os.path.exists(annotation_file):
            print(f"Annotation file '{annotation_file}' does not exist. Please create it.")
            return

        # Load the annotations
        annotations = pd.read_excel(annotation_file)

        # Extract the base name of the video
        video_filename = os.path.basename(video_path)

        # Check if annotations exist for the video
        if video_filename in annotations["filename"].values:
            # Get the manual step count
            manual_steps = annotations.loc[annotations["filename"] == video_filename, "manual_steps"].iloc[0]

            # Compare calculated steps with manual steps
            if calculated_steps == manual_steps:
                print(
                    f"Step counts match for '{video_filename}' (Manual: {manual_steps}, Calculated: {calculated_steps})."
                )
                summary["matches"] += 1
            else:
                print(
                    f"Step counts do NOT match for '{video_filename}' (Manual: {manual_steps}, Calculated: {calculated_steps}). Skipping."
                )
                summary["non_matches"].append(
                    {"filename": video_filename, "manual_steps": manual_steps, "calculated_steps": calculated_steps}
                )
                return  # Do not save if steps do not match
        else:
            print(f"No annotation found for '{video_filename}'. Processing with calculated steps.")

    # Create CSV file if it does not exist
    if not os.path.exists(metadata_csv):
        with open(metadata_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "resolution", "fps", "duration_seconds", "creation_time", "num_steps"])

    # Write metadata to CSV
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
        print(f"Metadata for '{video_path}' saved to '{metadata_csv}'.")
    else:
        print(f"Metadata for '{video_path}' is already in '{metadata_csv}'. Skipping.")


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
    Prints the summary of matching and non-matching annotations.

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
