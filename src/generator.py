import os
import csv
from tqdm import tqdm


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
        # If the CSV file does not exist, treat it as empty
        return False

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["filename"] == video_filename:
                return True
    return False


def count_videos_in_directory(root_dir):
    """
    Counts the total number of video files in the directory and its subdirectories.

    Parameters:
        root_dir (str): Path to the root directory.

    Returns:
        int: Total number of video files found.
    """
    if not root_dir or not os.path.isdir(root_dir):
        print("Error: Invalid root directory specified.")
        return 0

    video_extensions = {".mp4"}
    count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in video_extensions:  # Check if the file is a video
                count += 1
    return count


def video_file_generator(root_dir, metadata_csv):
    """
    Generator that iterates through all video files in a directory and its subdirectories,
    skipping files already listed in the metadata CSV, and showing a progress bar.

    Parameters:
        root_dir (str): Path to the root directory.
        metadata_csv (str): Path to the metadata CSV file.

    Yields:
        str: Path to the next video file that is not in the CSV.
    """
    if not root_dir or not os.path.isdir(root_dir):
        print("Error: Invalid root directory specified.")
        return

    video_extensions = {".mp4"}
    total_videos = count_videos_in_directory(root_dir)
    skipped = 0

    # Use tqdm to display the progress
    with tqdm(total=total_videos, desc="Processing videos", unit="file") as progress:
        for dirpath, _, filenames in os.walk(root_dir):
            # Filter video files once, early
            video_files = [
                os.path.join(dirpath, f) for f in filenames if os.path.splitext(f)[1].lower() in video_extensions
            ]

            for video_path in video_files:
                video_filename = os.path.basename(video_path)

                # Check if the video is already in the CSV
                if is_video_in_csv(metadata_csv, video_filename):
                    print(f"Skipping: {video_filename} (already in metadata).")
                    skipped += 1
                    progress.update(1)
                    continue

                yield video_path
                progress.update(1)

    print(f"Skipped {skipped} videos (already in metadata).")
