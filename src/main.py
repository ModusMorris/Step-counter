import argparse
import os
from metadata_handler import save_metadata, print_summary
from generator import video_file_generator
from video_processing import process_video, visualize_data, save_step_data_to_csv


def process_and_visualize_video(video_path, output_root, metadata_csv, annotation_file=None):
    """
    Processes a single video, saves metadata, and generates results (PDF, CSVs) in a dedicated folder.

    Parameters:
        video_path (str): Path to the video file.
        output_root (str): Root directory where the video folder will be created.
        metadata_csv (str): Path to the central metadata CSV file.
        annotation_file (str): Path to the Excel file with manual annotations (optional).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_root, video_name)
    os.makedirs(video_folder, exist_ok=True)

    print(f"Processing video: {video_path}")

    # Save metadata
    save_metadata(video_path, metadata_csv, annotation_file if annotation_file else None)

    # Process and visualize video
    result = process_video(video_path, display_video=False)
    if not result:
        print(f"Failed to process {video_path}")
        return

    # Unpack results
    metadata, joints_data, smoothed_data, peaks_data, step_counts_joint = result

    # Create and save the PDF
    pdf_path = os.path.join(video_folder, f"{video_name}.pdf")
    visualize_data(joints_data, smoothed_data, peaks_data, output_path=pdf_path)

    # Save CSV files
    save_step_data_to_csv(video_folder, joints_data, smoothed_data, peaks_data, step_counts_joint)

    print(f"Processing complete. Results saved in folder: {video_folder}")


def process_all_videos_in_directory(root_dir, output_root, annotation_file=None):
    """
    Scans a directory for videos and processes all unprocessed ones.
    Saves metadata and results for each video.

    Parameters:
        root_dir (str): Directory containing videos.
        output_root (str): Target directory for all results.
        annotation_file (str): Path to the Excel file with manual annotations (optional).
    """
    metadata_csv = os.path.join(output_root, "metadata.csv")
    for video_path in video_file_generator(root_dir, metadata_csv):
        process_and_visualize_video(video_path, output_root, metadata_csv, annotation_file)
    print("All videos processed.")


def main():
    """
    Main function for video processing.
    """
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument(
        "--action",
        type=str,
        choices=["save_metadata", "visualize_data"],
        required=True,
        help="Action to perform: 'save_metadata' or 'visualize_data'.",
    )
    parser.add_argument(
        "--root_dir", type=str, help="Directory containing videos (required for 'save_metadata' and 'visualize_data')."
    )
    parser.add_argument("--video_path", type=str, help="Path to a specific video (only for 'visualize_data').")
    parser.add_argument("--output_root", type=str, default="output", help="Root directory for output files.")
    parser.add_argument(
        "--annotation_file", type=str, required=False, help="Path to the Excel file with manual annotations."
    )

    args = parser.parse_args()

    if args.action == "visualize_data":
        if args.video_path:
            process_and_visualize_video(
                args.video_path, args.output_root, os.path.join(args.output_root, "metadata.csv"), args.annotation_file
            )
        elif args.root_dir:
            process_all_videos_in_directory(args.root_dir, args.output_root, args.annotation_file)
        else:
            print("Error: Please provide either '--video_path' or '--root_dir'.")
    elif args.action == "save_metadata":
        if not args.root_dir:
            print("Error: '--root_dir' is required for 'save_metadata'.")
            return

        metadata_csv = os.path.join(args.output_root, "metadata.csv")
        annotation_file = args.annotation_file or os.path.join(args.root_dir, "step_annotations.xlsx")
        for video_path in video_file_generator(args.root_dir, metadata_csv):
            save_metadata(video_path, metadata_csv, annotation_file)
        print_summary()
    else:
        print("Invalid action. Please choose 'save_metadata' or 'visualize_data'.")


if __name__ == "__main__":
    main()
