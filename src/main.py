import argparse
import os
from metadata_handler import save_metadata, print_summary
from generator import video_file_generator
from video_processing import process_video, visualize_data

def main():
    """
    Main function to process videos and perform actions based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument("--action", type=str, choices=["save_metadata", "visualize_data"], required=True,
                        help="Action to perform: 'save_metadata' or 'visualize_data'.")
    parser.add_argument("--root_dir", type=str,
                        help="Path to the root directory containing videos (required for 'save_metadata').")
    parser.add_argument("--video_path", type=str,
                        help="Path to a specific video file (required for 'visualize_data').")
    parser.add_argument("--annotation_file", type=str, required=False,
                        help="Path to the Excel file containing manual step annotations (optional).")

    args = parser.parse_args()

    if args.action == "save_metadata":
        if not args.root_dir:
            print("Error: You must specify the --root_dir argument for saving metadata.")
            return

        # Use the root directory as annotation file path if none is provided
        annotation_file = args.annotation_file or os.path.join(args.root_dir, "step_annotations.xlsx")

        # Define the central metadata.csv path
        metadata_csv = os.path.join(args.root_dir, "metadata.csv")

        # Process new videos
        for video_path in video_file_generator(args.root_dir, metadata_csv):
            print(f"Processing: {video_path}")
            save_metadata(video_path, metadata_csv, annotation_file)

        # Print summary at the end
        print_summary()

    elif args.action == "visualize_data":
        if not args.video_path:
            print("Error: You must specify the --video_path argument for visualizing data.")
            return

        video_path = args.video_path
        print(f"Processing video for visualization and data collection: {video_path}")

        # Extract video name for PDF naming
        video_name = os.path.basename(video_path).split('.')[0]
        pdf_folder = "PDF"
        os.makedirs(pdf_folder, exist_ok=True)
        pdf_path = os.path.join(pdf_folder, f"{video_name}.pdf")

        # Process the specified video for visualization and data collection
        result = process_video(video_path, display_video=True)
        if not result:
            print(f"Failed to process video '{video_path}'.")
            return

        # Unpack results
        metadata, joints_data, smoothed_data, peaks_data, step_counts_joint = result

        # Visualize the joint motion using the precomputed data and save to PDF
        visualize_data(joints_data, smoothed_data, peaks_data, output_path=pdf_path)
    else:
        print("Unsupported action.")

if __name__ == "__main__":
    main()
