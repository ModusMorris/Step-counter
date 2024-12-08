import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def process_video(video_path, display_video=False):
    """
    Processes a video file to extract joint motion data and metadata, optionally displaying the video.

    Parameters:
        video_path (str): Path to the video file.
        display_video (bool): Whether to display the video with landmarks.

    Returns:
        tuple: Metadata (dict), joint data (dict), smoothed data (dict), step counts (dict).
    """
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' does not exist.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file '{video_path}'.")
        return None

    pose = mp_pose.Pose()
    joints_data = {
        joint: []
        for joint in ["right_ankle", "left_ankle", "right_heel", "left_heel", "right_foot_index", "left_foot_index"]
    }

    # Extract metadata
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    duration = round(frame_count / fps, 2)
    creation_time = datetime.fromtimestamp(os.path.getctime(video_path)).strftime("%Y-%m-%d %H:%M:%S")

    # Process video frame by frame
    with tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame for pose landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Collect joint data
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for joint in joints_data.keys():
                    joint_landmark = getattr(mp_pose.PoseLandmark, joint.upper())
                    joints_data[joint].append(landmarks[joint_landmark].x)

                # Optionally display the video with skeleton overlay
                if display_video:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    resized_frame = cv2.resize(frame, (720, 480))
                    cv2.imshow("Pose Tracking - Data Collection", resized_frame)

                    # Exit on 'q' key press
                    if cv2.waitKey(5) & 0xFF == ord("q"):
                        break

            progress.update(1)  # Update the progress bar

    cap.release()
    if display_video:
        cv2.destroyAllWindows()

    # Analyze data and detect steps
    step_counts_joint, smoothed_data, peaks_data, num_steps = count_steps(joints_data)
    metadata = {
        "resolution": resolution,
        "fps": fps,
        "duration_seconds": duration,
        "creation_time": creation_time,
        "num_steps": num_steps,
    }

    return metadata, joints_data, smoothed_data, peaks_data, step_counts_joint


def count_steps(joints_data):
    """
    Dynamically calculates steps from joint motion with adaptive prominence.

    Parameters:
        joints_data (dict): Raw joint movement data.

    Returns:
        tuple: step_counts_joint (dict), smoothed_data (dict), peaks_data (dict), num_steps (int).
    """
    step_counts_joint = {}
    smoothed_data = {}
    peaks_data = {}

    for joint, data in joints_data.items():
        data = np.array(data)

        # Smooth the data using a moving average
        smoothed = np.convolve(data, np.ones(15) / 15, mode="valid")
        smoothed_data[joint] = smoothed

        # Calculate dynamic prominence based on standard deviation
        std_dev = np.std(smoothed)
        prominence = 0.85 * std_dev  # Adjust factor as needed

        # Detect peaks with dynamic prominence
        peaks, _ = find_peaks(smoothed, distance=30, prominence=prominence)
        peaks_data[joint] = peaks
        step_counts_joint[joint] = len(peaks)

    # Calculate total steps
    right_steps = step_counts_joint.get("right_foot_index", 0)
    left_steps = step_counts_joint.get("left_foot_index", 0)
    num_steps = right_steps + left_steps

    print(f"Step Counts:")
    print(f"  Right Steps: {right_steps}")
    print(f"  Left Steps: {left_steps}")
    print(f"  Total Steps: {num_steps}")

    return step_counts_joint, smoothed_data, peaks_data, num_steps


def save_step_data_to_csv(output_folder, joints_data, smoothed_data, peaks_data, step_counts_joint):
    """
    Saves raw data, smoothed data, and step counts to separate CSV files.
    Only uses 'left_foot_index' and 'right_foot_index' for total steps.

    Parameters:
        output_folder (str): Target folder for CSV files.
        joints_data (dict): Raw joint movement data.
        smoothed_data (dict): Smoothed joint movement data.
        peaks_data (dict): Detected peaks (steps) for each joint.
        step_counts_joint (dict): Step counts for each joint.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Save raw data
    raw_data_csv = os.path.join(output_folder, "raw_data.csv")
    with open(raw_data_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame"] + list(joints_data.keys()))  # Header
        max_frames = max(len(data) for data in joints_data.values())
        for frame in range(max_frames):
            row = [frame] + [
                joints_data[joint][frame] if frame < len(joints_data[joint]) else None for joint in joints_data
            ]
            writer.writerow(row)
    print(f"Raw data saved to: {raw_data_csv}")

    # Save smoothed data
    smoothed_data_csv = os.path.join(output_folder, "smoothed_data.csv")
    with open(smoothed_data_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame"] + list(smoothed_data.keys()))  # Header
        max_frames = max(len(data) for data in smoothed_data.values())
        for frame in range(max_frames):
            row = [frame] + [
                smoothed_data[joint][frame] if frame < len(smoothed_data[joint]) else None for joint in smoothed_data
            ]
            writer.writerow(row)
    print(f"Smoothed data saved to: {smoothed_data_csv}")

    # Save step counts
    steps_csv = os.path.join(output_folder, "step_counts.csv")
    with open(steps_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Joint", "Detected Steps", "Peaks"])  # Header

        for joint, steps in step_counts_joint.items():
            peaks = peaks_data[joint]
            writer.writerow([joint, steps, list(peaks)])

        # Compute total steps
        left_steps = len(peaks_data.get("left_foot_index", []))
        right_steps = len(peaks_data.get("right_foot_index", []))
        total_steps = left_steps + right_steps

        # Write total steps
        writer.writerow(["left_foot_index", left_steps, list(peaks_data.get("left_foot_index", []))])
        writer.writerow(["right_foot_index", right_steps, list(peaks_data.get("right_foot_index", []))])
        writer.writerow(["Total", total_steps, ""])

    print(f"Step counts saved to: {steps_csv}")
