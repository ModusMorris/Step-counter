import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from clap_detection_methods import detect_claps_first_last_segments

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def process_video(video_path, num_segments=4, display_video=False):
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
    joints_data = {joint: [] for joint in ["right_ankle", "left_ankle", "right_heel", "left_heel", "right_foot_index", "left_foot_index"]}

    # Metadata placeholders
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    duration = round(frame_count / fps, 2)
    creation_time = datetime.fromtimestamp(os.path.getctime(video_path)).strftime("%Y-%m-%d %H:%M:%S")

    # NEEEW

    clip = VideoFileClip(video_path)
    audio = clip.audio

    if audio is None:
        print("  No audio track found.")
        return None
    
    audio_array = audio.to_soundarray()
    claps = detect_claps_first_last_segments(audio_array, fps=fps, num_segments=num_segments)

    if not claps or len(claps) < 2:
        print("  Not enough claps detected for synchronization.")
        return None
    
    start_clap_time = claps[0][0]
    end_clap_time = claps[1][0]
    start_clap_frame = claps[0][1]
    end_clap_frame = claps[1][1]

    current_frame = 0

    # Progress bar for processing frames
    with tqdm(total=(end_clap_frame-start_clap_frame), desc=f"Processing {os.path.basename(video_path)}", unit="frame") as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or current_frame > end_frame:
                break

            if current_frame < start_frame:
                current_frame += 1
                continue

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
            current_frame += 1
            progress.update(1)  # Update the progress bar after each frame is processed

    cap.release()
    if display_video:
        cv2.destroyAllWindows()

    step_counts_joint, smoothed_data, peaks_data, num_steps = count_steps(joints_data)
    metadata = {
        "resolution": resolution,
        "fps": fps,
        "duration_seconds": duration,
        "creation_time": creation_time,
        "start_clap_frame": start_clap_frame,
        "end_clap_frame": end_clap_frame,
        "num_steps": num_steps
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
    right_steps = step_counts_joint.get("right_ankle", 0)
    left_steps = step_counts_joint.get("left_ankle", 0)
    num_steps = right_steps + left_steps

    print(f"Step Counts:")
    print(f"  Right Steps: {right_steps}")
    print(f"  Left Steps: {left_steps}")
    print(f"  Total Steps: {num_steps}")

    return step_counts_joint, smoothed_data, peaks_data, num_steps



def visualize_data(joints_data, smoothed_data, peaks_data):
    """
    Visualizes joint motion and detected steps, using raw data, smoothed data, and peak indices.

    Parameters:
        joints_data (dict): Raw joint movement data.
        smoothed_data (dict): Smoothed joint movement data.
        peaks_data (dict): Detected step (peak) indices for each joint.
    """
    # Plot joint motion data
    fig, axs = plt.subplots(len(joints_data), 1, figsize=(15, 20))
    joints = list(joints_data.keys())
    colors = ["orange", "green", "blue", "red", "purple", "brown"]

    for i, joint in enumerate(joints):
        raw_data = joints_data[joint]
        smoothed = smoothed_data[joint]
        peaks = peaks_data[joint]

        # Plot raw and smoothed data
        axs[i].plot(raw_data, label=f"{joint} (raw)", alpha=0.5, color=colors[i])
        axs[i].plot(smoothed, label=f"{joint} (smoothed)", linewidth=2, color=colors[i])
        
        # Highlight detected steps (peaks)
        axs[i].scatter(peaks, smoothed[peaks], color="black", label=f"Detected Steps ({len(peaks)})")
        axs[i].legend()
        axs[i].set_title(f"{joint} Horizontal Movement and Step Detection")
        axs[i].set_xlabel("Frames")
        axs[i].set_ylabel("Horizontal Position (x)")

    plt.tight_layout()
    plt.show()

