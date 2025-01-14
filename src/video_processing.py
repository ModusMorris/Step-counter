import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from moviepy import VideoFileClip
from clap_detection_methods import detect_claps_first_last_segments


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
    joints_data = {joint: [] for joint in ["right_ankle", "left_ankle", "right_heel", "left_heel", "right_foot_index", "left_foot_index"]}

    # Metadata placeholders
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    duration = round(frame_count / fps, 2)
    creation_time = datetime.fromtimestamp(os.path.getctime(video_path)).strftime("%Y-%m-%d %H:%M:%S")

    # Progress bar for processing frames
    with tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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

            progress.update(1)  # Update the progress bar after each frame is processed

    cap.release()
    if display_video:
        cv2.destroyAllWindows()

    ###NEW CODE###
    # Load the video audio file
    clip = VideoFileClip(video_path)
    if not clip.audio:
            print("No audio track found; using full video range")
    else:
        audio_array = clip.audio.to_soundarray()
        audio_fps = clip.audio.fps

        # Detect the two claps (start & end)
        claps = detect_claps_first_last_segments(audio_array, fps=audio_fps, num_segments=2)

        first_clap_time_sec = claps[0][0]
        second_clap_time_sec = claps[1][0]

        # Convert these times to the videos frames
        first_clap_frame = int(first_clap_time_sec * fps)
        second_clap_frame = int(second_clap_time_sec * fps)


        # Slice data in place
        for joint in joints_data:
            joints_data[joint] = joints_data[joint][first_clap_frame: second_clap_frame + 1]

        # update duration?
        #new_frame_count = len(joints_data["right_ankle"])  # or any joint
        #duration = round(new_frame_count / fps, 2)

        print(f"Sliced data to frames {first_clap_frame}–{second_clap_frame} (≈ {duration} sec).")

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
    right_steps = step_counts_joint.get("right_ankle", 0)
    left_steps = step_counts_joint.get("left_ankle", 0)
    num_steps = right_steps + left_steps

    print(f"Step Counts:")
    print(f"  Right Steps: {right_steps}")
    print(f"  Left Steps: {left_steps}")
    print(f"  Total Steps: {num_steps}")

    return step_counts_joint, smoothed_data, peaks_data, num_steps



def visualize_data(joints_data, smoothed_data, peaks_data, output_path=None):
    """
    Visualizes joint motion and detected steps, using raw data, smoothed data, and peak indices.
    Optionally saves the plots to a PDF file.

    Parameters:
        joints_data (dict): Raw joint movement data.
        smoothed_data (dict): Smoothed joint movement data.
        peaks_data (dict): Detected step (peak) indices for each joint.
        output_path (str): Path to save the PDF file (optional).
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

    # Save to PDF if output_path is provided
    if output_path:
        pdf_path = output_path if output_path.endswith(".pdf") else f"{output_path}.pdf"
        plt.savefig(pdf_path)
        print(f"Plots saved to {pdf_path}")
    else:
        plt.show()

    plt.close(fig)

def save_step_data_to_csv(output_folder, joints_data, smoothed_data, peaks_data, step_counts_joint):
    """
    Speichert Rohdaten, geglättete Daten und Schrittzählungen in separaten CSV-Dateien.
    Berechnet die Gesamtanzahl der Schritte nur basierend auf 'left_foot_index' und 'right_foot_index'.

    Parameters:
        output_folder (str): Zielordner für die CSV-Dateien.
        joints_data (dict): Rohdaten der Gelenkbewegungen.
        smoothed_data (dict): Geglättete Daten der Gelenkbewegungen.
        peaks_data (dict): Detektierte Peaks (Schritte) pro Gelenk.
        step_counts_joint (dict): Zählung der Schritte für jedes Gelenk.
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1. Rohdaten speichern
    raw_data_csv = os.path.join(output_folder, "raw_data.csv")
    with open(raw_data_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame"] + list(joints_data.keys()))  # Header
        max_frames = max(len(data) for data in joints_data.values())
        for frame in range(max_frames):
            row = [frame] + [joints_data[joint][frame] if frame < len(joints_data[joint]) else None
                             for joint in joints_data]
            writer.writerow(row)
    print(f"Rohdaten gespeichert: {raw_data_csv}")

    # 2. Geglättete Daten speichern
    smoothed_data_csv = os.path.join(output_folder, "smoothed_data.csv")
    with open(smoothed_data_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame"] + list(smoothed_data.keys()))  # Header
        max_frames = max(len(data) for data in smoothed_data.values())
        for frame in range(max_frames):
            row = [frame] + [smoothed_data[joint][frame] if frame < len(smoothed_data[joint]) else None
                             for joint in smoothed_data]
            writer.writerow(row)
    print(f"Geglättete Daten gespeichert: {smoothed_data_csv}")

    # 3. Schrittzählung speichern
    steps_csv = os.path.join(output_folder, "step_counts.csv")
    with open(steps_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Joint", "Detected Steps", "Peaks"])  # Header

        # Schreibe die Schrittzählungen für alle Gelenke außer den Total Steps
        for joint, steps in step_counts_joint.items():
            if joint not in ["left_foot_index", "right_foot_index"]:  # Ignoriere diese Gelenke vorerst
                peaks = peaks_data[joint]
                writer.writerow([joint, steps, list(peaks)])  # Fügt die Peaks mit hinzu

        # Berechnung der Total Steps basierend auf 'left_foot_index' und 'right_foot_index'
        left_steps = len(peaks_data.get("left_foot_index", []))
        right_steps = len(peaks_data.get("right_foot_index", []))
        total_steps = left_steps + right_steps

        # Schreibe die Einträge für 'left_foot_index', 'right_foot_index' und die Total Steps
        writer.writerow(["left_foot_index", left_steps, list(peaks_data.get("left_foot_index", []))])
        writer.writerow(["right_foot_index", right_steps, list(peaks_data.get("right_foot_index", []))])
        writer.writerow(["Total", total_steps, ""])

    print(f"Schrittzählungen gespeichert: {steps_csv}")