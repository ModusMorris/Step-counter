import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy import VideoFileClip


def plot_audio_waveform_from_array(audio_array, fps, num_segments=4):
    """
    Plot the audio waveform from an array while highlighting the first and last segments.

    Converts stereo audio to mono if needed, creates a time axis from fps, and shades the
    first and last segments based on the total number of segments.

    Parameters:
        audio_array (np.ndarray): Input audio data.
        fps (float): Sampling rate (frames per second).
        num_segments (int): Number of segments to split the audio into.
    """
    # If stereo, convert to mono by averaging channels
    if audio_array.ndim > 1:
        audio_data = np.mean(audio_array, axis=1)
    else:
        audio_data = audio_array

    audio_data = np.abs(audio_data)
    duration = len(audio_data) / fps
    time_axis = np.linspace(0, duration, num=len(audio_data))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio_data, color="steelblue", label="Waveform")

    # Calculate segment length and define segment boundaries
    segment_length = len(audio_data) // num_segments
    first_segment_start_time = 0.0
    first_segment_end_time = segment_length / fps
    last_segment_start_time = (num_segments - 1) * segment_length / fps
    last_segment_end_time = duration

    # Highlight first and last segments
    plt.axvspan(first_segment_start_time, first_segment_end_time, color="red", alpha=0.15, label="Erstes Segment")
    plt.axvspan(last_segment_start_time, last_segment_end_time, color="green", alpha=0.15, label="Letztes Segment")

    plt.title("Audio Waveform with First & Last Segment Shaded")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def detect_claps_first_last_segments(audio_array, fps, num_segments):
    """
    Detect claps by locating the peak amplitude in the first and last segments.

    Converts stereo audio to mono, divides the audio into segments, and finds the maximum
    amplitude (assumed to be a clap) in the first and last segments.

    Parameters:
        audio_array (np.ndarray): Input audio data.
        fps (float): Sampling rate (frames per second).
        num_segments (int): Number of segments to split the audio into.

    Returns:
        list: Tuples containing (clap time in seconds, frame index) for each detected clap.
    """
    # Convert stereo to mono
    audio_data = np.mean(audio_array, axis=1)
    abs_audio = np.abs(audio_data)
    segment_length = len(abs_audio) // num_segments
    claps = []

    # Only process the first and last segments
    segments_to_process = [0, num_segments - 1]
    for segment_index in segments_to_process:
        start_idx = segment_index * segment_length
        end_idx = (segment_index + 1) * segment_length if segment_index < num_segments - 1 else len(abs_audio)
        segment = abs_audio[start_idx:end_idx]

        if len(segment) > 0:
            max_idx = segment.argmax()
            clap_time = (start_idx + max_idx) / float(fps)
            claps.append((clap_time, start_idx + max_idx))

    return claps


def process_videos_in_directory(directory_path, num_segments):
    """
    Process video files by plotting their audio waveforms and detecting claps.

    Iterates through .mp4 files in the specified directory. For each video, it extracts the
    audio track, plots the waveform, detects claps in the first and last segments, and records
    the results.

    Parameters:
        directory_path (str): Path to the folder containing video files.
        num_segments (int): Number of segments to split the audio for clap detection.

    Returns:
        list: Dictionaries with video filename, detected clap times, frame indices, and duration between claps.
    """
    video_extension = ".mp4"
    clap_results = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename.lower())[1] == video_extension:
            print(f"\nProcessing video: {filename}")

            try:
                clip = VideoFileClip(file_path)
                audio = clip.audio
                if audio is None:
                    print("No audio track found.")
                    continue

                fps = audio.fps
                audio_array = audio.to_soundarray()

                plot_audio_waveform_from_array(audio_array, fps)

                # Detect claps in first and last segments
                claps = detect_claps_first_last_segments(audio_array, fps=fps, num_segments=num_segments)

                if claps:
                    for c_time, frame in claps:
                        print(f"  Clap detected at {c_time:.2f} seconds (frame {frame})")

                duration_between_claps = claps[1][0] - claps[0][0]
                clap_results.append({
                    "Filename": filename,
                    "Start Clap Seconds": claps[0][0],
                    "End Clap Seconds": claps[1][0],
                    "Start Clap Frame": claps[0][1],
                    "End Clap Frame": claps[1][1],
                    "Duration Between Claps": duration_between_claps,
                })

            except Exception as e:
                print(f"  Error processing {filename}: {e}")

    return clap_results


def load_accelerometer_data(file_path, sampling_frequency=256):
    """
    Load accelerometer data from a CSV file and create a time axis.

    Reads the CSV (skipping the first 11 rows), converts the data to floats (adjusting for commas),
    and generates a time axis based on the sampling frequency.

    Parameters:
        file_path (str): Path to the accelerometer CSV file.
        sampling_frequency (int): Sampling rate for the accelerometer data.

    Returns:
        tuple: (DataFrame with accelerometer data, time axis as a Series)
    """
    data = pd.read_csv(file_path, delimiter=",", skiprows=11, names=["X", "Y", "Z"], dtype=str)
    data = data.map(lambda x: x.replace(",", ".")).astype(float)
    data.reset_index(drop=True, inplace=True)
    time_seconds = data.index / sampling_frequency

    return data, time_seconds


def normalize_data(data):
    """
    Compute the Euclidean norm of accelerometer data from X, Y, and Z columns.

    Parameters:
        data (DataFrame): Accelerometer data with columns "X", "Y", and "Z".

    Returns:
        np.ndarray: Array of normalized acceleration values.
    """
    return np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2)


def plot_accelerometer_data_interval(cleaned_data, time_seconds, start_time=None, end_time=None, title_suffix="Full Duration", plot_each_axis=False):
    """
    Plot accelerometer data over a specified time interval.

    Can either plot individual X, Y, and Z axes or plot the normalized data. The displayed
    time interval is controlled by start_time and end_time.

    Parameters:
        cleaned_data (DataFrame): Accelerometer data.
        time_seconds (array-like): Corresponding time values.
        start_time (float, optional): Start time for plotting.
        end_time (float, optional): End time for plotting.
        title_suffix (str): Text to append to the plot title.
        plot_each_axis (bool): If True, plot each axis separately.
    """
    if start_time is not None or end_time is not None:
        mask = (time_seconds >= (start_time or time_seconds.min())) & (time_seconds <= (end_time or time_seconds.max()))
        filtered_data = cleaned_data[mask]
        filtered_time = time_seconds[mask]
    else:
        filtered_data = cleaned_data
        filtered_time = time_seconds

    if plot_each_axis:
        plt.figure(figsize=(15, 10))
        for i, (axis, color) in enumerate(zip(["X", "Y", "Z"], ["blue", "green", "red"]), 1):
            plt.subplot(3, 1, i)
            plt.plot(filtered_time, filtered_data[axis], label=f"{axis}-axis", color=color)
            plt.title(f"{axis}-Axis Acceleration ({title_suffix})")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Acceleration")
            plt.legend()

        plt.tight_layout()
        plt.show()

    normalized_data = normalize_data(filtered_data)
    plt.figure(figsize=(15, 5))
    plt.plot(filtered_time, normalized_data, label="Norm (X, Y, Z)", color="purple")
    plt.title(f"Normalized Accelerometer Data ({title_suffix})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration Norm")
    plt.legend()
    plt.show()


def find_peaks_in_interval(normalized_data, time_seconds, start_time=None, end_time=None):
    """
    Find peaks in normalized accelerometer data within a specified interval.

    The function splits the data into two halves and identifies the peak (maximum value) in each half.

    Parameters:
        normalized_data (Series or ndarray): Normalized accelerometer values.
        time_seconds (array-like): Corresponding time values.
        start_time (float, optional): Start of the interval.
        end_time (float, optional): End of the interval.

    Returns:
        tuple: (First half peak time, second half peak time, first half index, second half index)
    """
    if start_time is not None or end_time is not None:
        mask = (time_seconds >= (start_time or time_seconds.min())) & (time_seconds <= (end_time or time_seconds.max()))
        filtered_data = normalized_data[mask]
        filtered_time = time_seconds[mask]
    else:
        filtered_data = normalized_data
        filtered_time = time_seconds

    filtered_time = pd.Series(filtered_time.values, index=filtered_data.index)
    mid_point = len(filtered_data) // 2
    first_half_data = filtered_data.iloc[:mid_point]
    second_half_data = filtered_data.iloc[mid_point:]
    first_half_time = filtered_time.iloc[:mid_point]
    second_half_time = filtered_time.iloc[mid_point:]

    index_first_clap = first_half_data.idxmax()
    index_last_clap = second_half_data.idxmax()
    first_half_max_time = first_half_time[index_first_clap]
    second_half_max_time = second_half_time[index_last_clap]
    return first_half_max_time, second_half_max_time, index_first_clap, index_last_clap


def slice_accelerometer_data(metadata_csv, raw_accel_data_dir, output_root):
    """
    Slice accelerometer data based on metadata and save the slices to CSV files.

    For each entry in the metadata CSV, loads the corresponding accelerometer data,
    slices it between the provided start and end indices, and writes the sliced data
    to an output directory.

    Parameters:
        metadata_csv (str): Path to the metadata CSV.
        raw_accel_data_dir (str): Directory with raw accelerometer CSV files.
        output_root (str): Root directory where sliced data will be saved.

    Returns:
        list: A list of dictionaries with slice information per video.
    """
    df = pd.read_csv(metadata_csv)
    for _, row in df.iterrows():
        video_id = row["video_id"]
        watch_filename = row["watch"]
        start_idx = int(row["index_start_clap"])
        end_idx = int(row["index_end_clap"])
        acc_data_path = os.path.join(raw_accel_data_dir, f"{watch_filename}.csv")
        if not os.path.exists(acc_data_path):
            print(f"File not found: {acc_data_path}")
            continue
        cleaned_data, _ = load_accelerometer_data(acc_data_path)
        sliced_data = cleaned_data.loc[start_idx:end_idx]
        video_folder = os.path.join(output_root, video_id)
        os.makedirs(video_folder, exist_ok=True)
        output_filename = f"{video_id}_acceleration_data.csv"
        output_path = os.path.join(video_folder, output_filename)
        sliced_data.to_csv(output_path, index=False)
        print(f"Sliced data saved to: {output_path}")
