import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy import VideoFileClip





# Method to plot the audio waveform of a video file

def plot_audio_waveform_from_array(audio_array, fps):
    # If stereo, convert to mono by averaging channels
    if audio_array.ndim > 1:
        audio_data = np.mean(audio_array, axis=1)
    else:
        audio_data = audio_array

    audio_data = np.abs(audio_data)

    duration = len(audio_data) / fps
    time_axis = np.linspace(0, duration, num=len(audio_data))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio_data, color='steelblue')
    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Method to detect Claps using segmentation (first and last segments only)

def detect_claps_first_last_segments(audio_array, fps, num_segments):

    # convert stereo to mono
    audio_data = np.mean(audio_array, axis=1)

    abs_audio = np.abs(audio_data)
    segment_length = len(abs_audio) // num_segments
    claps = []

    # Process only the first and last segments
    segments_to_process = [0, num_segments - 1]
    for segment_index in segments_to_process:
        start_idx = segment_index * segment_length
        end_idx = (segment_index + 1) * segment_length if segment_index < num_segments - 1 else len(abs_audio)
        segment = abs_audio[start_idx:end_idx]
        
        if len(segment) > 0:
            # Find the maximum value in the segment
            max_idx = segment.argmax()

            clap_time = (start_idx + max_idx) / float(fps)
            claps.append((clap_time, start_idx + max_idx))

    return claps

# Method to process videos in a directory

def process_videos_in_directory(directory_path, audio_output_dir, num_segments):
    video_extension = ".mp4"
    clap_results = []

    # Create the audio output directory if it doesn't exist
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename.lower())[1] == video_extension:

            print(f"\nProcessing video: {filename}")
            try:
                clip = VideoFileClip(file_path)
                audio = clip.audio
                if audio is None:
                    print("  No audio track found.")
                    continue

                fps = audio.fps
                audio_array = audio.to_soundarray()

                # Save the audio to a file
                audio_output_path = os.path.join(audio_output_dir, f"{os.path.splitext(filename)[0]}.wav")
                audio.write_audiofile(audio_output_path)
                print(f"  Audio saved to {audio_output_path}")


                plot_audio_waveform_from_array(audio_array, fps)

                # Now detect claps
                claps = detect_claps_first_last_segments(audio_array, fps=fps, num_segments=num_segments)

                if claps:
                    for c_time, frame in claps:
                        print(f"  Clap detected at {c_time:.2f} seconds (frame {frame})")
                
                duration_between_claps = claps[1][0] - claps[0][0]

                clap_results.append({"Filename": filename,
                                     "Start Clap Seconds": claps[0][0],
                                     "End Clap Seconds": claps[1][0],
                                     "Start Clap Frame": claps[0][1],
                                     "End Clap Frame": claps[1][1],
                                     "Audio Path": audio_output_path,
                                     "Duration Between Claps": duration_between_claps})


            except Exception as e:
                print(f"  Error processing {filename}: {e}")

    return clap_results




# Function to load accelerometer data
def load_accelerometer_data(acc_data_path, sampling_frequency=256):

    raw_data = pd.read_csv(
        acc_data_path,
        skiprows=10,
        names=["X", "Y", "Z"],
        delimiter=',',
        decimal=","
    )
    raw_data = raw_data.iloc[1:].reset_index(drop=True)
    cleaned_data = raw_data.apply(pd.to_numeric, errors='coerce').dropna()
    time_seconds = (cleaned_data.index - cleaned_data.index[0]) / sampling_frequency
    return cleaned_data, time_seconds

# Function to normalize data
def normalize_data(data):
    return np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)

# Function to plot data for a specific time interval
def plot_accelerometer_data_interval(cleaned_data, time_seconds, start_time=None, end_time=None, title_suffix="Full Duration"):

    if start_time is not None or end_time is not None:
        mask = (time_seconds >= (start_time or time_seconds.min())) & (time_seconds <= (end_time or time_seconds.max()))
        filtered_data = cleaned_data[mask]
        filtered_time = time_seconds[mask]
    else:
        filtered_data = cleaned_data
        filtered_time = time_seconds

    # Plot each axis
    plt.figure(figsize=(15, 10))
    for i, (axis, color) in enumerate(zip(['X', 'Y', 'Z'], ['blue', 'green', 'red']), 1):
        plt.subplot(3, 1, i)
        plt.plot(filtered_time, filtered_data[axis], label=f'{axis}-axis', color=color)
        plt.title(f'{axis}-Axis Acceleration ({title_suffix})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Acceleration')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot normalized data
    normalized_data = normalize_data(filtered_data)
    plt.figure(figsize=(15, 5))
    plt.plot(filtered_time, normalized_data, label='Norm (X, Y, Z)', color='purple')
    plt.title(f'Normalized Accelerometer Data ({title_suffix})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration Norm')
    plt.legend()
    plt.show()