import os
import numpy as np
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