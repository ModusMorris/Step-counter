import cv2
import numpy as np

def clap_detection_frames(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    
    # Video-Parameter
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    frame_index = 0
    last_clap_frame = -1
    min_frame_gap = int(frame_rate * 0.5)  # 0.5 seconds frame gap
    motion_history = []
    claps_detected = []
    
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # grayscaling and smoothing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # difference calculation
        diff = cv2.absdiff(prev_gray, gray)
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff_sum = np.sum(diff_thresh)
        motion_history.append(diff_sum)
        
        # Adaptive threshold
        if len(motion_history) > 30:
            motion_history.pop(0)
        adaptive_threshold = np.mean(motion_history) * 2.2  # 2.2 works best (after parameter analysis. still not perfect)
        
        # Clap detection
        if diff_sum > adaptive_threshold:
            if frame_index - last_clap_frame > min_frame_gap:
                claps_detected.append((frame_index, frame_index / frame_rate))
                last_clap_frame = frame_index
        
        prev_gray = gray
        frame_index += 1
    
    cap.release()
    return claps_detected

# start clap detection
clap_frames_optimized = clap_detection_frames("../videos/006.mp4", frame_skip=1)
clap_frames_optimized

# ouput detected claps in terminal
if clap_frames_optimized:
    for clap in clap_frames_optimized:
        print(f"Clap detected at Frame {clap[0]}, Time {clap[1]:.2f} seconds")
else:
    print("Keine Klatscher erkannt!")