import cv2
import mediapipe as mp

# Initialisiere MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Video einlesen
video_path = 'Videos/Laufen 003.mp4'  # Pfad zum Video
cap = cv2.VideoCapture(video_path)

# Optional: Video speichern
output_path = 'output_with_skeleton.avi'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Skalierungsfaktor für das angezeigte Fenster
scale_factor = 0.5  # 50% der Originalgröße

# Schrittzähler-Variablen
prev_left_ankle = None
prev_right_ankle = None
step_count = 0
threshold = 22.4  # Pixelwert, der Bewegung definiert

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertiere Frame zu RGB (für MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Überprüfen, ob Landmarks gefunden wurden
    if results.pose_landmarks:
        # Zeichne Landmarks und Verbindungen (Skelett)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extrahiere relevante Landmarks
        landmarks = results.pose_landmarks.landmark
        # Schrittzählung basierend auf horizontaler (x) und vertikaler (y) Bewegung
        left_ankle = (
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]
        )
        right_ankle = (
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]
        )

        if prev_left_ankle is not None and prev_right_ankle is not None:
            # Berechne die Bewegung (Differenz in x und y)
            left_movement = abs(left_ankle[0] - prev_left_ankle[0]) + abs(left_ankle[1] - prev_left_ankle[1])
            right_movement = abs(right_ankle[0] - prev_right_ankle[0]) + abs(right_ankle[1] - prev_right_ankle[1])

            # Schrittzählung, wenn Bewegung eines Knöchels den Schwellwert überschreitet
            if left_movement > threshold or right_movement > threshold:
                step_count += 1

        prev_left_ankle = left_ankle
        prev_right_ankle = right_ankle

    # Verkleinere das Frame für die Anzeige
    display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Zeige das verkleinerte Video mit Skelett und Schrittzählung
    cv2.putText(display_frame, f"Steps: {step_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pose Tracking", display_frame)

    # Optional: Schreibe das Frame ins Output-Video (unverändert)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Gesamtanzahl der Schritte: {step_count}")
