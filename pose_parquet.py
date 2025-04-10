import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Store all frames' data in a list
landmark_data = []
frame_count = 0

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            landmark_data.append({
                "frame": frame_count,
                "landmark_id": id,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

    cv2.imshow("Pose Logger - Press Q to Quit", frame)
    frame_count += 1

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert to DataFrame and save as Parquet
df = pd.DataFrame(landmark_data)
df.to_parquet("pose_landmarks.parquet", compression="gzip")
print("Saved to pose_landmarks.parquet (GZIP compressed)")