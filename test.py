import cv2
import mediapipe as mp
import numpy as np
from norfair import Detection, Tracker, draw_tracked_objects
import torch

# Load YOLO model for person detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]  # Filter for person class only (class 0 in COCO)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Norfair tracker
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=30,
    initialization_delay=2
)

# Open webcam feed
cap = cv2.VideoCapture(0)

def yolo_detections_to_norfair_detections(yolo_results, frame_width, frame_height):
    """Convert YOLO detections to Norfair detections format."""
    norfair_detections = []
    
    # Get the pandas dataframe from YOLO results
    yolo_detections = yolo_results.pandas().xyxy[0]
    
    if len(yolo_detections) == 0:
        return []
    
    # Process each YOLO detection
    for _, detection in yolo_detections.iterrows():
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
        
        # Convert to Norfair format (normalized coordinates)
        bbox_points = np.array([
            [x1/frame_width, y1/frame_height],
            [x2/frame_width, y1/frame_height],
            [x2/frame_width, y2/frame_height],
            [x1/frame_width, y2/frame_height]
        ])
        
        # Create Norfair detection with confidence score
        confidence = detection['confidence']
        norfair_detections.append(Detection(
            points=bbox_points,
            scores=np.array([float(confidence)] * 4)
        ))
    
    return norfair_detections

def crop_person(frame, bbox):
    """Crop person from frame using bounding box."""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure coordinates are within frame boundaries
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Crop person
    return frame[y1:y2, x1:x2]

def process_pose(person_crop, full_frame, bbox):
    """Process pose for a single person crop."""
    if person_crop.size == 0:
        return None
    
    # Convert to RGB for MediaPipe
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(rgb_crop)
    
    if not results.pose_landmarks:
        return None
    
    # Convert landmarks to original frame coordinates
    h, w = person_crop.shape[:2]
    x1, y1 = bbox[0], bbox[1]
    
    # Draw landmarks on the full frame
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # Convert normalized coordinates to pixel coordinates within the crop
        px = int(landmark.x * w)
        py = int(landmark.y * h)
        
        # Map back to the original frame
        frame_px = px + int(x1)
        frame_py = py + int(y1)
        
        # Draw keypoint on full frame
        if landmark.visibility > 0.5:
            cv2.circle(full_frame, (frame_px, frame_py), 3, (0, 255, 0), -1)
    
    return results

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Convert YOLO detections to Norfair format
    detections = yolo_detections_to_norfair_detections(results, width, height)
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    
    # Process each tracked person
    for obj in tracked_objects:
        # Get bounding box coordinates
        bbox_points = obj.estimate
        bbox_points = bbox_points * np.array([width, height])  # Denormalize
        
        # Extract bounding box coordinates
        x1, y1 = bbox_points[0]
        x2, y2 = bbox_points[2]
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Crop person from frame
        person_crop = crop_person(frame, [x1, y1, x2, y2])
        
        # Process pose for this person
        process_pose(person_crop, frame, [x1, y1, x2, y2])
        
        # Draw person ID
        cv2.putText(
            frame, 
            f"ID: {obj.id}", 
            (int(x1), int(y1) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
    
    # Display number of people tracked
    cv2.putText(
        frame,
        f"People tracked: {len(tracked_objects)}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Display the resulting frame
    cv2.imshow("Multi-Person Pose Tracking", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()