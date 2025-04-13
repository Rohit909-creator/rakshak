# 🔍 Rakshak - Pose Tracking Surveillance System (Work in Progress)

> ⚠️ **Note**: This project is currently under active development. Features, functionality, and performance may change as development progresses.

## 📝 Overview

Rakshak is a **Real-Time Pose Tracking Surveillance System** that detects and tracks multiple people in a video feed. The system uses a combination of **YOLOv5** for person detection, **MediaPipe Pose** for pose estimation, and **Norfair** for multi-object tracking. It assigns unique IDs to each person and performs pose tracking, making it suitable for surveillance, activity monitoring, or other applications requiring real-time human tracking and analysis.

## ✨ Features

- 👥 **Multi-person tracking** using **Norfair** with unique ID assignment
- 🧍 **Pose estimation** for each person detected in the video
- 🔎 **Person detection** using **YOLOv5** (focused only on people)
- 📹 **Real-time video feed processing** with webcam support
- 🏷️ Display of tracked person IDs along with pose landmarks

## 🛠️ Tech Stack

- **Programming Language**: Python 3.7+
- **Object Detection**: YOLOv5
- **Pose Estimation**: MediaPipe Pose
- **Multi-Object Tracking**: Norfair
- **Video Processing**: OpenCV
- **Numerical Operations**: NumPy

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam or video source

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rakshak.git
   cd rakshak
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the script to start the surveillance system:
```bash
python main.py
```

The system will use your webcam for real-time person detection and pose tracking, assigning a unique ID to each tracked person and displaying pose landmarks on the video feed.

Press `q` to exit the application.

## 🔍 How It Works

1. **Person Detection (YOLOv5)**:
   - The system uses YOLOv5 to detect people in each frame of the video feed
   - Only person detections (class 0 in COCO dataset) are considered
   - This focused approach reduces processing requirements and improves performance

2. **Pose Estimation (MediaPipe)**:
   - Once people are detected, MediaPipe Pose estimates landmarks for each person
   - These landmarks include key body points like shoulders, elbows, and knees
   - The landmarks provide detailed information about body positioning and movement

3. **Tracking (Norfair)**:
   - Norfair tracks multiple people across frames
   - Each person is assigned a unique ID that persists even if they move in and out of frame
   - Euclidean distance is used to associate current detections with previous ones

4. **Visualization**:
   - Tracked people are shown with bounding boxes and unique IDs
   - Pose landmarks (joints and body parts) are visualized on each person

## ⚙️ Customization

- **Distance Threshold**: Adjust the `distance_threshold` parameter in Norfair's Tracker to fine-tune tracking behavior
- **Pose Visibility**: Modify the visibility threshold for pose landmarks to control when they are displayed

## 🚧 Current Limitations

- The system currently tracks only persons and may not perform optimally in crowded scenes
- Initialization delay can affect real-time performance if set too high
- Performance may vary based on hardware capabilities

## 🔮 Planned Improvements

- Improve multi-person detection in crowded scenes
- Integrate with a database to store tracked data
- Add anomaly detection based on unusual movements or poses
- Optimize performance for resource-constrained environments
- Add alerting mechanisms for security applications

## 🙏 Acknowledgements

- YOLOv5 for object detection
- MediaPipe for pose estimation
- Norfair for multi-object tracking
- OpenCV for video processing
