# 🛡️ PoseTrack AI Surveillance – Project Plan

An intelligent surveillance system leveraging human **pose estimation**, **tracking**, and **action recognition**, hosted on a server for real-time or near-real-time behavior monitoring and alerting.

---

## 📌 Project Objective

Build a modular, scalable AI surveillance pipeline that:

- Detects and tracks human body poses from video
- Recognizes human activities (e.g., walking, falling, fighting)
- Logs and alerts on suspicious or dangerous behavior
- Serves pose/action data via backend APIs
- Visualizes activity via a live web dashboard

---

## 🗺️ Project Phases & Timeline

### ✅ Phase 1: Requirements & Environment Setup (Week 1)
- [ ] Define surveillance goals (e.g., fall detection, violence)
- [ ] Choose camera/edge input (IP cam, webcam, Jetson, etc.)
- [ ] Select stack: Python 3.9+, OpenCV, FastAPI, MediaPipe
- [ ] Set up local development & version control (Git)

### 👤 Phase 2: Pose Estimation (Week 2)
- [ ] Integrate **MediaPipe Pose** for keypoint extraction
- [ ] Test pose inference on sample video/webcam
- [ ] Visualize skeleton overlay using OpenCV

### 🧭 Phase 3: Pose Tracking (Week 3)
- [ ] Add **DeepSORT** or **Norfair** for ID-based tracking
- [ ] Track individual people across frames
- [ ] Output keypoints with person ID over time

### 🧠 Phase 4: Action Recognition (Week 4)
- [ ] Collect or simulate action-labeled pose sequences
- [ ] Train a lightweight **LSTM** or **TCN** model
- [ ] Integrate model for real-time action inference
- [ ] Validate with actions like: walk, sit, fall, fight

### ☁️ Phase 5: Backend Server & APIs (Week 5)
- [ ] Build **FastAPI** server with routes:
  - `/upload_frame` – image upload for processing
  - `/upload_keypoints` – direct keypoint input (edge-side)
  - `/get_alerts` – returns logs of detected actions
- [ ] Store alerts/events in SQLite or MongoDB
- [ ] (Optional) Export inference pipeline as Docker container

### 💻 Phase 6: Web Dashboard UI (Week 6)
- [ ] Build simple frontend (HTML/JS or React)
- [ ] Show:
  - Video feed with skeleton overlay
  - Action status per person
  - Live alert feed/logs
- [ ] Use WebSocket/MJPEG for live video display

### 🚨 Phase 7: Alerts & Optimization (Week 7)
- [ ] Add alerts for specific actions (e.g., fall, fight)
- [ ] Implement email or Telegram notifications
- [ ] Optimize models using **ONNX / TensorRT**
- [ ] Test performance (FPS, latency)

---

## 🛠️ Tools & Technologies

| Component | Tool/Library |
|----------|---------------|
| Pose Estimation | MediaPipe / MMPose |
| Tracking | DeepSORT / Norfair |
| Action Recognition | LSTM / TCN / ST-GCN |
| Backend | FastAPI / Flask |
| Frontend | React or HTML + JS |
| Deployment | Docker + NGINX |
| Visualization | OpenCV, WebSocket, MJPEG |
| Hardware | GPU Server / Jetson / Cloud VM |

---

## 🎯 Milestone Targets

| Week | Goal |
|------|------|
| 1 | Project setup + environment ready |
| 2 | Pose estimation module working |
| 3 | Tracking working with consistent person IDs |
| 4 | Basic action recognition running on pose data |
| 5 | Server API with processing pipeline |
| 6 | Web UI with live skeleton/action overlay |
| 7 | Alerts, optimizations, Docker deployment |

---

## 📦 Deliverables

- [ ] Full pose + action detection pipeline
- [ ] Dockerized backend API
- [ ] Web dashboard interface
- [ ] Real-time video with pose/action overlays
- [ ] Action detection logs & alerting system

---

## 🔒 Future Enhancements

- [ ] Add support for multi-camera views
- [ ] Integrate WebRTC for ultra-low-latency feeds
- [ ] Train on domain-specific actions (e.g., factory, hospital)
- [ ] Add role-based access & auth to the dashboard

---

## 📬 Contact

> Built by [Your Name / Team]  
> 📧 you@example.com  
> 🔗 [GitHub](https://github.com/yourname) | [LinkedIn](https://linkedin.com/in/yourname)

