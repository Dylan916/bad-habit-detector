# Bad Habit Detector 🎯

A real-time Python desktop application that monitors a webcam feed to detect three unwanted habits — **hair scratching**, **finger/nail biting**, and **phone use** — providing real-time HUD alerts, synthesized audio notifications, and CSV event logging.

![demo](demo.gif)

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)
- A working webcam

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bad_habit_tracker.git
   cd bad_habit_tracker
   ```

2. **Install dependencies & run application:**
   ```bash
   uv sync
   uv run main.py
   ```

3. **Controls:**
   - Press **`d`**: Toggle debug overlay (MediaPipe facial/hand wireframes & threshold guide lines)
   - Press **`q`**: Quit application and display session summary report

---

## 🧠 How It Works

This project is **inference-only** (no custom model training required), combining pre-trained Computer Vision models with heuristic threshold rules:

1. **Facial & Hand Landmark Tracking**:
   - Uses **MediaPipe Holistic** to extract normalized 3D coordinates for facial mesh landmarks and hand joints in real time.
   - **Hair Scratching**: Triggers when any hand landmark y-coordinate remains vertically above the head top landmark (Face Mesh index 10) for $N$ consecutive frames.
   - **Finger/Nail Biting**: Triggers when the normalized Euclidean distance between hand landmarks and the inner mouth center landmark (Face Mesh index 13) remains below `BITING_MOUTH_THRESH` for $N$ consecutive frames.

2. **Cell Phone Detection**:
   - Uses a pre-trained **YOLOv8n** model (COCO dataset class ID 67: `cell phone`).
   - Runs inference every `PHONE_CHECK_INTERVAL` frames to optimize CPU/GPU utilization while maintaining responsive detection.

3. **Alerts & Logging**:
   - **Audio**: Plays non-blocking synthesized sine wave tones at distinct frequencies per habit (e.g. 600 Hz for scratching, 880 Hz for biting, 440 Hz for phone use).
   - **Cooldown**: Prevents repeated spam by enforcing an `ALERT_COOLDOWN_SEC` cooldown between consecutive alerts.
   - **CSV Logging**: Appends timestamped records to `habit_log.csv`.

---

## ⚙️ Threshold Tuning Guide

All detection thresholds are cleanly exposed at the top of `config.py` for easy customization:

| Constant | Default | Description |
| :--- | :--- | :--- |
| `SCRATCH_HEAD_MARGIN` | `0.05` | Vertical normalized margin above forehead top (index 10) |
| `SCRATCH_CONSECUTIVE_FRAMES` | `8` | Consecutive frames required to trigger scratching alert |
| `BITING_MOUTH_THRESH` | `0.08` | Max normalized distance between hand joints and mouth center |
| `BITING_CONSECUTIVE_FRAMES` | `8` | Consecutive frames required to trigger biting alert |
| `PHONE_CONF_THRESH` | `0.50` | YOLOv8 confidence score cutoff for cell phone detection |
| `PHONE_CHECK_INTERVAL` | `5` | Run YOLO object detection every $N$ frames |
| `PHONE_CONSECUTIVE_CHECKS` | `2` | Consecutive positive YOLO checks required for phone alert |
| `ALERT_COOLDOWN_SEC` | `3.0` | Cooldown period (in seconds) between repeated alerts |

---

## 📊 Sample Habit Log Format (`habit_log.csv`)

When an alert fires, a row is automatically saved to `habit_log.csv`:

```csv
Timestamp,Habit,Details
2026-08-05 23:15:02,Hair Scratching,Counter: 8
2026-08-05 23:15:18,Finger/Nail Biting,Counter: 8
2026-08-05 23:15:35,Phone Use,Confidence: 0.84
```

---

## ⚠️ Known Limitations

- **Lighting & Camera Angle**: Requires sufficient room lighting so MediaPipe can distinguish hand joints and facial landmarks.
- **Occlusion**: Extreme head tilts or holding objects that block the face/mouth may affect landmark detection accuracy.
- **Phone Visibility**: YOLO cell phone detection requires the phone to be visibly exposed within the camera frame.
- **Single User Assumption**: Designed for single-user desk webcam setups (tracks dominant face and hands in frame).
