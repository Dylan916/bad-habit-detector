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
   git clone https://github.com/Dylan916/bad-habit-detector.git
   cd bad-habit-detector
   ```

2. **Install dependencies & run application:**
   ```bash
   uv sync
   uv run main.py
   ```

3. **Controls:**
   - Press **`d`**: Toggle debug overlay (MediaPipe landmarks, scratch zone, mouth opening gap, and jitter score)
   - Press **`q`**: Quit application and display session summary report

---

## 🧠 How It Works

This project is **inference-only** (no custom model training required), combining pre-trained Computer Vision models with heuristic threshold rules:

1. **Hair Scratching Detection**:
   - Uses **MediaPipe Holistic** to construct a head bounding box from forehead (index 10), chin (index 152), and temple landmarks.
   - Evaluates hand landmark proximity within the upper third head zone and measures frame-to-frame displacement (**jitter motion**).
   - Triggers when both proximity and jitter motion (`SCRATCH_JITTER_THRESH`) hold for `SCRATCH_CONSECUTIVE_FRAMES` consecutive frames.

2. **Finger/Nail Biting Detection**:
   - Measures normalized Euclidean distance between hand landmarks and the inner mouth center (Face Mesh index 13).
   - Enforces **mouth opening verification**: calculates vertical lip gap between upper lip (index 13) and lower lip (index 14) to confirm mouth is open (`MOUTH_OPEN_THRESH`), preventing false positives when resting chin on hand.

3. **Cell Phone Detection**:
   - Uses pre-trained **YOLOv8s** (small model) filtering for COCO dataset class ID 67 (`cell phone`).
   - Runs inference every `PHONE_CHECK_INTERVAL` frames to optimize CPU utilization while maintaining responsive detection.

4. **Alerts & Logging**:
   - **Audio**: Plays non-blocking synthesized sine wave tones at distinct frequencies per habit (600 Hz for scratching, 880 Hz for biting, 440 Hz for phone use).
   - **Cooldown**: Prevents repeated spam by enforcing an `ALERT_COOLDOWN_SEC` cooldown between consecutive alerts.
   - **CSV Logging**: Appends timestamped records to `habit_log.csv`.

---

## ⚙️ Threshold Tuning Guide

All detection thresholds are exposed at the top of `config.py` for easy customization:

| Constant | Current Value | Description |
| :--- | :--- | :--- |
| `SCRATCH_HEAD_ZONE_MARGIN` | `0.08` | Extra normalized margin around face bounding box for scratch zone |
| `SCRATCH_JITTER_THRESH` | `0.008` | Min avg frame-to-frame hand displacement to count as scratching motion |
| `SCRATCH_CONSECUTIVE_FRAMES` | `12` | Consecutive frames required to trigger scratching alert |
| `BITING_MOUTH_THRESH` | `0.08` | Max normalized distance between hand joints and mouth center |
| `REQUIRE_MOUTH_OPEN_FOR_BITING` | `True` | Require mouth to be open to confirm nail biting |
| `MOUTH_OPEN_THRESH` | `0.025` | Min vertical lip gap distance between upper & lower lips |
| `BITING_CONSECUTIVE_FRAMES` | `8` | Consecutive frames required to trigger biting alert |
| `PHONE_COCO_CLASS_IDS` | `[67]` | COCO class ID list (`67` = cell phone) |
| `PHONE_CONF_THRESH` | `0.30` | YOLOv8 confidence score cutoff for cell phone detection |
| `PHONE_CHECK_INTERVAL` | `8` | Run YOLO object detection every $N$ frames |
| `PHONE_CONSECUTIVE_CHECKS` | `2` | Consecutive positive YOLO checks required for phone alert |
| `ALERT_COOLDOWN_SEC` | `3.0` | Cooldown period (in seconds) between repeated alerts |

---

## 📊 Sample Habit Log Format (`habit_log.csv`)

When an alert fires, a row is automatically saved to `habit_log.csv`:

```csv
Timestamp,Habit,Details
2026-08-07 00:55:02,Hair Scratching,Jitter: 0.0124
2026-08-07 00:55:18,Finger/Nail Biting,Counter: 8
2026-08-07 00:55:35,Phone Use,Confidence: 0.74
```

---

## ⚠️ Known Limitations

- **Lighting & Camera Angle**: Requires sufficient room lighting so MediaPipe can distinguish hand joints and facial landmarks.
- **Occlusion**: Extreme head tilts or holding objects that block the face/mouth may affect landmark detection accuracy.
- **Phone Visibility**: YOLO cell phone detection requires the phone to be visibly exposed within the camera frame.
- **Single User Assumption**: Designed for single-user desk webcam setups (tracks dominant face and hands in frame).
