"""
Configuration settings and threshold constants for Bad Habit Detector.
Tune these constants to adjust sensitivity for your webcam setup.
"""

# ==========================================
# 1. Hair Scratching Detection Settings
# ==========================================
# Vertical normalized offset above top-of-head (Face Mesh landmark 10)
# Lower y in normalized space means higher on screen.
# Hand above (head_y - SCRATCH_HEAD_MARGIN) triggers potential scratching.
SCRATCH_HEAD_MARGIN = 0.05

# Number of consecutive frames the condition must hold to trigger alert
SCRATCH_CONSECUTIVE_FRAMES = 8

# ==========================================
# 2. Finger / Nail Biting Detection Settings
# ==========================================
# Maximum normalized Euclidean distance between hand landmarks and mouth landmark (Face Mesh landmark 13)
BITING_MOUTH_THRESH = 0.08

# Require mouth to be open to confirm nail biting (reduces false positives when hand is resting near chin)
REQUIRE_MOUTH_OPEN_FOR_BITING = True

# Normalized distance between upper lip (landmark 13) and lower lip (landmark 14) to consider mouth open
MOUTH_OPEN_THRESH = 0.025

# Number of consecutive frames hand must remain close to mouth to trigger alert
BITING_CONSECUTIVE_FRAMES = 8


# ==========================================
# 3. Phone Use Detection Settings (YOLOv8)
# ==========================================
# COCO class ID for "cell phone" is 67
PHONE_COCO_CLASS_ID = 67

# Minimum confidence threshold for cell phone detection (increase to 0.65+ to reduce false positives)
PHONE_CONF_THRESH = 0.65

# Require a hand to be touching/holding the phone bounding box to trigger alert
PHONE_REQUIRE_HAND_HOLDING = True

# Perform YOLO inference every N frames to save CPU/GPU resources
PHONE_CHECK_INTERVAL = 5

# Number of consecutive positive YOLO check intervals required to trigger alert
PHONE_CONSECUTIVE_CHECKS = 3

# ==========================================
# 4. Alert & Audio Settings
# ==========================================
# Cooldown period (in seconds) between repeated alerts for the same habit
ALERT_COOLDOWN_SEC = 3.0

# Audio sample rate for sounddevice
AUDIO_SAMPLE_RATE = 44100

# Beep frequencies (Hz) for distinct habit audio cues
SCRATCH_BEEP_FREQ = 600   # Medium pitch
BITING_BEEP_FREQ = 880    # High pitch
PHONE_BEEP_FREQ = 440     # Low pitch

# Duration of beep sound (in seconds)
BEEP_DURATION = 0.25

# ==========================================
# 5. Logging & UI Settings
# ==========================================
LOG_FILE_PATH = "habit_log.csv"

# Initial state for debug overlay visualization (True = show landmarks & bounding boxes)
DEFAULT_DEBUG_MODE = True
