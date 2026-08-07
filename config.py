"""
Configuration settings and threshold constants for Bad Habit Detector.
Tune these constants to adjust sensitivity for your webcam setup.
"""

# ==========================================
# 1. Hair Scratching Detection Settings
# ==========================================
# --- Head Proximity Zone ---
# Instead of a simple y-line, we define a zone around the head using face mesh landmarks.
# The hand must be within this expanded bounding box to be considered "near the head".
# Normalized margin added around the face bounding box (derived from forehead, chin, temples).
SCRATCH_HEAD_ZONE_MARGIN = 0.08  # Extra margin around face bbox (normalized coords)

# --- Motion / Jitter Detection ---
# Scratching involves rapid small hand movements. We track hand positions over recent
# frames and require a minimum "jitter" (average frame-to-frame displacement) to
# distinguish scratching from a static raised hand.
SCRATCH_JITTER_THRESH = 0.008       # Min avg frame-to-frame displacement (normalized) to count as motion
SCRATCH_JITTER_WINDOW = 8          # Number of recent frames to track for jitter calculation

# --- Alert Trigger ---
# Number of consecutive frames with BOTH conditions (near head + jittering) to trigger alert
SCRATCH_CONSECUTIVE_FRAMES = 12


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
# COCO class IDs: 67 = "cell phone", 65 = "remote" (tilted/sideways phones frequently trigger class 65)
PHONE_COCO_CLASS_IDS = [67]

# Minimum confidence threshold for cell phone detection
PHONE_CONF_THRESH = 0.30

# Require a hand to be touching/holding the phone bounding box to trigger alert
# NOTE: Set to False because holding a phone occludes the hand from MediaPipe
PHONE_REQUIRE_HAND_HOLDING = False

# Perform YOLO inference every N frames to save CPU/GPU resources
PHONE_CHECK_INTERVAL = 8

# Number of consecutive positive YOLO check intervals required to trigger alert
PHONE_CONSECUTIVE_CHECKS = 2

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
