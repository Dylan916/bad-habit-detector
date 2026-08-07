"""
CSV Habit Logging Module.
Appends detection events with timestamps to habit_log.csv.
"""

import csv
import os
import threading
from datetime import datetime
import config

class HabitLogger:
    def __init__(self, log_file: str = config.LOG_FILE_PATH):
        self.log_file = log_file
        self.lock = threading.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        with self.lock:
            if not os.path.exists(self.log_file):
                with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Habit", "Details"])

    def log_habit(self, habit_name: str, details: str = ""):
        """
        Appends a habit detection log entry.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, habit_name, details])
