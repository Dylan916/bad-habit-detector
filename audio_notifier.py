"""
Audio Notification Module.
Provides non-blocking synthesized beep alerts using sounddevice and numpy.
"""

import threading
import numpy as np
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

import config

class AudioNotifier:
    def __init__(self, sample_rate: int = config.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.enabled = HAS_SOUNDDEVICE

    def _generate_sine_wave(self, frequency: float, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # 0.5 amplitude to avoid clipping
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Apply a smooth fade in / fade out to avoid clicks
        fade_samples = int(self.sample_rate * 0.01)
        if len(tone) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        return tone.astype(np.float32)

    def play_beep(self, frequency: float, duration: float = config.BEEP_DURATION):
        """
        Play a synthesized sine wave beep at frequency for duration in a background thread.
        """
        if not self.enabled:
            return

        def _play():
            try:
                wave = self._generate_sine_wave(frequency, duration)
                sd.play(wave, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                # Silently catch any audio device issues
                pass

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
