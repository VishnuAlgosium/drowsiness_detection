#!/usr/bin/env python
"""
Simple standalone test: plays the drowsiness alert beep every 2 seconds.
Uses pygame + numpy only (same approach as audio.py), so it works
identically on a laptop and on a Raspberry Pi (pygame's SDL audio
backend runs fine on Pi's default audio output / 3.5mm jack / HDMI audio).

Run:  python test_alert_sound.py
Stop: Ctrl+C
"""

import time
import numpy as np

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"[ERROR] pygame audio init failed: {e}")
    print("Install with: pip install pygame")
    print("On Raspberry Pi, also make sure ALSA is set up: sudo apt-get install -y alsa-utils")


def _generate_beep(frequency: float = 880.0, duration: float = 0.28,
                    volume: float = 0.6, sample_rate: int = 44100) -> np.ndarray:
    """Synthesize a short beep as a numpy array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)
    envelope = np.exp(-t * 8)
    wave = (wave * envelope * volume * 32767).astype(np.int16)
    return wave


def play_alert_sound():
    """Play a triple-beep alert."""
    if not AUDIO_AVAILABLE:
        return
    try:
        sample_rate = 44100
        silence = np.zeros(int(sample_rate * 0.1), dtype=np.int16)
        beep1 = _generate_beep(880)
        beep2 = _generate_beep(660)
        sequence = np.concatenate([beep1, silence, beep1, silence, beep2])
        stereo = np.column_stack([sequence, sequence])
        sound = pygame.sndarray.make_sound(sequence)
        sound.play()
    except Exception as e:
        print(f"[WARN] Audio error: {e}")


if __name__ == "__main__":
    if not AUDIO_AVAILABLE:
        raise SystemExit(1)

    print("[INFO] Playing alert sound every 2 seconds. Press Ctrl+C to stop.\n")
    count = 0
    try:
        while True:
            count += 1
            print(f"[BEEP #{count}] {time.strftime('%H:%M:%S')}")
            play_alert_sound()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        pygame.mixer.quit()