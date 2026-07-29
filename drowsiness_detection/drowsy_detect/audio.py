"""
audio.py
--------
Pygame-based beep alert used when drowsiness is detected.
"""

import numpy as np

try:
    import pygame

    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    AUDIO_AVAILABLE = True

except Exception as e:  # pragma: no cover - environment dependent
    AUDIO_AVAILABLE = False
    print(f"[WARN] pygame audio unavailable: {e}")


def _beep(freq: float = 880, dur: float = 0.28, vol: float = 0.6, sr: int = 44100) -> np.ndarray:
    """Generate a single decaying sine-wave beep as int16 PCM samples."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    env = np.exp(-t * 8)
    return (wave * env * vol * 32767).astype(np.int16)


# Keep a module-level reference so the Sound object isn't garbage-collected
# mid-playback (SDL mixer holds the buffer, but there's no reason to risk it).
_last_sound = None


def play_alert() -> None:
    """
    Fire-and-forget: starts the two-tone alert sound and returns immediately.
    Playback happens on pygame/SDL's own audio thread, so this does NOT block
    the caller's main loop (camera capture / display / keyboard polling keep
    running while the beep plays).
    """
    global _last_sound

    if not AUDIO_AVAILABLE:
        print("[WARN] Audio not available")
        return

    try:
        silence = np.zeros(int(44100 * 0.1), dtype=np.int16)

        b1 = _beep(880, 0.3)
        b2 = _beep(660, 0.3)

        seq = np.concatenate([b1, silence, b1, silence, b2])

        _last_sound = pygame.sndarray.make_sound(seq)
        _last_sound.play()
        # No busy-wait here — that was what froze the video feed during alerts.

    except Exception as e:
        print(f"[WARN] Audio error: {e}")


def shutdown_audio() -> None:
    """Cleanly shut down the pygame mixer, if it was initialized."""
    if AUDIO_AVAILABLE:
        pygame.mixer.quit()
