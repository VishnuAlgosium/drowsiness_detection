"""
audio.py
--------
Pygame-based beep alerts. Each detector gets its own distinct tone so an
alert can be identified by ear alone:
    drowsiness  -> two-tone, low-pitched, descending
    yawn        -> single, low-pitched
    phone       -> triple, high-pitched
    distraction -> double, mid-pitched
    head drop   -> fast three-tone descending sweep
    occlusion   -> two-tone, mid-pitched, ascending (eyes hidden from camera)
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


# Module-level reference so Sound objects aren't garbage-collected mid-playback.
_last_sound = None


def _play_sequence(seq: np.ndarray) -> None:
    """Fire-and-forget playback on pygame/SDL's own audio thread; doesn't block the main loop."""
    global _last_sound

    if not AUDIO_AVAILABLE:
        print("[WARN] Audio not available")
        return

    try:
        _last_sound = pygame.sndarray.make_sound(seq)
        _last_sound.play()
    except Exception as e:
        print(f"[WARN] Audio error: {e}")


def play_alert() -> None:
    """Two-tone alert used for drowsiness (sustained eye closure)."""
    silence = np.zeros(int(44100 * 0.1), dtype=np.int16)

    b1 = _beep(880, 0.3)
    b2 = _beep(660, 0.3)

    seq = np.concatenate([b1, silence, b1, silence, b2])
    _play_sequence(seq)


def play_yawn_alert() -> None:
    """Single, lower-pitched tone used for yawn detection."""
    seq = _beep(440, 0.35, vol=0.5)
    _play_sequence(seq)


def play_phone_alert() -> None:
    """Triple, higher-pitched tone used for phone-use detection."""
    silence = np.zeros(int(44100 * 0.08), dtype=np.int16)
    beep = _beep(1200, 0.15, vol=0.5)
    seq = np.concatenate([beep, silence, beep, silence, beep])
    _play_sequence(seq)


def play_distraction_alert() -> None:
    """Double, mid-pitched tone used for distraction (gaze away / low-confidence phone)."""
    silence = np.zeros(int(44100 * 0.12), dtype=np.int16)
    beep = _beep(750, 0.25, vol=0.55)
    seq = np.concatenate([beep, silence, beep])
    _play_sequence(seq)


def play_head_drop_alert() -> None:
    """Fast, descending three-tone alert used for a sudden head drop (nodding off)."""
    b1 = _beep(1000, 0.12, vol=0.6)
    b2 = _beep(750, 0.12, vol=0.6)
    b3 = _beep(500, 0.18, vol=0.65)
    seq = np.concatenate([b1, b2, b3])
    _play_sequence(seq)


def play_occlusion_alert() -> None:
    """Rising two-tone reminder used when the eyes are hidden from the camera
    (e.g. sunglasses), so EAR-based drowsiness detection can't be trusted."""
    silence = np.zeros(int(44100 * 0.1), dtype=np.int16)
    b1 = _beep(500, 0.25, vol=0.5)
    b2 = _beep(700, 0.25, vol=0.5)
    seq = np.concatenate([b1, silence, b2])
    _play_sequence(seq)


def shutdown_audio() -> None:
    """Cleanly shut down the pygame mixer, if it was initialized."""
    if AUDIO_AVAILABLE:
        pygame.mixer.quit()