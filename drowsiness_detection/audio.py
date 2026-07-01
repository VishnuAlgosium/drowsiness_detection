"""Synthesized beep audio alerts via pygame (optional dependency)."""

import numpy as np

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False
    print("[WARN] pygame not found — audio alerts disabled. Install with: pip install pygame")


def _generate_beep(frequency: float = 880.0, duration: float = 0.28,
                    volume: float = 0.6, sample_rate: int = 44100) -> np.ndarray:
    """Synthesize a short beep as a numpy array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)
    envelope = np.exp(-t * 8)          # quick decay
    wave = (wave * envelope * volume * 32767).astype(np.int16)
    return wave


def play_alert_sound():
    """Play a triple-beep alert if pygame is available."""
    if not AUDIO_AVAILABLE:
        return
    try:
        sample_rate = 44100
        silence = np.zeros(int(sample_rate * 0.1), dtype=np.int16)
        beep1 = _generate_beep(880)
        beep2 = _generate_beep(660)
        sequence = np.concatenate([beep1, silence, beep1, silence, beep2])
        stereo = np.column_stack([sequence, sequence])
        sound = pygame.sndarray.make_sound(stereo)
        sound.play()
    except Exception as e:
        print(f"[WARN] Audio error: {e}")


def shutdown_audio():
    if AUDIO_AVAILABLE:
        pygame.mixer.quit()
