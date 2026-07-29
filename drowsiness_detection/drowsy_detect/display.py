"""
display.py
----------
Pipes raw BGR frames to an `ffplay` subprocess for low-latency display,
since cv2.imshow is unreliable/unavailable in some headless-ish setups.
"""

import shutil
import subprocess

import numpy as np


class LazyDisplay:
    """Lazily starts an ffplay process and streams raw frames to it."""

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None

    def start(self) -> None:
        if self.proc:
            return

        if shutil.which("ffplay") is None:
            print("[WARN] ffplay missing. Install: sudo apt install ffmpeg")
            return

        cmd = [
            "ffplay",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-framedrop",
            "-an",
            "-i", "-",
        ]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def show(self, frame: np.ndarray) -> None:
        if self.proc is None:
            return

        try:
            self.proc.stdin.write(frame.tobytes())
        except Exception:
            self.proc = None

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
