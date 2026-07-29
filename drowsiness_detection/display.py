"""GPU-accelerated frame display via an ffplay subprocess (replaces cv2.imshow).

ffplay renders through SDL2/OpenGL, so this avoids needing OpenCV built with
GTK/Qt support and works fine with opencv-python-headless.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import shutil
import subprocess

import cv2
import numpy as np


class FFplayDisplay:
    """Pipes raw BGR frames to an ffplay subprocess for display."""

    def __init__(self, width: int, height: int, title: str = "Display", fps: int = 30):
        if shutil.which("ffplay") is None:
            raise RuntimeError(
                "ffplay not found on PATH. Install ffmpeg: sudo apt-get install -y ffmpeg"
            )
        self.width = width
        self.height = height
        cmd = [
            "ffplay",
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-framedrop",
            "-an", "-sn",
            "-window_title", title,
            "-i", "-",
        ]
        env = os.environ.copy()
        env["SDL_RENDER_DRIVER"] = "software"  # Use OpenGL for GPU acceleration
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,env=env)

    def show(self, frame: np.ndarray) -> bool:
        """Write a frame to ffplay. Returns False if ffplay has exited."""
        if self.proc.poll() is not None:
            return False
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            return False
        return True

    def is_open(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=1)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
