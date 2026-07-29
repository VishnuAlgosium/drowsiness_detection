"""
keyboard_input.py
-----------------
Non-blocking single-key terminal reader (used for the 'v' / 'q' hotkeys).
"""

import sys
import select
import termios
import tty
from typing import Optional


class KeyReader:
    """Reads a single keypress from stdin without blocking, when run in a TTY."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._old = None

        if self.enabled:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def get_key(self) -> Optional[str]:
        if not self.enabled:
            return None

        r, _, _ = select.select([sys.stdin], [], [], 0)

        if r:
            return sys.stdin.read(1)

        return None

    def restore(self) -> None:
        if self.enabled and self._old:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
