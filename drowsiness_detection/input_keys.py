"""Non-blocking single-keypress terminal input (replaces cv2.waitKey)."""

import sys
import select
import termios
import tty


class TerminalKeyReader:
    """Reads single keypresses from the terminal without blocking / needing Enter."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._old_settings = None
        if self.enabled:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def get_key(self):
        """Return a single character if available, else None. Non-blocking."""
        if not self.enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None

    def restore(self):
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
