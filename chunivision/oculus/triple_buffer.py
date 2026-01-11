"""Small triple buffer helper used by the viewer and camera callbacks."""
from __future__ import annotations

import threading
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class TripleBuffer(Generic[T]):
    """A tiny triple buffer that keeps the latest value without blocking.

    This is not lock-free, but the locks are short-lived and fine for the
    USB/event threads used here.
    """

    def __init__(self) -> None:
        self._buffers: list[Optional[T]] = [None, None, None]
        self._display = 0
        self._pending = 1
        self._scratch = 2
        self._lock = threading.Lock()

    def publish(self, value: T) -> None:
        """Store a new value, making it the next frame for readers."""
        with self._lock:
            self._buffers[self._pending] = value
            # Rotate roles: display<-pending, pending<-scratch, scratch<-display
            self._display, self._pending, self._scratch = (
                self._pending,
                self._scratch,
                self._display,
            )

    def get_latest(self) -> Optional[T]:
        """Return the most recently published value (may be None)."""
        with self._lock:
            return self._buffers[self._display]

    def clear(self) -> None:
        with self._lock:
            self._buffers = [None, None, None]
            self._display, self._pending, self._scratch = 0, 1, 2
