"""
State management for ChunIVision touch and height tracking.

Manages:
- Current game state (32 touch zones + 6 height levels)
- State change detection (edge detection)
- Debouncing for noisy inputs
- State history buffer
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque
import numpy as np
import time


@dataclass
class TouchState:
    """
    Touch state for all 32 zones.

    Attributes:
        zones: Boolean array of shape (32,) indicating touch state for each zone
               Zone numbering: 1-32 (zone 1 = bottom-right, zone 32 = top-left)
        timestamp: Time when state was captured (seconds since epoch)
    """

    zones: np.ndarray  # shape (32,) dtype bool
    timestamp: float = 0.0

    def __post_init__(self):
        """Validate touch state."""
        if self.zones.shape != (32,):
            raise ValueError(f"TouchState must have 32 zones, got {self.zones.shape}")
        if self.zones.dtype != bool:
            self.zones = self.zones.astype(bool)


@dataclass
class HeightState:
    """
    Height state for 6 IR sensors.

    Attributes:
        levels: Integer array of shape (6,) with height level for each sensor (0-6)
                0 = no detection, 1-6 = height levels from bottom to top
        timestamp: Time when state was captured (seconds since epoch)
    """

    levels: np.ndarray  # shape (6,) dtype int
    timestamp: float = 0.0

    def __post_init__(self):
        """Validate height state."""
        if self.levels.shape != (6,):
            raise ValueError(f"HeightState must have 6 levels, got {self.levels.shape}")
        if self.levels.dtype != int:
            self.levels = self.levels.astype(int)


class StateManager:
    """
    Manages game state and detects changes.

    Features:
    - Tracks current touch and height state
    - Detects state changes (pressed/released)
    - Debouncing to filter noise
    - State history buffer
    """

    def __init__(self, debounce_frames: int = 2, history_size: int = 100):
        """
        Initialize state manager.

        Args:
            debounce_frames: Number of consecutive frames required to confirm state change
            history_size: Maximum number of historical states to keep
        """
        self.debounce_frames = debounce_frames
        self.history_size = history_size

        # Current confirmed state
        self._current_touch = TouchState(zones=np.zeros(32, dtype=bool))
        self._current_height = HeightState(levels=np.zeros(6, dtype=int))

        # Debounce buffers (stores recent raw states)
        self._touch_buffer: deque = deque(maxlen=debounce_frames)
        self._height_buffer: deque = deque(maxlen=debounce_frames)

        # State change flags
        self._touch_changes = np.zeros(
            32, dtype=int
        )  # -1=released, 0=no change, 1=pressed
        self._height_changes = np.zeros(
            6, dtype=int
        )  # -1=decreased, 0=no change, 1=increased

        # History buffers
        self._touch_history: deque = deque(maxlen=history_size)
        self._height_history: deque = deque(maxlen=history_size)

        # Statistics
        self._total_updates = 0
        self._total_changes = 0

    def update_state(self, touch_state: TouchState, height_state: HeightState) -> bool:
        """
        Update current state and detect changes.

        Args:
            touch_state: New touch state from detection
            height_state: New height state from detection

        Returns:
            True if state changed after debouncing, False otherwise
        """
        self._total_updates += 1

        # Add to debounce buffers
        self._touch_buffer.append(touch_state.zones.copy())
        self._height_buffer.append(height_state.levels.copy())

        # Check if we have enough frames for debouncing
        if len(self._touch_buffer) < self.debounce_frames:
            return False

        # Perform debouncing by majority vote
        touch_debounced = self._debounce_touch()
        height_debounced = self._debounce_height()

        # Detect changes
        touch_changed = np.any(touch_debounced != self._current_touch.zones)
        height_changed = np.any(height_debounced != self._current_height.levels)

        if touch_changed or height_changed:
            # Calculate change arrays
            self._touch_changes = self._calculate_touch_changes(touch_debounced)
            self._height_changes = self._calculate_height_changes(height_debounced)

            # Update current state
            self._current_touch = TouchState(
                zones=touch_debounced, timestamp=time.time()
            )
            self._current_height = HeightState(
                levels=height_debounced, timestamp=time.time()
            )

            # Add to history
            self._touch_history.append(self._current_touch)
            self._height_history.append(self._current_height)

            self._total_changes += 1
            return True

        # No change
        self._touch_changes = np.zeros(32, dtype=int)
        self._height_changes = np.zeros(6, dtype=int)
        return False

    def _debounce_touch(self) -> np.ndarray:
        """
        Debounce touch state using majority vote.

        Returns:
            Debounced touch state array (32,)
        """
        # Stack all buffered states
        stacked = np.stack(list(self._touch_buffer))  # shape (debounce_frames, 32)

        # Majority vote: zone is pressed if >= half of frames say so
        threshold = self.debounce_frames / 2.0
        return np.sum(stacked, axis=0) >= threshold

    def _debounce_height(self) -> np.ndarray:
        """
        Debounce height state using median.

        Returns:
            Debounced height state array (6,)
        """
        # Stack all buffered states
        stacked = np.stack(list(self._height_buffer))  # shape (debounce_frames, 6)

        # Use median for height levels
        return np.median(stacked, axis=0).astype(int)

    def _calculate_touch_changes(self, new_state: np.ndarray) -> np.ndarray:
        """
        Calculate touch state changes.

        Args:
            new_state: New debounced touch state

        Returns:
            Change array: -1 (released), 0 (no change), 1 (pressed)
        """
        old_state = self._current_touch.zones
        changes = np.zeros(32, dtype=int)

        # Pressed: was False, now True
        changes[~old_state & new_state] = 1

        # Released: was True, now False
        changes[old_state & ~new_state] = -1

        return changes

    def _calculate_height_changes(self, new_state: np.ndarray) -> np.ndarray:
        """
        Calculate height state changes.

        Args:
            new_state: New debounced height state

        Returns:
            Change array: -1 (decreased), 0 (no change), 1 (increased)
        """
        old_state = self._current_height.levels
        return np.sign(new_state - old_state).astype(int)

    def get_current_state(self) -> Tuple[TouchState, HeightState]:
        """
        Get current confirmed state.

        Returns:
            Tuple of (TouchState, HeightState)
        """
        return self._current_touch, self._current_height

    def get_changes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state changes since last update.

        Returns:
            Tuple of (touch_changes, height_changes)
            - touch_changes: array (32,) with -1 (released), 0 (no change), 1 (pressed)
            - height_changes: array (6,) with -1 (decreased), 0 (no change), 1 (increased)
        """
        return self._touch_changes.copy(), self._height_changes.copy()

    def get_pressed_zones(self) -> np.ndarray:
        """
        Get indices of currently pressed zones.

        Returns:
            Array of zone indices (1-based) that are currently pressed
        """
        return np.where(self._current_touch.zones)[0] + 1  # Convert to 1-based

    def get_history(self, num_frames: Optional[int] = None) -> Tuple[list, list]:
        """
        Get state history.

        Args:
            num_frames: Number of recent frames to return. If None, return all.

        Returns:
            Tuple of (touch_history, height_history)
        """
        if num_frames is None:
            return list(self._touch_history), list(self._height_history)

        return (
            list(self._touch_history)[-num_frames:],
            list(self._height_history)[-num_frames:],
        )

    def reset(self) -> None:
        """Reset state manager to initial state."""
        self._current_touch = TouchState(zones=np.zeros(32, dtype=bool))
        self._current_height = HeightState(levels=np.zeros(6, dtype=int))
        self._touch_buffer.clear()
        self._height_buffer.clear()
        self._touch_changes = np.zeros(32, dtype=int)
        self._height_changes = np.zeros(6, dtype=int)
        self._touch_history.clear()
        self._height_history.clear()
        self._total_updates = 0
        self._total_changes = 0

    def get_statistics(self) -> dict:
        """
        Get state manager statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_updates": self._total_updates,
            "total_changes": self._total_changes,
            "change_rate": self._total_changes / max(1, self._total_updates),
            "debounce_frames": self.debounce_frames,
            "history_size": len(self._touch_history),
            "currently_pressed_zones": len(self.get_pressed_zones()),
        }
