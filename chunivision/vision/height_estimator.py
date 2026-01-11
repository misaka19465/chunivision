"""
Height Estimation module for ChunIVision.

Estimates which height levels are occupied by hands, emulating the behavior
of traditional IR sensor arrays used in Chunithm cabinets.

The Chunithm air sensor has 6 height levels:
- Air 0: 17.9 cm
- Air 1: 21.3 cm
- Air 2: 24.7 cm
- Air 3: 28.1 cm
- Air 4: 31.5 cm
- Air 5: 34.9 cm

Height levels are spaced 3.4 cm apart, starting at 17.9 cm from the touch surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import Logger
from .hand_detector import Hand

logger = Logger.get_logger(__name__)


# Default height thresholds matching Chunithm air sensor positions (in cm)
DEFAULT_HEIGHT_THRESHOLDS: List[float] = [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]


@dataclass
class HeightState:
    """
    Represents state of 6 height levels.

    The levels array indicates which height levels currently have a hand present.
    Level 0 is the lowest (closest to touch surface), Level 5 is the highest.

    Attributes:
        levels: Boolean array of size 6 (True = hand present at level)
        timestamp: State timestamp in seconds since epoch
        exact_heights: Optional list of exact hand heights in cm
        active_levels: List of level indices that are currently active (0-5)
    """

    levels: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))
    timestamp: float = 0.0
    exact_heights: Optional[List[float]] = None
    active_levels: Optional[List[int]] = None

    def __post_init__(self) -> None:
        """Ensure levels array is correct type and shape."""
        if not isinstance(self.levels, np.ndarray):
            self.levels = np.array(self.levels, dtype=bool)
        if self.levels.shape != (6,):
            raise ValueError(f"levels must have shape (6,), got {self.levels.shape}")
        # Compute active levels if not provided
        if self.active_levels is None:
            self.active_levels = [i for i in range(6) if self.levels[i]]

    def is_active(self, level: int) -> bool:
        """
        Check if a specific height level is active.

        Args:
            level: Level index (0-5)

        Returns:
            True if level is active, False otherwise

        Raises:
            ValueError: If level is not in range 0-5
        """
        if not 0 <= level <= 5:
            raise ValueError(f"Level must be 0-5, got {level}")
        return bool(self.levels[level])

    def get_active_count(self) -> int:
        """Get number of currently active height levels."""
        return int(np.sum(self.levels))

    def get_highest_active_level(self) -> Optional[int]:
        """
        Get the highest active height level.

        Returns:
            Highest active level (0-5) or None if no levels active
        """
        for level in range(5, -1, -1):
            if self.levels[level]:
                return level
        return None

    def get_lowest_active_level(self) -> Optional[int]:
        """
        Get the lowest active height level.

        Returns:
            Lowest active level (0-5) or None if no levels active
        """
        for level in range(6):
            if self.levels[level]:
                return level
        return None

    def to_byte(self) -> int:
        """
        Convert height state to a single byte.

        Returns:
            Byte with bits 0-5 representing levels 0-5
        """
        result = 0
        for i in range(6):
            if self.levels[i]:
                result |= 1 << i
        return result

    def to_bytes(self) -> bytes:
        """
        Convert height state to bytes representation.

        Returns:
            Single byte representing all 6 levels
        """
        return bytes([self.to_byte()])

    @classmethod
    def from_byte(cls, data: int, timestamp: float = 0.0) -> "HeightState":
        """
        Create HeightState from a byte value.

        Args:
            data: Byte with bits 0-5 representing levels 0-5
            timestamp: State timestamp

        Returns:
            HeightState object
        """
        levels = np.zeros(6, dtype=bool)
        for i in range(6):
            levels[i] = bool(data & (1 << i))
        return cls(levels=levels, timestamp=timestamp)

    @classmethod
    def from_bytes(cls, data: bytes, timestamp: float = 0.0) -> "HeightState":
        """
        Create HeightState from bytes.

        Args:
            data: At least 1 byte representing 6 levels
            timestamp: State timestamp

        Returns:
            HeightState object
        """
        if len(data) < 1:
            raise ValueError(f"Expected at least 1 byte, got {len(data)}")
        return cls.from_byte(data[0], timestamp)

    def __repr__(self) -> str:
        active_count = self.get_active_count()
        levels_str = ",".join(str(lvl) for lvl in (self.active_levels or []))
        return f"HeightState(active={active_count}, levels=[{levels_str}])"


@dataclass
class HeightEstimatorConfig:
    """
    Configuration for height estimation.

    Attributes:
        height_thresholds: Height thresholds for each level in cm.
                          A hand at or above threshold[i] triggers level i.
        hysteresis: Hysteresis margin in cm to prevent rapid level switching.
        min_confidence: Minimum hand confidence for height detection.
        debounce_frames: Number of consecutive frames for state change.
        x_range: Optional (min_x, max_x) range for valid air detection.
                 Hands outside this range are ignored for height detection.
        cumulative_levels: If True, all levels below the detected level are
                          also activated (like breaking IR beams).
    """

    height_thresholds: List[float] = field(
        default_factory=lambda: DEFAULT_HEIGHT_THRESHOLDS.copy()
    )
    hysteresis: float = 1.0  # cm
    min_confidence: float = 0.3
    debounce_frames: int = 2
    x_range: Optional[Tuple[float, float]] = None  # None = no X filtering
    cumulative_levels: bool = True  # Chunithm-style: hand at level 3 also triggers 0-2

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if len(self.height_thresholds) != 6:
            errors.append(
                f"height_thresholds must have 6 values, got {len(self.height_thresholds)}"
            )

        # Check thresholds are monotonically increasing
        for i in range(1, len(self.height_thresholds)):
            if self.height_thresholds[i] <= self.height_thresholds[i - 1]:
                errors.append(
                    f"height_thresholds must be strictly increasing: "
                    f"threshold[{i}]={self.height_thresholds[i]} <= "
                    f"threshold[{i - 1}]={self.height_thresholds[i - 1]}"
                )

        if self.hysteresis < 0:
            errors.append(f"hysteresis must be >= 0, got {self.hysteresis}")

        if not 0 <= self.min_confidence <= 1:
            errors.append(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )

        if self.debounce_frames < 0:
            errors.append(f"debounce_frames must be >= 0, got {self.debounce_frames}")

        if self.x_range is not None:
            if self.x_range[0] >= self.x_range[1]:
                errors.append(f"x_range[0] must be < x_range[1], got {self.x_range}")

        return errors


class HeightEstimator:
    """
    Assigns hands to discrete height levels (emulates IR sensor array).

    The Chunithm air sensor uses 6 IR beam pairs to detect hand height.
    This class emulates that behavior by mapping 3D hand positions to
    discrete height levels.

    When cumulative_levels is True (default), a hand at height H activates
    all levels whose threshold is <= H. This matches the physical behavior
    where a hand at height H would break all IR beams below it.
    """

    def __init__(
        self,
        config: Optional[HeightEstimatorConfig] = None,
    ):
        """
        Initialize with height level configuration.

        Args:
            config: Height detection parameters. Uses default if None.
        """
        self.config = config or HeightEstimatorConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Pre-compute threshold arrays for efficient comparison
        self._thresholds = np.array(self.config.height_thresholds, dtype=np.float64)
        self._release_thresholds = self._thresholds - self.config.hysteresis

        # State for debouncing
        self._pending_activations: Dict[int, int] = {}  # level -> consecutive frames
        self._pending_deactivations: Dict[int, int] = {}  # level -> consecutive frames

        # Current state
        self._current_state = HeightState(timestamp=time.time())

        # History for debugging
        self._state_history: List[HeightState] = []
        self._max_history = 100

        logger.debug(
            f"HeightEstimator initialized with thresholds={self.config.height_thresholds}, "
            f"hysteresis={self.config.hysteresis}cm"
        )

    def estimate_heights(self, hands: List[Hand]) -> HeightState:
        """
        Determine which height levels are occupied.

        Args:
            hands: List of detected hands from HandDetector

        Returns:
            HeightState indicating occupied levels
        """
        timestamp = time.time()

        # Filter hands by confidence
        valid_hands = [h for h in hands if h.confidence >= self.config.min_confidence]

        # Filter by X range if configured
        if self.config.x_range is not None:
            x_min, x_max = self.config.x_range
            valid_hands = [h for h in valid_hands if x_min <= h.x <= x_max]

        # Determine raw level activations (before debouncing)
        raw_levels, exact_heights = self._compute_active_levels(valid_hands)

        # Apply debouncing
        final_levels = self._apply_debouncing(raw_levels)

        # Create new state
        levels = np.zeros(6, dtype=bool)
        for level in final_levels:
            levels[level] = True

        new_state = HeightState(
            levels=levels,
            timestamp=timestamp,
            exact_heights=exact_heights if exact_heights else None,
            active_levels=list(final_levels),
        )

        # Update current state
        self._current_state = new_state

        # Store history
        self._state_history.append(new_state)
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        return new_state

    def _compute_active_levels(self, hands: List[Hand]) -> Tuple[set, List[float]]:
        """
        Compute which levels are active based on hand heights.

        Args:
            hands: List of valid hands

        Returns:
            Tuple of (set of active level indices, list of exact heights)
        """
        active_levels = set()
        exact_heights = []
        current_active = set(i for i in range(6) if self._current_state.levels[i])

        for hand in hands:
            height = hand.z
            exact_heights.append(height)

            # Find which levels this hand activates
            hand_levels = self._height_to_levels(height, current_active)
            active_levels.update(hand_levels)

        return active_levels, exact_heights

    def _height_to_levels(self, height: float, current_active: set) -> set:
        """
        Convert a height value to active level indices.

        Uses hysteresis: requires higher threshold to activate than to deactivate.

        Args:
            height: Hand height in cm
            current_active: Currently active levels (for hysteresis)

        Returns:
            Set of level indices that should be active
        """
        levels = set()

        for level in range(6):
            threshold = self._thresholds[level]
            release_threshold = self._release_thresholds[level]

            # Apply hysteresis
            if level in current_active:
                # Level is currently active - use lower threshold to maintain
                is_active = height >= release_threshold
            else:
                # Level is currently inactive - use higher threshold to activate
                is_active = height >= threshold

            if is_active:
                if self.config.cumulative_levels:
                    # Activate this level and all levels below
                    for l in range(level + 1):
                        levels.add(l)
                else:
                    levels.add(level)

        return levels

    def _apply_debouncing(self, raw_levels: set) -> set:
        """
        Apply debouncing to prevent rapid level switching.

        Args:
            raw_levels: Set of levels currently active (raw)

        Returns:
            Set of levels after debouncing applied
        """
        if self.config.debounce_frames == 0:
            return raw_levels

        current_active = set(i for i in range(6) if self._current_state.levels[i])
        final_levels = current_active.copy()

        # Handle new activations
        for level in raw_levels:
            if level not in current_active:
                self._pending_activations[level] = (
                    self._pending_activations.get(level, 0) + 1
                )
                if self._pending_activations[level] >= self.config.debounce_frames:
                    final_levels.add(level)
                    del self._pending_activations[level]
            else:
                # Reset pending deactivation counter
                if level in self._pending_deactivations:
                    del self._pending_deactivations[level]

        # Handle deactivations
        for level in current_active:
            if level not in raw_levels:
                self._pending_deactivations[level] = (
                    self._pending_deactivations.get(level, 0) + 1
                )
                if self._pending_deactivations[level] >= self.config.debounce_frames:
                    final_levels.discard(level)
                    del self._pending_deactivations[level]
            else:
                # Reset pending activation counter
                if level in self._pending_activations:
                    del self._pending_activations[level]

        # Clean up stale pending counts
        stale_pending = [l for l in self._pending_activations if l not in raw_levels]
        for l in stale_pending:
            del self._pending_activations[l]

        stale_deact = [l for l in self._pending_deactivations if l in raw_levels]
        for l in stale_deact:
            del self._pending_deactivations[l]

        return final_levels

    def set_thresholds(self, thresholds: List[float]) -> None:
        """
        Update height level thresholds.

        Args:
            thresholds: List of 6 threshold values in cm (must be increasing)

        Raises:
            ValueError: If thresholds are invalid
        """
        if len(thresholds) != 6:
            raise ValueError(f"Expected 6 thresholds, got {len(thresholds)}")

        for i in range(1, 6):
            if thresholds[i] <= thresholds[i - 1]:
                raise ValueError(
                    f"Thresholds must be strictly increasing: "
                    f"threshold[{i}]={thresholds[i]} <= threshold[{i - 1}]={thresholds[i - 1]}"
                )

        self.config.height_thresholds = list(thresholds)
        self._thresholds = np.array(thresholds, dtype=np.float64)
        self._release_thresholds = self._thresholds - self.config.hysteresis

        logger.info(f"Height thresholds updated to {thresholds}")

    def set_hysteresis(self, hysteresis: float) -> None:
        """
        Update hysteresis margin.

        Args:
            hysteresis: Hysteresis margin in cm

        Raises:
            ValueError: If hysteresis is negative
        """
        if hysteresis < 0:
            raise ValueError(f"Hysteresis must be >= 0, got {hysteresis}")

        self.config.hysteresis = hysteresis
        self._release_thresholds = self._thresholds - hysteresis

        logger.info(f"Hysteresis updated to {hysteresis}cm")

    def set_x_range(self, x_range: Optional[Tuple[float, float]]) -> None:
        """
        Set X-coordinate range for valid height detection.

        Args:
            x_range: (min_x, max_x) tuple or None to disable filtering
        """
        if x_range is not None and x_range[0] >= x_range[1]:
            raise ValueError(f"x_range[0] must be < x_range[1], got {x_range}")

        self.config.x_range = x_range
        logger.info(f"X range updated to {x_range}")

    def get_threshold(self, level: int) -> float:
        """
        Get threshold for a specific level.

        Args:
            level: Level index (0-5)

        Returns:
            Threshold height in cm

        Raises:
            ValueError: If level is not in range 0-5
        """
        if not 0 <= level <= 5:
            raise ValueError(f"Level must be 0-5, got {level}")
        return float(self._thresholds[level])

    def get_all_thresholds(self) -> List[float]:
        """Get all 6 height thresholds."""
        return list(self._thresholds)

    def get_current_state(self) -> HeightState:
        """Get the current height state."""
        return self._current_state

    def get_state_history(self) -> List[HeightState]:
        """Get recent state history for debugging."""
        return self._state_history.copy()

    def reset(self) -> None:
        """Reset estimator state (clear all levels and pending counts)."""
        self._current_state = HeightState(timestamp=time.time())
        self._pending_activations.clear()
        self._pending_deactivations.clear()
        self._state_history.clear()
        logger.debug("HeightEstimator state reset")

    def height_to_level(self, height: float) -> Optional[int]:
        """
        Get the highest level that a given height would trigger.

        This is a utility method for testing/debugging.

        Args:
            height: Height in cm

        Returns:
            Highest level index (0-5) or None if below all thresholds
        """
        highest_level = None
        for level in range(6):
            if height >= self._thresholds[level]:
                highest_level = level
        return highest_level

    def level_to_height_range(self, level: int) -> Tuple[float, float]:
        """
        Get the height range for a specific level.

        Args:
            level: Level index (0-5)

        Returns:
            Tuple of (min_height, max_height) in cm

        Raises:
            ValueError: If level is not in range 0-5
        """
        if not 0 <= level <= 5:
            raise ValueError(f"Level must be 0-5, got {level}")

        min_height = self._thresholds[level]

        if level == 5:
            # Highest level has no upper bound
            max_height = float("inf")
        else:
            max_height = self._thresholds[level + 1]

        return min_height, max_height
