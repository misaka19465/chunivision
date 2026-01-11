"""
Touch Detection module for ChunIVision.

Detects which touch zones are being touched based on hand positions.
Maps 3D hand coordinates to the 32-zone touch grid (2 rows × 16 columns).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.zone_config import ZoneConfig
from ..utils.logger import Logger
from .hand_detector import Hand

logger = Logger.get_logger(__name__)


@dataclass
class TouchState:
    """
    Represents state of all 32 touch zones.

    Attributes:
        zones: Boolean array of size 32 (True = touched, indexed by zone_id - 1)
        timestamp: State timestamp in seconds since epoch
        touch_positions: Optional list of exact touch positions (x, y, z) for each touch
        active_zone_ids: List of zone IDs that are currently touched (1-32)
    """

    zones: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=bool))
    timestamp: float = 0.0
    touch_positions: Optional[List[np.ndarray]] = None
    active_zone_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        """Ensure zones array is correct type and shape."""
        if not isinstance(self.zones, np.ndarray):
            self.zones = np.array(self.zones, dtype=bool)
        if self.zones.shape != (32,):
            raise ValueError(f"zones must have shape (32,), got {self.zones.shape}")
        # Compute active zones if not provided
        if self.active_zone_ids is None:
            self.active_zone_ids = [i + 1 for i in range(32) if self.zones[i]]

    def is_touched(self, zone_id: int) -> bool:
        """
        Check if a specific zone is touched.

        Args:
            zone_id: Zone ID (1-32)

        Returns:
            True if zone is touched, False otherwise

        Raises:
            ValueError: If zone_id is not in range 1-32
        """
        if not 1 <= zone_id <= 32:
            raise ValueError(f"Zone ID must be 1-32, got {zone_id}")
        return bool(self.zones[zone_id - 1])

    def get_touched_count(self) -> int:
        """Get number of currently touched zones."""
        return int(np.sum(self.zones))

    def to_bytes(self) -> bytes:
        """
        Convert touch state to 4-byte packed format.

        Returns:
            4 bytes representing 32 zones (1 bit per zone)
            Zone 1 is bit 0 of byte 0, Zone 8 is bit 7 of byte 0, etc.
        """
        result = bytearray(4)
        for i in range(32):
            if self.zones[i]:
                byte_idx = i // 8
                bit_idx = i % 8
                result[byte_idx] |= 1 << bit_idx
        return bytes(result)

    @classmethod
    def from_bytes(cls, data: bytes, timestamp: float = 0.0) -> "TouchState":
        """
        Create TouchState from packed bytes.

        Args:
            data: 4 bytes representing 32 zones
            timestamp: State timestamp

        Returns:
            TouchState object
        """
        if len(data) < 4:
            raise ValueError(f"Expected at least 4 bytes, got {len(data)}")

        zones = np.zeros(32, dtype=bool)
        for i in range(32):
            byte_idx = i // 8
            bit_idx = i % 8
            zones[i] = bool(data[byte_idx] & (1 << bit_idx))

        return cls(zones=zones, timestamp=timestamp)

    def __repr__(self) -> str:
        touched = self.get_touched_count()
        zones_str = ",".join(str(z) for z in (self.active_zone_ids or []))
        return f"TouchState(touched={touched}, zones=[{zones_str}])"


@dataclass
class TouchDetectorConfig:
    """
    Configuration for touch detection.

    Attributes:
        touch_threshold_z: Z-coordinate threshold for touch detection in cm.
                          Hands below this height are considered touching.
        release_threshold_z: Z-coordinate for release (with hysteresis) in cm.
        min_confidence: Minimum hand confidence for touch detection.
        debounce_frames: Number of consecutive frames for state change.
        overlap_priority: How to handle overlapping touches ('lowest_z', 'highest_conf').
        enable_multi_touch: Allow multiple zones to be touched simultaneously.
    """

    touch_threshold_z: float = 2.0  # cm
    release_threshold_z: float = 3.0  # cm (hysteresis)
    min_confidence: float = 0.3
    debounce_frames: int = 2
    overlap_priority: str = "lowest_z"
    enable_multi_touch: bool = True

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if self.touch_threshold_z < 0:
            errors.append(
                f"touch_threshold_z must be >= 0, got {self.touch_threshold_z}"
            )
        if self.release_threshold_z < self.touch_threshold_z:
            errors.append(
                "release_threshold_z must be >= touch_threshold_z for hysteresis"
            )
        if not 0 <= self.min_confidence <= 1:
            errors.append(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )
        if self.debounce_frames < 0:
            errors.append(f"debounce_frames must be >= 0, got {self.debounce_frames}")
        if self.overlap_priority not in ("lowest_z", "highest_conf"):
            errors.append(
                f"overlap_priority must be 'lowest_z' or 'highest_conf', "
                f"got {self.overlap_priority}"
            )

        return errors


class TouchDetector:
    """
    Determines which touch zones are being touched based on hand positions.

    This class maps 3D hand positions to the 32-zone touch grid, handling
    multi-touch scenarios and implementing hysteresis for stable touch detection.
    """

    def __init__(
        self,
        zone_config: Optional[ZoneConfig] = None,
        config: Optional[TouchDetectorConfig] = None,
    ):
        """
        Initialize with zone layout and detection configuration.

        Args:
            zone_config: Defines zone boundaries and layout. Uses default if None.
            config: Detection parameters. Uses default if None.
        """
        self.zone_config = zone_config or ZoneConfig()
        self.config = config or TouchDetectorConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Pre-compute zone boundaries and centers for efficiency
        self._zone_boundaries = self.zone_config.get_all_zone_boundaries()
        self._zone_centers = self.zone_config.get_all_zone_centers()

        # State for debouncing
        self._pending_touches: Dict[int, int] = {}  # zone_id -> consecutive frame count
        self._pending_releases: Dict[int, int] = (
            {}
        )  # zone_id -> consecutive frame count

        # Current state
        self._current_state = TouchState(timestamp=time.time())

        # History for debugging
        self._state_history: List[TouchState] = []
        self._max_history = 100

        logger.debug(
            f"TouchDetector initialized with threshold_z={self.config.touch_threshold_z}cm"
        )

    def detect_touches(self, hands: List[Hand]) -> TouchState:
        """
        Determine touch state from hand positions.

        Args:
            hands: List of detected hands from HandDetector

        Returns:
            TouchState indicating which zones are touched
        """
        timestamp = time.time()

        # Filter hands by confidence
        valid_hands = [h for h in hands if h.confidence >= self.config.min_confidence]

        # Determine raw touches (before debouncing)
        raw_touches = self._map_hands_to_zones(valid_hands)

        # Apply debouncing
        final_touches = self._apply_debouncing(raw_touches)

        # Collect touch positions for active zones
        touch_positions = []
        for hand in valid_hands:
            if hand.z <= self.config.touch_threshold_z:
                touch_positions.append(hand.position.copy())

        # Create new state
        zones = np.zeros(32, dtype=bool)
        for zone_id in final_touches:
            zones[zone_id - 1] = True

        new_state = TouchState(
            zones=zones,
            timestamp=timestamp,
            touch_positions=touch_positions if touch_positions else None,
            active_zone_ids=list(final_touches),
        )

        # Update current state
        self._current_state = new_state

        # Store history
        self._state_history.append(new_state)
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        return new_state

    def _map_hands_to_zones(self, hands: List[Hand]) -> set:
        """
        Map hand positions to zone IDs.

        Args:
            hands: List of valid hands

        Returns:
            Set of zone IDs that are being touched
        """
        touched_zones = set()
        zone_touch_info: Dict[int, List[Tuple[float, float]]] = (
            {}
        )  # zone -> [(z, conf), ...]

        for hand in hands:
            # Check if hand is below touch threshold (touching)
            if hand.z > self.config.touch_threshold_z:
                continue

            # Find which zone this hand is in (use X, Y coordinates)
            zone_id = self.zone_config.point_to_zone_id(hand.x, hand.y)

            if zone_id == 0:
                # Hand is outside zone grid
                continue

            # Track touch info for overlap resolution
            if zone_id not in zone_touch_info:
                zone_touch_info[zone_id] = []
            zone_touch_info[zone_id].append((hand.z, hand.confidence))

            if self.config.enable_multi_touch:
                # Also check adjacent zones for large hand coverage
                self._check_adjacent_zones(hand, touched_zones, zone_touch_info)

            touched_zones.add(zone_id)

        return touched_zones

    def _check_adjacent_zones(
        self,
        hand: Hand,
        touched_zones: set,
        zone_touch_info: Dict[int, List[Tuple[float, float]]],
    ) -> None:
        """
        Check if hand might be touching adjacent zones based on hand size.

        A hand might cover multiple zones if it's positioned near zone boundaries.
        This only checks directly adjacent zones and requires the hand position
        to be very close to the zone boundary.
        """
        # Only trigger adjacent zone detection if hand is very close to a boundary
        # This is a conservative threshold - about 30% into the adjacent zone
        boundary_margin = 0.5  # cm - hand must be within this distance of boundary

        # Get the primary zone the hand is in
        primary_zone_id = self.zone_config.point_to_zone_id(hand.x, hand.y)
        if primary_zone_id == 0:
            return

        # Get primary zone boundary
        primary_boundary = self._zone_boundaries[primary_zone_id - 1]
        x_min = primary_boundary[:, 0].min()
        x_max = primary_boundary[:, 0].max()
        y_min = primary_boundary[:, 1].min()
        y_max = primary_boundary[:, 1].max()

        # Check if hand is near any boundary
        near_left = hand.x - x_min < boundary_margin
        near_right = x_max - hand.x < boundary_margin
        near_bottom = hand.y - y_min < boundary_margin
        near_top = y_max - hand.y < boundary_margin

        # Get grid position of primary zone
        row, col = self.zone_config._zone_id_to_grid(primary_zone_id)

        # Check adjacent zones only if hand is near their boundary
        adjacent_zones = []

        if near_left and col > 0:
            # Zone to the left
            adj_zone = self.zone_config._grid_to_zone_id(row, col - 1)
            adjacent_zones.append(adj_zone)

        if near_right and col < self.zone_config.num_cols - 1:
            # Zone to the right
            adj_zone = self.zone_config._grid_to_zone_id(row, col + 1)
            adjacent_zones.append(adj_zone)

        if near_bottom and row > 0:
            # Zone below (should not happen for row 0)
            adj_zone = self.zone_config._grid_to_zone_id(row - 1, col)
            adjacent_zones.append(adj_zone)

        if near_top and row < self.zone_config.num_rows - 1:
            # Zone above
            adj_zone = self.zone_config._grid_to_zone_id(row + 1, col)
            adjacent_zones.append(adj_zone)

        for zone_id in adjacent_zones:
            if zone_id not in touched_zones:
                touched_zones.add(zone_id)
                if zone_id not in zone_touch_info:
                    zone_touch_info[zone_id] = []
                # Lower confidence for adjacent zone touches
                zone_touch_info[zone_id].append((hand.z, hand.confidence * 0.5))

    def _point_in_zone(
        self, x: float, y: float, boundary: np.ndarray, margin: float = 0.0
    ) -> bool:
        """
        Check if a point is within a zone boundary.

        Args:
            x: X coordinate
            y: Y coordinate
            boundary: Zone boundary array of shape (4, 2)
            margin: Additional margin around zone

        Returns:
            True if point is within zone (plus margin)
        """
        x_min = boundary[:, 0].min() - margin
        x_max = boundary[:, 0].max() + margin
        y_min = boundary[:, 1].min() - margin
        y_max = boundary[:, 1].max() + margin

        return x_min <= x <= x_max and y_min <= y <= y_max

    def _apply_debouncing(self, raw_touches: set) -> set:
        """
        Apply debouncing to prevent rapid touch/release flicker.

        Args:
            raw_touches: Set of zones currently being touched (raw)

        Returns:
            Set of zones after debouncing applied
        """
        if self.config.debounce_frames == 0:
            return raw_touches

        current_active = set(i + 1 for i in range(32) if self._current_state.zones[i])
        final_touches = current_active.copy()

        # Handle new touches
        for zone_id in raw_touches:
            if zone_id not in current_active:
                self._pending_touches[zone_id] = (
                    self._pending_touches.get(zone_id, 0) + 1
                )
                if self._pending_touches[zone_id] >= self.config.debounce_frames:
                    final_touches.add(zone_id)
                    del self._pending_touches[zone_id]
            else:
                # Reset pending release counter
                if zone_id in self._pending_releases:
                    del self._pending_releases[zone_id]

        # Handle releases
        for zone_id in current_active:
            if zone_id not in raw_touches:
                self._pending_releases[zone_id] = (
                    self._pending_releases.get(zone_id, 0) + 1
                )
                if self._pending_releases[zone_id] >= self.config.debounce_frames:
                    final_touches.discard(zone_id)
                    del self._pending_releases[zone_id]
            else:
                # Reset pending touch counter
                if zone_id in self._pending_touches:
                    del self._pending_touches[zone_id]

        # Clean up stale pending counts for zones not in current consideration
        stale_pending = [z for z in self._pending_touches if z not in raw_touches]
        for z in stale_pending:
            del self._pending_touches[z]

        stale_releases = [z for z in self._pending_releases if z in raw_touches]
        for z in stale_releases:
            del self._pending_releases[z]

        return final_touches

    def set_touch_threshold(self, z_threshold: float) -> None:
        """
        Set Z-coordinate threshold for touch detection.

        Args:
            z_threshold: Height in cm below which hand is considered touching
        """
        if z_threshold < 0:
            raise ValueError(f"z_threshold must be >= 0, got {z_threshold}")

        self.config.touch_threshold_z = z_threshold

        # Adjust release threshold to maintain hysteresis
        if self.config.release_threshold_z < z_threshold:
            self.config.release_threshold_z = z_threshold + 1.0

        logger.info(f"Touch threshold updated to {z_threshold}cm")

    def set_release_threshold(self, z_threshold: float) -> None:
        """
        Set Z-coordinate threshold for release detection (hysteresis).

        Args:
            z_threshold: Height in cm above which touch is released
        """
        if z_threshold < self.config.touch_threshold_z:
            raise ValueError(
                f"release_threshold must be >= touch_threshold "
                f"({self.config.touch_threshold_z}), got {z_threshold}"
            )

        self.config.release_threshold_z = z_threshold
        logger.info(f"Release threshold updated to {z_threshold}cm")

    def get_zone_boundary(self, zone_id: int) -> np.ndarray:
        """
        Get boundary polygon for a specific zone.

        Args:
            zone_id: Zone number (1-32)

        Returns:
            Array of 2D boundary points

        Raises:
            ValueError: If zone_id is not in range 1-32
        """
        if not 1 <= zone_id <= 32:
            raise ValueError(f"Zone ID must be 1-32, got {zone_id}")
        return self._zone_boundaries[zone_id - 1].copy()

    def get_zone_center(self, zone_id: int) -> np.ndarray:
        """
        Get center point of a specific zone.

        Args:
            zone_id: Zone number (1-32)

        Returns:
            Array of 2D center coordinates [x, y]

        Raises:
            ValueError: If zone_id is not in range 1-32
        """
        if not 1 <= zone_id <= 32:
            raise ValueError(f"Zone ID must be 1-32, got {zone_id}")
        return self._zone_centers[zone_id - 1].copy()

    def get_current_state(self) -> TouchState:
        """Get the current touch state."""
        return self._current_state

    def get_state_history(self) -> List[TouchState]:
        """Get recent state history for debugging."""
        return self._state_history.copy()

    def reset(self) -> None:
        """Reset detector state (clear all touches and pending counts)."""
        self._current_state = TouchState(timestamp=time.time())
        self._pending_touches.clear()
        self._pending_releases.clear()
        self._state_history.clear()
        logger.debug("TouchDetector state reset")

    def get_zone_for_position(self, x: float, y: float) -> int:
        """
        Get zone ID for a given X, Y position.

        Args:
            x: X coordinate in cm
            y: Y coordinate in cm

        Returns:
            Zone ID (1-32) or 0 if outside all zones
        """
        return self.zone_config.point_to_zone_id(x, y)
