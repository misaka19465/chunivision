"""
Touch zone layout configuration for ChunIVision.

Defines the physical layout of 32 touch zones in a 2x16 grid.
Zone numbering: Bottom-right is zone 1, bottom row has odd numbers (31,29...1 from left),
top row has even numbers (32,30...2 from left).
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ZoneConfig:
    """
    Touch zone layout configuration.

    Zone numbering convention:
    - Total zones: 32 (2 rows × 16 columns)
    - Zone 1: Bottom-right corner
    - Bottom row (odd numbers): 31, 29, 27, ..., 3, 1 (left to right)
    - Top row (even numbers): 32, 30, 28, ..., 4, 2 (left to right)
    - Origin: Between columns 8 and 9, at bottom row level

    Attributes:
        num_rows: Number of rows (2)
        num_cols: Number of columns (16)
        zone_width: Width of each zone in cm
        zone_height: Height of each zone in cm
        origin_x: X coordinate of origin (center of grid) in cm
        origin_y: Y coordinate of origin (bottom edge) in cm
        touch_threshold_z: Z threshold for touch detection in cm
    """

    num_rows: int = 2
    num_cols: int = 16
    zone_width: float = 2.75  # cm (27.5mm per zone)
    zone_height: float = 4.5  # cm (45mm per zone)
    origin_x: float = 22.0  # cm (half of total width: 16 * 2.75 / 2)
    origin_y: float = 0.0  # cm (bottom row)
    touch_threshold_z: float = 2.0  # cm above surface

    def get_zone_boundary(self, zone_id: int) -> np.ndarray:
        """
        Get boundary polygon for a zone.

        Args:
            zone_id: Zone ID (1-32)

        Returns:
            Array of shape (4, 2) containing corner coordinates [x, y]
            Order: [bottom-left, bottom-right, top-right, top-left]

        Raises:
            ValueError: If zone_id is not in range 1-32
        """
        if not 1 <= zone_id <= 32:
            raise ValueError(f"Zone ID must be 1-32, got {zone_id}")

        # Get grid position
        row, col = self._zone_id_to_grid(zone_id)

        # Calculate corner positions relative to origin
        # Origin is at center-bottom of the grid
        x_min = (col - self.num_cols / 2.0) * self.zone_width
        x_max = x_min + self.zone_width
        y_min = row * self.zone_height
        y_max = y_min + self.zone_height

        # Create boundary polygon (4 corners)
        boundary = np.array(
            [
                [x_min, y_min],  # Bottom-left
                [x_max, y_min],  # Bottom-right
                [x_max, y_max],  # Top-right
                [x_min, y_max],  # Top-left
            ]
        )

        return boundary

    def get_zone_center(self, zone_id: int) -> np.ndarray:
        """
        Get center point of a zone.

        Args:
            zone_id: Zone ID (1-32)

        Returns:
            Array of shape (2,) containing center coordinates [x, y]

        Raises:
            ValueError: If zone_id is not in range 1-32
        """
        if not 1 <= zone_id <= 32:
            raise ValueError(f"Zone ID must be 1-32, got {zone_id}")

        # Get grid position
        row, col = self._zone_id_to_grid(zone_id)

        # Calculate center position relative to origin
        x = (col - self.num_cols / 2.0 + 0.5) * self.zone_width
        y = (row + 0.5) * self.zone_height

        return np.array([x, y])

    def _zone_id_to_grid(self, zone_id: int) -> tuple:
        """
        Convert zone ID to grid position (row, col).

        Zone numbering:
        - Bottom row (row 0): odd numbers from right to left (1, 3, 5, ..., 31)
        - Top row (row 1): even numbers from right to left (2, 4, 6, ..., 32)

        Args:
            zone_id: Zone ID (1-32)

        Returns:
            Tuple of (row, col) where row is 0-1 and col is 0-15
        """
        # Determine row (odd = bottom row 0, even = top row 1)
        row = 0 if zone_id % 2 == 1 else 1

        # Calculate column position
        # For bottom row (odd): zone_id=1 -> col=15, zone_id=3 -> col=14, ..., zone_id=31 -> col=0
        # For top row (even): zone_id=2 -> col=15, zone_id=4 -> col=14, ..., zone_id=32 -> col=0
        col = self.num_cols - 1 - (zone_id - 1) // 2

        return row, col

    def _grid_to_zone_id(self, row: int, col: int) -> int:
        """
        Convert grid position to zone ID.

        Args:
            row: Row index (0-1)
            col: Column index (0-15)

        Returns:
            Zone ID (1-32)

        Raises:
            ValueError: If row or col is out of bounds
        """
        if not 0 <= row < self.num_rows:
            raise ValueError(f"Row must be 0-{self.num_rows-1}, got {row}")
        if not 0 <= col < self.num_cols:
            raise ValueError(f"Column must be 0-{self.num_cols-1}, got {col}")

        # Reverse of _zone_id_to_grid
        # col=0 is leftmost, col=15 is rightmost
        # Bottom row (row=0): col=15 -> zone=1, col=14 -> zone=3, ..., col=0 -> zone=31
        # Top row (row=1): col=15 -> zone=2, col=14 -> zone=4, ..., col=0 -> zone=32
        zone_id = 2 * (self.num_cols - 1 - col) + 1 + row

        return zone_id

    def get_all_zone_centers(self) -> np.ndarray:
        """
        Get centers of all zones.

        Returns:
            Array of shape (32, 2) containing all zone centers [x, y]
            Indexed by zone_id - 1 (zone 1 at index 0, zone 32 at index 31)
        """
        centers = np.zeros((32, 2))
        for zone_id in range(1, 33):
            centers[zone_id - 1] = self.get_zone_center(zone_id)
        return centers

    def get_all_zone_boundaries(self) -> List[np.ndarray]:
        """
        Get boundaries of all zones.

        Returns:
            List of 32 arrays, each of shape (4, 2) containing zone corners
            Indexed by zone_id - 1 (zone 1 at index 0, zone 32 at index 31)
        """
        boundaries = []
        for zone_id in range(1, 33):
            boundaries.append(self.get_zone_boundary(zone_id))
        return boundaries

    def point_to_zone_id(self, x: float, y: float) -> int:
        """
        Find which zone contains a given point.

        Args:
            x: X coordinate in cm (relative to origin)
            y: Y coordinate in cm (relative to origin)

        Returns:
            Zone ID (1-32) if point is in a zone, 0 if outside all zones
        """
        # Check if point is within grid bounds
        x_min = -self.num_cols / 2.0 * self.zone_width
        x_max = self.num_cols / 2.0 * self.zone_width
        y_min = 0.0
        y_max = self.num_rows * self.zone_height

        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return 0

        # Calculate grid position
        col = int((x - x_min) / self.zone_width)
        row = int((y - y_min) / self.zone_height)

        # Clamp to valid range
        col = max(0, min(self.num_cols - 1, col))
        row = max(0, min(self.num_rows - 1, row))

        return self._grid_to_zone_id(row, col)

    def get_total_width(self) -> float:
        """Get total width of zone grid in cm."""
        return self.num_cols * self.zone_width

    def get_total_height(self) -> float:
        """Get total height of zone grid in cm."""
        return self.num_rows * self.zone_height
