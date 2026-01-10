"""
Camera hardware configuration for ChunIVision.

Defines camera positions, orientations, and capture settings.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class CameraConfig:
    """
    Camera hardware configuration.

    Attributes:
        left_camera_index: Camera device index for left camera
        right_camera_index: Camera device index for right camera
        resolution: Camera resolution as (width, height)
        fps: Target frame rate
        exposure: Exposure setting (-1 for auto)
        left_camera_position: 3D position of left camera [x, y, z] in cm
        right_camera_position: 3D position of right camera [x, y, z] in cm
        baseline_distance: Distance between cameras in cm
        left_camera_angle: Angle of left camera in degrees
        right_camera_angle: Angle of right camera in degrees
    """

    left_camera_index: int = 0
    right_camera_index: int = 1
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 60
    exposure: int = -1  # -1 for auto
    left_camera_position: np.ndarray = field(
        default_factory=lambda: np.array([-10.0, 15.0, 30.0])
    )
    right_camera_position: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 15.0, 30.0])
    )
    baseline_distance: float = 20.0  # cm
    left_camera_angle: float = 45.0  # degrees
    right_camera_angle: float = 45.0  # degrees

    def __post_init__(self):
        """Validate and convert configuration after initialization."""
        # Ensure camera positions are numpy arrays
        if not isinstance(self.left_camera_position, np.ndarray):
            if isinstance(self.left_camera_position, (list, tuple)):
                self.left_camera_position = np.array(self.left_camera_position)
            else:
                raise TypeError("left_camera_position must be array-like")

        if not isinstance(self.right_camera_position, np.ndarray):
            if isinstance(self.right_camera_position, (list, tuple)):
                self.right_camera_position = np.array(self.right_camera_position)
            else:
                raise TypeError("right_camera_position must be array-like")

        # Ensure resolution is a tuple
        if isinstance(self.resolution, list):
            if len(self.resolution) == 2:
                object.__setattr__(
                    self, "resolution", (self.resolution[0], self.resolution[1])
                )

        # Validate
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid camera configuration: {', '.join(errors)}")

    def validate(self) -> List[str]:
        """
        Validate camera configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate camera indices
        if self.left_camera_index < 0:
            errors.append(
                f"Invalid left camera index: {self.left_camera_index} (must be >= 0)"
            )

        if self.right_camera_index < 0:
            errors.append(
                f"Invalid right camera index: {self.right_camera_index} (must be >= 0)"
            )

        if self.left_camera_index == self.right_camera_index:
            errors.append("Left and right camera indices must be different")

        # Validate resolution
        if len(self.resolution) != 2:
            errors.append(f"Resolution must be (width, height), got {self.resolution}")
        elif self.resolution[0] <= 0 or self.resolution[1] <= 0:
            errors.append(f"Invalid resolution {self.resolution} (must be > 0)")

        # Validate FPS
        if self.fps <= 0:
            errors.append(f"Invalid FPS: {self.fps} (must be > 0)")

        # Validate camera positions
        if self.left_camera_position.shape != (3,):
            errors.append(
                f"left_camera_position must have shape (3,), got {self.left_camera_position.shape}"
            )

        if self.right_camera_position.shape != (3,):
            errors.append(
                f"right_camera_position must have shape (3,), got {self.right_camera_position.shape}"
            )

        # Validate baseline
        if self.baseline_distance <= 0:
            errors.append(
                f"Invalid baseline distance: {self.baseline_distance} (must be > 0)"
            )

        # Validate angles
        if not -180 <= self.left_camera_angle <= 180:
            errors.append(
                f"Invalid left camera angle: {self.left_camera_angle} (must be -180 to 180)"
            )

        if not -180 <= self.right_camera_angle <= 180:
            errors.append(
                f"Invalid right camera angle: {self.right_camera_angle} (must be -180 to 180)"
            )

        return errors

    def get_width(self) -> int:
        """Get camera resolution width."""
        return self.resolution[0]

    def get_height(self) -> int:
        """Get camera resolution height."""
        return self.resolution[1]

    def get_aspect_ratio(self) -> float:
        """Get camera aspect ratio (width/height)."""
        return self.resolution[0] / self.resolution[1]

    def get_baseline_vector(self) -> np.ndarray:
        """
        Get baseline vector between cameras.

        Returns:
            Vector from left camera to right camera (3,) array
        """
        return self.right_camera_position - self.left_camera_position

    def get_calculated_baseline(self) -> float:
        """
        Calculate actual baseline distance from camera positions.

        Returns:
            Euclidean distance between cameras in cm
        """
        return float(np.linalg.norm(self.get_baseline_vector()))

    def check_baseline_consistency(self, tolerance: float = 0.1) -> bool:
        """
        Check if specified baseline matches calculated baseline.

        Args:
            tolerance: Maximum allowed difference in cm

        Returns:
            True if baseline is consistent, False otherwise
        """
        calculated = self.get_calculated_baseline()
        return abs(calculated - self.baseline_distance) <= tolerance

    def get_camera_center(self) -> np.ndarray:
        """
        Get center point between the two cameras.

        Returns:
            Center position (3,) array in cm
        """
        return (self.left_camera_position + self.right_camera_position) / 2.0

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of camera config
        """
        return {
            "left_camera_index": self.left_camera_index,
            "right_camera_index": self.right_camera_index,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "exposure": self.exposure,
            "left_camera_position": self.left_camera_position.tolist(),
            "right_camera_position": self.right_camera_position.tolist(),
            "baseline_distance": self.baseline_distance,
            "left_camera_angle": self.left_camera_angle,
            "right_camera_angle": self.right_camera_angle,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        """
        Create CameraConfig from dictionary.

        Args:
            data: Dictionary with camera configuration

        Returns:
            CameraConfig instance
        """
        # Convert lists to numpy arrays for positions
        if "left_camera_position" in data and isinstance(
            data["left_camera_position"], list
        ):
            data["left_camera_position"] = np.array(data["left_camera_position"])

        if "right_camera_position" in data and isinstance(
            data["right_camera_position"], list
        ):
            data["right_camera_position"] = np.array(data["right_camera_position"])

        # Convert resolution to tuple
        if "resolution" in data and isinstance(data["resolution"], list):
            data["resolution"] = tuple(data["resolution"])

        return cls(**data)
