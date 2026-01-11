"""
Calibration data structures for ChunIVision.

Stores and manages stereo camera calibration parameters including
intrinsic matrices, extrinsic parameters, and transformation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class CalibrationData:
    """
    Complete calibration data for the stereo vision system.

    Contains all parameters necessary for stereo reconstruction:
    - Camera intrinsic matrices (focal length, principal point)
    - Camera extrinsic matrices (rotation, translation)
    - Stereo rectification parameters
    - Disparity-to-depth mapping (Q matrix)

    Attributes:
        version: Calibration format version for compatibility
        timestamp: When calibration was performed
        camera_left_matrix: 3x3 intrinsic matrix for left camera
        camera_right_matrix: 3x3 intrinsic matrix for right camera
        dist_coeffs_left: Distortion coefficients for left camera
        dist_coeffs_right: Distortion coefficients for right camera
        rotation_matrix: 3x3 rotation matrix from left to right camera
        translation_vector: 3x1 translation vector from left to right camera
        rectify_left: 3x3 rectification transform for left camera
        rectify_right: 3x3 rectification transform for right camera
        projection_left: 3x4 projection matrix for left camera (after rectification)
        projection_right: 3x4 projection matrix for right camera (after rectification)
        disparity_to_depth: 4x4 Q matrix for disparity-to-depth mapping
        stereo_baseline: Distance between camera centers in cm
        image_size: Image dimensions as (width, height)
        zone_boundaries: Physical boundaries of 32 touch zones
        height_thresholds: Z-coordinate thresholds for 6 air sensor levels in cm
        reference_board_size: Size of calibration board (width, height) in cm
        camera_left_transform: Perspective transform for left camera (image to world)
        camera_right_transform: Perspective transform for right camera (image to world)
    """

    version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)

    # Camera intrinsic matrices (3x3)
    camera_left_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    camera_right_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    # Distortion coefficients (5,) or (8,)
    dist_coeffs_left: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )
    dist_coeffs_right: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    # Extrinsic parameters (rotation and translation between cameras)
    rotation_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    translation_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    # Stereo rectification parameters
    rectify_left: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    rectify_right: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    # Projection matrices (3x4) after rectification
    projection_left: np.ndarray = field(
        default_factory=lambda: np.hstack([np.eye(3), np.zeros((3, 1))])
    )
    projection_right: np.ndarray = field(
        default_factory=lambda: np.hstack([np.eye(3), np.zeros((3, 1))])
    )

    # Disparity-to-depth matrix (4x4)
    disparity_to_depth: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    # Physical parameters
    stereo_baseline: float = 20.0  # cm
    image_size: Tuple[int, int] = (640, 480)  # width, height

    # Zone and height configuration
    zone_boundaries: np.ndarray = field(
        default_factory=lambda: np.zeros((32, 4, 2), dtype=np.float64)
    )
    height_thresholds: np.ndarray = field(
        default_factory=lambda: np.array(
            [17.9, 21.3, 24.7, 28.1, 31.5, 34.9], dtype=np.float64
        )
    )

    # Reference calibration board
    reference_board_size: Tuple[float, float] = (44.0, 9.0)  # cm (16*2.75, 2*4.5)

    # Perspective transforms for each camera (image to world 2D)
    camera_left_transform: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    camera_right_transform: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        """Ensure all arrays are numpy arrays with correct dtype."""
        array_fields = [
            "camera_left_matrix",
            "camera_right_matrix",
            "dist_coeffs_left",
            "dist_coeffs_right",
            "rotation_matrix",
            "translation_vector",
            "rectify_left",
            "rectify_right",
            "projection_left",
            "projection_right",
            "disparity_to_depth",
            "zone_boundaries",
            "height_thresholds",
            "camera_left_transform",
            "camera_right_transform",
        ]

        for field_name in array_fields:
            value = getattr(self, field_name)
            if not isinstance(value, np.ndarray):
                if isinstance(value, (list, tuple)):
                    setattr(self, field_name, np.array(value, dtype=np.float64))
                else:
                    raise TypeError(f"{field_name} must be array-like")

    def validate(self) -> List[str]:
        """
        Validate calibration data integrity.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check camera matrices are 3x3
        if self.camera_left_matrix.shape != (3, 3):
            errors.append(
                f"camera_left_matrix must be 3x3, got {self.camera_left_matrix.shape}"
            )
        if self.camera_right_matrix.shape != (3, 3):
            errors.append(
                f"camera_right_matrix must be 3x3, got {self.camera_right_matrix.shape}"
            )

        # Check rotation matrix is 3x3
        if self.rotation_matrix.shape != (3, 3):
            errors.append(
                f"rotation_matrix must be 3x3, got {self.rotation_matrix.shape}"
            )

        # Check rotation matrix is orthonormal
        if self.rotation_matrix.shape == (3, 3):
            det = np.linalg.det(self.rotation_matrix)
            if not np.isclose(det, 1.0, atol=1e-6):
                errors.append(
                    f"rotation_matrix determinant should be 1.0, got {det:.6f}"
                )

        # Check translation vector
        if self.translation_vector.shape not in [(3,), (3, 1)]:
            errors.append(
                f"translation_vector must be (3,) or (3,1), got {self.translation_vector.shape}"
            )

        # Check Q matrix is 4x4
        if self.disparity_to_depth.shape != (4, 4):
            errors.append(
                f"disparity_to_depth must be 4x4, got {self.disparity_to_depth.shape}"
            )

        # Check height thresholds
        if len(self.height_thresholds) != 6:
            errors.append(
                f"height_thresholds must have 6 values, got {len(self.height_thresholds)}"
            )
        if len(self.height_thresholds) == 6:
            if not all(
                self.height_thresholds[i] < self.height_thresholds[i + 1]
                for i in range(5)
            ):
                errors.append("height_thresholds must be in ascending order")

        # Check zone boundaries
        if self.zone_boundaries.shape != (32, 4, 2):
            errors.append(
                f"zone_boundaries must be (32, 4, 2), got {self.zone_boundaries.shape}"
            )

        # Check baseline
        if self.stereo_baseline <= 0:
            errors.append(
                f"stereo_baseline must be positive, got {self.stereo_baseline}"
            )

        return errors

    def is_valid(self) -> bool:
        """
        Check if calibration data is valid.

        Returns:
            True if valid, False otherwise
        """
        return len(self.validate()) == 0

    def save(self, path: str) -> None:
        """
        Save calibration data to YAML file.

        Args:
            path: File path to save to
        """
        data = {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "stereo_baseline": float(self.stereo_baseline),
            "image_size": list(self.image_size),
            "reference_board_size": list(self.reference_board_size),
            "camera_left_matrix": self.camera_left_matrix.tolist(),
            "camera_right_matrix": self.camera_right_matrix.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.flatten().tolist(),
            "rectify_left": self.rectify_left.tolist(),
            "rectify_right": self.rectify_right.tolist(),
            "projection_left": self.projection_left.tolist(),
            "projection_right": self.projection_right.tolist(),
            "disparity_to_depth": self.disparity_to_depth.tolist(),
            "zone_boundaries": self.zone_boundaries.tolist(),
            "height_thresholds": self.height_thresholds.tolist(),
            "camera_left_transform": self.camera_left_transform.tolist(),
            "camera_right_transform": self.camera_right_transform.tolist(),
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "CalibrationData":
        """
        Load calibration data from YAML file.

        Args:
            path: File path to load from

        Returns:
            CalibrationData instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path_obj, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid calibration file format: {path}")

        # Parse timestamp
        timestamp = datetime.fromisoformat(
            data.get("timestamp", datetime.now().isoformat())
        )

        return cls(
            version=data.get("version", "1.0"),
            timestamp=timestamp,
            stereo_baseline=data.get("stereo_baseline", 20.0),
            image_size=tuple(data.get("image_size", [640, 480])),
            reference_board_size=tuple(data.get("reference_board_size", [44.0, 9.0])),
            camera_left_matrix=np.array(data.get("camera_left_matrix", np.eye(3))),
            camera_right_matrix=np.array(data.get("camera_right_matrix", np.eye(3))),
            dist_coeffs_left=np.array(data.get("dist_coeffs_left", np.zeros(5))),
            dist_coeffs_right=np.array(data.get("dist_coeffs_right", np.zeros(5))),
            rotation_matrix=np.array(data.get("rotation_matrix", np.eye(3))),
            translation_vector=np.array(data.get("translation_vector", np.zeros(3))),
            rectify_left=np.array(data.get("rectify_left", np.eye(3))),
            rectify_right=np.array(data.get("rectify_right", np.eye(3))),
            projection_left=np.array(
                data.get("projection_left", np.hstack([np.eye(3), np.zeros((3, 1))]))
            ),
            projection_right=np.array(
                data.get("projection_right", np.hstack([np.eye(3), np.zeros((3, 1))]))
            ),
            disparity_to_depth=np.array(data.get("disparity_to_depth", np.eye(4))),
            zone_boundaries=np.array(data.get("zone_boundaries", np.zeros((32, 4, 2)))),
            height_thresholds=np.array(
                data.get("height_thresholds", [17.9, 21.3, 24.7, 28.1, 31.5, 34.9])
            ),
            camera_left_transform=np.array(
                data.get("camera_left_transform", np.eye(3))
            ),
            camera_right_transform=np.array(
                data.get("camera_right_transform", np.eye(3))
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert calibration data to dictionary.

        Returns:
            Dictionary representation of calibration data
        """
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "stereo_baseline": self.stereo_baseline,
            "image_size": self.image_size,
            "reference_board_size": self.reference_board_size,
            "camera_left_matrix": self.camera_left_matrix,
            "camera_right_matrix": self.camera_right_matrix,
            "dist_coeffs_left": self.dist_coeffs_left,
            "dist_coeffs_right": self.dist_coeffs_right,
            "rotation_matrix": self.rotation_matrix,
            "translation_vector": self.translation_vector,
            "rectify_left": self.rectify_left,
            "rectify_right": self.rectify_right,
            "projection_left": self.projection_left,
            "projection_right": self.projection_right,
            "disparity_to_depth": self.disparity_to_depth,
            "zone_boundaries": self.zone_boundaries,
            "height_thresholds": self.height_thresholds,
            "camera_left_transform": self.camera_left_transform,
            "camera_right_transform": self.camera_right_transform,
        }

    @classmethod
    def create_default(
        cls, image_size: Tuple[int, int] = (640, 480)
    ) -> "CalibrationData":
        """
        Create default calibration data with reasonable estimates.

        This is useful for initial testing or when no calibration is available.
        The values assume typical webcam-like parameters.

        Args:
            image_size: Image dimensions (width, height)

        Returns:
            CalibrationData with default values
        """
        width, height = image_size
        fx = fy = width  # Approximate focal length in pixels
        cx, cy = width / 2, height / 2  # Principal point at image center

        camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

        # Baseline of 20cm (typical for stereo setup)
        baseline = 20.0
        translation = np.array([-baseline, 0, 0], dtype=np.float64)

        # Compute Q matrix for disparity-to-depth
        # Q = [[1, 0, 0, -cx],
        #      [0, 1, 0, -cy],
        #      [0, 0, 0, fx],
        #      [0, 0, -1/Tx, (cx - cx')/Tx]]
        # For rectified stereo with same camera matrices: cx' = cx
        Q = np.array(
            [
                [1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, fx],
                [0, 0, -1.0 / baseline, 0],
            ],
            dtype=np.float64,
        )

        return cls(
            image_size=image_size,
            camera_left_matrix=camera_matrix.copy(),
            camera_right_matrix=camera_matrix.copy(),
            translation_vector=translation,
            disparity_to_depth=Q,
            stereo_baseline=baseline,
        )
