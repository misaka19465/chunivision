"""
3D Point Cloud data structure for ChunIVision.

Represents 3D point clouds from stereo vision processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class PointCloud3D:
    """
    3D point cloud representation.

    Stores 3D coordinates along with optional color/intensity data
    and metadata from stereo processing.

    Attributes:
        points: Nx3 array of 3D coordinates (x, y, z) in cm
        intensities: Optional Nx1 array of intensity values [0, 255]
        colors: Optional Nx3 array of RGB color values [0, 255]
        mask: Optional HxW boolean mask indicating valid points in original image
        timestamp: Capture timestamp in seconds since epoch
        frame_id: Frame sequence number
    """

    points: np.ndarray  # Shape (N, 3), dtype=float64
    intensities: Optional[np.ndarray] = None  # Shape (N,), dtype=uint8
    colors: Optional[np.ndarray] = None  # Shape (N, 3), dtype=uint8
    mask: Optional[np.ndarray] = None  # Shape (H, W), dtype=bool
    timestamp: float = 0.0
    frame_id: int = 0

    def __post_init__(self) -> None:
        """Validate point cloud data."""
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points, dtype=np.float64)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            if self.points.size == 0:
                self.points = np.empty((0, 3), dtype=np.float64)
            else:
                raise ValueError(
                    f"points must have shape (N, 3), got {self.points.shape}"
                )

    @property
    def num_points(self) -> int:
        """Return number of points in cloud."""
        return self.points.shape[0]

    @property
    def is_empty(self) -> bool:
        """Check if point cloud is empty."""
        return self.num_points == 0

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of point cloud.

        Returns:
            Tuple of (min_xyz, max_xyz) arrays of shape (3,)
        """
        if self.is_empty:
            return np.zeros(3), np.zeros(3)
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        """
        Get centroid (mean position) of point cloud.

        Returns:
            Array of shape (3,) with mean x, y, z
        """
        if self.is_empty:
            return np.zeros(3)
        return self.points.mean(axis=0)

    def filter_by_depth(self, min_depth: float, max_depth: float) -> "PointCloud3D":
        """
        Filter points by Z coordinate (depth).

        Args:
            min_depth: Minimum Z value in cm
            max_depth: Maximum Z value in cm

        Returns:
            New PointCloud3D with filtered points
        """
        if self.is_empty:
            return PointCloud3D(
                points=np.empty((0, 3)),
                timestamp=self.timestamp,
                frame_id=self.frame_id,
            )

        z_coords = self.points[:, 2]
        mask = (z_coords >= min_depth) & (z_coords <= max_depth)

        return PointCloud3D(
            points=self.points[mask].copy(),
            intensities=(
                self.intensities[mask].copy() if self.intensities is not None else None
            ),
            colors=self.colors[mask].copy() if self.colors is not None else None,
            mask=None,  # Mask no longer valid after filtering
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )

    def filter_by_region(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Optional[Tuple[float, float]] = None,
    ) -> "PointCloud3D":
        """
        Filter points by spatial region.

        Args:
            x_range: (min_x, max_x) in cm
            y_range: (min_y, max_y) in cm
            z_range: Optional (min_z, max_z) in cm

        Returns:
            New PointCloud3D with filtered points
        """
        if self.is_empty:
            return PointCloud3D(
                points=np.empty((0, 3)),
                timestamp=self.timestamp,
                frame_id=self.frame_id,
            )

        x_mask = (self.points[:, 0] >= x_range[0]) & (self.points[:, 0] <= x_range[1])
        y_mask = (self.points[:, 1] >= y_range[0]) & (self.points[:, 1] <= y_range[1])
        mask = x_mask & y_mask

        if z_range is not None:
            z_mask = (self.points[:, 2] >= z_range[0]) & (
                self.points[:, 2] <= z_range[1]
            )
            mask = mask & z_mask

        return PointCloud3D(
            points=self.points[mask].copy(),
            intensities=(
                self.intensities[mask].copy() if self.intensities is not None else None
            ),
            colors=self.colors[mask].copy() if self.colors is not None else None,
            mask=None,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )

    def downsample(self, voxel_size: float) -> "PointCloud3D":
        """
        Downsample point cloud using voxel grid filter.

        Args:
            voxel_size: Size of voxels in cm

        Returns:
            New downsampled PointCloud3D
        """
        if self.is_empty or voxel_size <= 0:
            return PointCloud3D(
                points=self.points.copy(),
                intensities=(
                    self.intensities.copy() if self.intensities is not None else None
                ),
                colors=self.colors.copy() if self.colors is not None else None,
                timestamp=self.timestamp,
                frame_id=self.frame_id,
            )

        # Compute voxel indices
        min_bound = self.points.min(axis=0)
        voxel_indices = ((self.points - min_bound) / voxel_size).astype(np.int32)

        # Find unique voxels and keep one point per voxel
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

        return PointCloud3D(
            points=self.points[unique_indices].copy(),
            intensities=(
                self.intensities[unique_indices].copy()
                if self.intensities is not None
                else None
            ),
            colors=(
                self.colors[unique_indices].copy() if self.colors is not None else None
            ),
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )

    def transform(
        self, rotation: np.ndarray, translation: np.ndarray
    ) -> "PointCloud3D":
        """
        Apply rigid transformation to point cloud.

        Args:
            rotation: 3x3 rotation matrix
            translation: 3-element translation vector

        Returns:
            New transformed PointCloud3D
        """
        if self.is_empty:
            return PointCloud3D(
                points=np.empty((0, 3)),
                timestamp=self.timestamp,
                frame_id=self.frame_id,
            )

        # Apply rotation and translation: p' = R @ p + t
        transformed_points = (rotation @ self.points.T).T + translation

        return PointCloud3D(
            points=transformed_points,
            intensities=(
                self.intensities.copy() if self.intensities is not None else None
            ),
            colors=self.colors.copy() if self.colors is not None else None,
            mask=self.mask.copy() if self.mask is not None else None,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )

    def merge(self, other: "PointCloud3D") -> "PointCloud3D":
        """
        Merge with another point cloud.

        Args:
            other: Another PointCloud3D to merge

        Returns:
            New merged PointCloud3D
        """
        if self.is_empty:
            return PointCloud3D(
                points=other.points.copy(),
                intensities=(
                    other.intensities.copy() if other.intensities is not None else None
                ),
                colors=other.colors.copy() if other.colors is not None else None,
                timestamp=max(self.timestamp, other.timestamp),
                frame_id=max(self.frame_id, other.frame_id),
            )
        if other.is_empty:
            return PointCloud3D(
                points=self.points.copy(),
                intensities=(
                    self.intensities.copy() if self.intensities is not None else None
                ),
                colors=self.colors.copy() if self.colors is not None else None,
                timestamp=self.timestamp,
                frame_id=self.frame_id,
            )

        merged_points = np.vstack([self.points, other.points])

        # Handle intensities
        merged_intensities = None
        if self.intensities is not None and other.intensities is not None:
            merged_intensities = np.concatenate([self.intensities, other.intensities])

        # Handle colors
        merged_colors = None
        if self.colors is not None and other.colors is not None:
            merged_colors = np.vstack([self.colors, other.colors])

        return PointCloud3D(
            points=merged_points,
            intensities=merged_intensities,
            colors=merged_colors,
            timestamp=max(self.timestamp, other.timestamp),
            frame_id=max(self.frame_id, other.frame_id),
        )

    def get_statistics(self) -> dict:
        """
        Get statistical summary of point cloud.

        Returns:
            Dictionary with statistics
        """
        if self.is_empty:
            return {
                "num_points": 0,
                "bounds_min": [0, 0, 0],
                "bounds_max": [0, 0, 0],
                "centroid": [0, 0, 0],
            }

        min_bounds, max_bounds = self.bounds
        return {
            "num_points": self.num_points,
            "bounds_min": min_bounds.tolist(),
            "bounds_max": max_bounds.tolist(),
            "centroid": self.centroid.tolist(),
            "std": self.points.std(axis=0).tolist(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PointCloud3D(num_points={self.num_points}, "
            f"frame_id={self.frame_id}, timestamp={self.timestamp:.3f})"
        )
