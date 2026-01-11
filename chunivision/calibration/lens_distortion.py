"""
Lens distortion correction for Oculus Rift CV1 cameras.

This module provides lens distortion correction using fixed calibration
parameters from the camera's factory calibration. It handles conversion
between distorted (raw camera) and undistorted (corrected) pixel coordinates.

Note: This module contains the distortion parameters directly to avoid
circular imports with the viewer module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LensDistortionParams:
    """
    Lens distortion calibration parameters for Oculus Rift CV1 camera.

    These are factory-calibrated parameters for the wide-angle lens.

    Attributes:
        frame_size: Camera frame size (width, height)
        center: Optical center (cx, cy)
        kappas: Radial distortion coefficients (k0, k1, k2)
        rhos: Tangential distortion coefficients (p1, p2)
        max_r2: Maximum squared radius for valid undistortion
    """

    frame_size: Tuple[int, int] = (1280, 960)
    center: Tuple[float, float] = (655.052, 475.083)
    kappas: Tuple[float, float, float] = (5.16403e-07, 2.44492e-13, 6.881e-19)
    rhos: Tuple[float, float] = (-8.66716e-07, 8.37108e-07)
    max_r2: float = 0.0

    @classmethod
    def get_default(cls) -> "LensDistortionParams":
        """Get default Oculus Rift CV1 distortion parameters."""
        return cls()

    @classmethod
    def from_camera_params(cls, cal_params: dict) -> "LensDistortionParams":
        """
        Create distortion parameters from camera calibration dict.

        Args:
            cal_params: Dictionary with keys 'frame_size', 'cx', 'cy', 'k', 'max_r2'

        Returns:
            LensDistortionParams instance
        """
        return cls(
            frame_size=cal_params.get("frame_size", (1280, 960)),
            center=(cal_params.get("cx", 655.052), cal_params.get("cy", 475.083)),
            kappas=tuple(
                cal_params.get("k", [5.16403e-07, 2.44492e-13, 6.881e-19])[:3]
            ),
            rhos=tuple(cal_params.get("rhos", [-8.66716e-07, 8.37108e-07])[:2]),
            max_r2=cal_params.get("max_r2", 0.0),
        )

    def to_opencv_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to OpenCV camera matrix and distortion coefficients.

        Returns:
            Tuple of (camera_matrix, dist_coeffs)
            - camera_matrix: 3x3 intrinsic matrix
            - dist_coeffs: distortion coefficients (k1, k2, p1, p2, k3)
        """
        # Approximate focal length from frame size (typical for wide-angle)
        fx = fy = self.frame_size[0] * 0.8
        cx, cy = self.center

        camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

        # OpenCV distortion: (k1, k2, p1, p2, k3, ...)
        # Our kappas are for polynomial: r' = r * (1 + k0*r^2 + k1*r^4 + k2*r^6)
        # OpenCV uses: r' = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)
        dist_coeffs = np.array(
            [
                self.kappas[0],  # k1
                self.kappas[1],  # k2
                self.rhos[0],  # p1
                self.rhos[1],  # p2
                self.kappas[2],  # k3
            ],
            dtype=np.float64,
        )

        return camera_matrix, dist_coeffs


class LensDistortion:
    """
    Lens distortion correction handler for Oculus Rift CV1 cameras.

    This class handles conversion between distorted (raw camera) and undistorted
    (corrected) pixel coordinates using the camera's calibration parameters.

    Features:
        - Undistort individual points
        - Undistort entire frames using precomputed remap tables
        - Distort points (inverse operation for certain workflows)

    Example:
        # Create with default parameters
        distortion = LensDistortion()

        # Undistort a point
        undistorted_pt = distortion.undistort_point((320, 240))

        # Undistort entire frame
        undistorted_frame = distortion.undistort_frame(frame)
    """

    def __init__(self, params: Optional[LensDistortionParams] = None):
        """
        Initialize lens distortion correction.

        Args:
            params: Distortion parameters. If None, uses default Oculus CV1 params.
        """
        self.params = params or LensDistortionParams.get_default()

        # Precomputed remap tables (lazy initialization)
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._maps_computed = False

    @property
    def center(self) -> Tuple[float, float]:
        """Get optical center."""
        return self.params.center

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame size."""
        return self.params.frame_size

    def can_undistort(self, pixel: Tuple[float, float]) -> bool:
        """
        Check if a pixel can be undistorted.

        Args:
            pixel: (x, y) coordinate tuple

        Returns:
            True if pixel is within valid undistortion range
        """
        if pixel is None or len(pixel) != 2:
            return False
        dx = pixel[0] - self.params.center[0]
        dy = pixel[1] - self.params.center[1]
        if self.params.max_r2 <= 0:
            return True
        return (dx * dx + dy * dy) < self.params.max_r2

    def undistort_point(self, pixel: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert a distorted pixel coordinate to undistorted coordinate.

        Args:
            pixel: (x, y) coordinate in distorted (raw camera) space

        Returns:
            (x, y) coordinate in undistorted (corrected) space

        Raises:
            ValueError: If pixel is invalid
        """
        if pixel is None or len(pixel) != 2:
            raise ValueError("Pixel must be a tuple of (x, y) coordinates")

        cx, cy = self.params.center
        dx = pixel[0] - cx
        dy = pixel[1] - cy
        r2 = dx * dx + dy * dy

        # Compute radial distortion
        radial = 0.0
        for kappa in reversed(self.params.kappas):
            radial = (radial + kappa) * r2
        radial += 1.0

        # Apply radial and tangential distortion
        rho1, rho2 = self.params.rhos
        return (
            cx + dx * radial + 2.0 * rho1 * dx * dy + rho2 * (r2 + 2.0 * dx * dx),
            cy + dy * radial + rho1 * (r2 + 2.0 * dy * dy) + 2.0 * rho2 * dx * dy,
        )

    def distort_point(
        self,
        pixel: Tuple[float, float],
        max_iterations: int = 20,
        tolerance: float = 1e-6,
    ) -> Tuple[float, float]:
        """
        Convert an undistorted pixel coordinate to distorted coordinate.

        This is the inverse of undistort_point, needed for creating remap tables.

        Args:
            pixel: (x, y) coordinate in undistorted (corrected) space
            max_iterations: Maximum Newton-Raphson iterations
            tolerance: Convergence tolerance in pixels

        Returns:
            (x, y) coordinate in distorted (raw camera) space
        """
        if pixel is None or len(pixel) != 2:
            raise ValueError("Pixel must be a tuple of (x, y) coordinates")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")

        # Use fixed-point iteration
        distorted_x, distorted_y = pixel

        for _ in range(max_iterations):
            undistorted_x, undistorted_y = self.undistort_point(
                (distorted_x, distorted_y)
            )
            error_x = undistorted_x - pixel[0]
            error_y = undistorted_y - pixel[1]

            error = math.sqrt(error_x * error_x + error_y * error_y)
            if error < tolerance:
                break

            distorted_x -= error_x
            distorted_y -= error_y

        # Clamp to valid image bounds
        w, h = self.params.frame_size
        distorted_x = max(0.0, min(float(w - 1), distorted_x))
        distorted_y = max(0.0, min(float(h - 1), distorted_y))

        return (distorted_x, distorted_y)

    def compute_remap_tables(
        self, output_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Precompute remap tables for efficient frame undistortion.

        Args:
            output_size: Output frame size. If None, uses input frame size.
        """
        if output_size is None:
            output_size = self.params.frame_size

        w, h = output_size
        self._map_x = np.zeros((h, w), dtype=np.float32)
        self._map_y = np.zeros((h, w), dtype=np.float32)

        for yy in range(h):
            for xx in range(w):
                # For each output pixel, find the input (distorted) pixel
                dx, dy = self.distort_point((float(xx), float(yy)))
                self._map_x[yy, xx] = dx
                self._map_y[yy, xx] = dy

        self._maps_computed = True

    def undistort_frame(
        self,
        frame: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        border_mode: int = cv2.BORDER_REPLICATE,
    ) -> np.ndarray:
        """
        Undistort an entire frame.

        Uses precomputed remap tables for efficiency. Tables are computed
        on first call if not already done.

        Args:
            frame: Input distorted frame
            output_size: Output frame size. If None, uses input size.
            border_mode: OpenCV border mode for edges

        Returns:
            Undistorted frame
        """
        if not self._maps_computed:
            self.compute_remap_tables(output_size or (frame.shape[1], frame.shape[0]))

        return cv2.remap(
            frame,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort multiple points.

        Args:
            points: Array of points, shape (N, 2)

        Returns:
            Array of undistorted points, shape (N, 2)
        """
        if points.ndim == 1:
            return np.array(self.undistort_point(tuple(points)))

        result = np.zeros_like(points, dtype=np.float64)
        for i, pt in enumerate(points):
            result[i] = self.undistort_point(tuple(pt))
        return result

    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Distort multiple points (inverse operation).

        Args:
            points: Array of points, shape (N, 2)

        Returns:
            Array of distorted points, shape (N, 2)
        """
        if points.ndim == 1:
            return np.array(self.distort_point(tuple(points)))

        result = np.zeros_like(points, dtype=np.float64)
        for i, pt in enumerate(points):
            result[i] = self.distort_point(tuple(pt))
        return result

    def get_opencv_undistort_maps(
        self,
        new_camera_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get OpenCV-compatible undistortion maps using cv2.initUndistortRectifyMap.

        This is an alternative method using OpenCV's built-in undistortion.

        Args:
            new_camera_matrix: Optional new camera matrix for scaling

        Returns:
            Tuple of (map_x, map_y) for use with cv2.remap
        """
        camera_matrix, dist_coeffs = self.params.to_opencv_params()
        h, w = self.params.frame_size[1], self.params.frame_size[0]

        if new_camera_matrix is None:
            new_camera_matrix = camera_matrix

        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
            (w, h),
            cv2.CV_32FC1,
        )
        return map_x, map_y
