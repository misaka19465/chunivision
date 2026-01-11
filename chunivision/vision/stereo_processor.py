"""
Stereo Processing module for ChunIVision.

Processes stereo image pairs to generate depth maps and 3D point clouds
using semi-global block matching (SGBM) and other stereo algorithms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..calibration.calibration_data import CalibrationData
from ..utils.logger import Logger
from .point_cloud import PointCloud3D

logger = Logger.get_logger(__name__)


class StereoAlgorithm(Enum):
    """Available stereo matching algorithms."""

    BM = "BM"  # Block Matching (faster, lower quality)
    SGBM = "SGBM"  # Semi-Global Block Matching (better quality)


@dataclass
class StereoConfig:
    """
    Configuration for stereo processing.

    Attributes:
        algorithm: Stereo matching algorithm to use
        num_disparities: Number of disparities to search (must be divisible by 16)
        block_size: Block size for matching (odd number >= 5)
        min_depth: Minimum depth to consider in cm
        max_depth: Maximum depth to consider in cm
        speckle_window_size: Size of disparity checking window
        speckle_range: Maximum disparity variation in window
        uniqueness_ratio: Margin in % by which best match must beat second best
        disp12_max_diff: Maximum allowed difference in left-right consistency check
        pre_filter_cap: Truncation value for prefiltered image pixels
        p1: First parameter controlling disparity smoothness (SGBM only)
        p2: Second parameter controlling disparity smoothness (SGBM only)
    """

    algorithm: StereoAlgorithm = StereoAlgorithm.SGBM
    num_disparities: int = 128  # Must be divisible by 16
    block_size: int = 5  # Odd number, >= 3 for SGBM, >= 5 for BM
    min_depth: float = 0.0  # cm
    max_depth: float = 35.0  # cm
    speckle_window_size: int = 100
    speckle_range: int = 2
    uniqueness_ratio: int = 10
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    p1: int = 0  # 0 = auto-compute based on block_size
    p2: int = 0  # 0 = auto-compute based on block_size

    def validate(self) -> list[str]:
        """
        Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            errors.append(
                f"num_disparities must be positive and divisible by 16, got {self.num_disparities}"
            )

        min_block = 5 if self.algorithm == StereoAlgorithm.BM else 3
        if self.block_size < min_block or self.block_size % 2 == 0:
            errors.append(
                f"block_size must be odd and >= {min_block}, got {self.block_size}"
            )

        if self.min_depth < 0:
            errors.append(f"min_depth must be non-negative, got {self.min_depth}")

        if self.max_depth <= self.min_depth:
            errors.append(
                f"max_depth ({self.max_depth}) must be greater than min_depth ({self.min_depth})"
            )

        return errors


class StereoProcessor:
    """
    Processes stereo image pairs to generate depth maps and 3D point clouds.

    Uses calibration data to perform stereo matching and 3D reconstruction.
    The processor handles:
    - Image rectification using calibration parameters
    - Stereo matching using SGBM or BM algorithm
    - Disparity to depth conversion
    - 3D point cloud generation

    Usage:
        calibration = CalibrationData.load("calibration.yaml")
        processor = StereoProcessor(calibration)

        for left_frame, right_frame in frame_pairs:
            point_cloud = processor.process(left_frame, right_frame)
            depth_map = processor.get_depth_map(left_frame, right_frame)
    """

    def __init__(
        self,
        calibration_data: CalibrationData,
        config: Optional[StereoConfig] = None,
    ):
        """
        Initialize processor with calibration data.

        Args:
            calibration_data: Calibration matrices and parameters
            config: Optional stereo processing configuration
        """
        self.calibration_data = calibration_data
        self.config = config or StereoConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid stereo config: {', '.join(errors)}")

        # Create stereo matcher
        self._stereo_matcher = self._create_stereo_matcher()

        # Pre-compute rectification maps for efficiency
        self._map_left_x: Optional[np.ndarray] = None
        self._map_left_y: Optional[np.ndarray] = None
        self._map_right_x: Optional[np.ndarray] = None
        self._map_right_y: Optional[np.ndarray] = None
        self._compute_rectification_maps()

        # Frame counter for point cloud metadata
        self._frame_count = 0

        # Performance tracking
        self._last_processing_time_ms = 0.0

        logger.info(
            f"StereoProcessor initialized: algorithm={self.config.algorithm.value}, "
            f"num_disparities={self.config.num_disparities}, "
            f"depth_range=[{self.config.min_depth}, {self.config.max_depth}]cm"
        )

    def _create_stereo_matcher(self) -> cv2.StereoSGBM | cv2.StereoBM:
        """Create stereo matching algorithm based on configuration."""
        cfg = self.config

        if cfg.algorithm == StereoAlgorithm.BM:
            matcher = cv2.StereoBM_create(
                numDisparities=cfg.num_disparities,
                blockSize=cfg.block_size,
            )
            matcher.setSpeckleWindowSize(cfg.speckle_window_size)
            matcher.setSpeckleRange(cfg.speckle_range)
            matcher.setUniquenessRatio(cfg.uniqueness_ratio)
            matcher.setDisp12MaxDiff(cfg.disp12_max_diff)
            matcher.setPreFilterCap(cfg.pre_filter_cap)
            logger.debug("Created StereoBM matcher")
            return matcher

        # SGBM (default)
        # Compute P1 and P2 if not specified
        p1 = cfg.p1 if cfg.p1 > 0 else 8 * 1 * cfg.block_size**2
        p2 = cfg.p2 if cfg.p2 > 0 else 32 * 1 * cfg.block_size**2

        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=cfg.num_disparities,
            blockSize=cfg.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=cfg.disp12_max_diff,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window_size,
            speckleRange=cfg.speckle_range,
            preFilterCap=cfg.pre_filter_cap,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        logger.debug("Created StereoSGBM matcher")
        return matcher

    def _compute_rectification_maps(self) -> None:
        """Pre-compute rectification maps from calibration data."""
        cal = self.calibration_data
        h, w = cal.image_size[1], cal.image_size[0]

        # Compute rectification maps for left camera
        self._map_left_x, self._map_left_y = cv2.initUndistortRectifyMap(
            cal.camera_left_matrix,
            cal.dist_coeffs_left,
            cal.rectify_left,
            cal.projection_left,
            (w, h),
            cv2.CV_32FC1,
        )

        # Compute rectification maps for right camera
        self._map_right_x, self._map_right_y = cv2.initUndistortRectifyMap(
            cal.camera_right_matrix,
            cal.dist_coeffs_right,
            cal.rectify_right,
            cal.projection_right,
            (w, h),
            cv2.CV_32FC1,
        )

        logger.debug(f"Computed rectification maps for image size {w}x{h}")

    def rectify_images(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stereo rectification to image pair.

        Rectification aligns the epipolar lines horizontally,
        which is required for stereo matching.

        Args:
            left_frame: Left camera frame (H, W) grayscale
            right_frame: Right camera frame (H, W) grayscale

        Returns:
            Tuple of (rectified_left, rectified_right) images
        """
        if (
            self._map_left_x is None
            or self._map_left_y is None
            or self._map_right_x is None
            or self._map_right_y is None
        ):
            self._compute_rectification_maps()

        left_rectified = cv2.remap(
            left_frame,
            self._map_left_x,
            self._map_left_y,
            cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right_frame,
            self._map_right_x,
            self._map_right_y,
            cv2.INTER_LINEAR,
        )

        return left_rectified, right_rectified

    def compute_disparity(
        self, left_frame: np.ndarray, right_frame: np.ndarray, rectify: bool = True
    ) -> np.ndarray:
        """
        Compute disparity map from stereo pair.

        Args:
            left_frame: Left camera frame (H, W) grayscale
            right_frame: Right camera frame (H, W) grayscale
            rectify: Whether to apply rectification first

        Returns:
            Disparity map as float32 array (disparity in pixels)
        """
        # Ensure grayscale
        if len(left_frame.shape) == 3:
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        if len(right_frame.shape) == 3:
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Rectify if requested
        if rectify:
            left_frame, right_frame = self.rectify_images(left_frame, right_frame)

        # Compute disparity (returned as 16-bit signed, with 4 fractional bits)
        disparity_raw = self._stereo_matcher.compute(left_frame, right_frame)

        # Convert to float and scale (divide by 16 to get actual disparity)
        disparity = disparity_raw.astype(np.float32) / 16.0

        # Set invalid disparities to NaN
        disparity[disparity <= 0] = np.nan

        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.

        Uses the Q matrix from stereo calibration.

        Args:
            disparity: Disparity map in pixels

        Returns:
            Depth map in cm
        """
        # The Q matrix converts (u, v, disparity) to (X, Y, Z)
        # Z = focal_length * baseline / disparity
        # From Q matrix: Q[2,3] = focal_length, Q[3,2] = -1/baseline
        Q = self.calibration_data.disparity_to_depth

        # Extract focal length and baseline from Q matrix
        # Q[2,3] = f, Q[3,2] = -1/Tx (where Tx is baseline)
        focal_length = Q[2, 3]
        baseline = (
            -1.0 / Q[3, 2] if Q[3, 2] != 0 else self.calibration_data.stereo_baseline
        )

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (focal_length * baseline) / disparity

        # Apply depth range filter
        depth[depth < self.config.min_depth] = np.nan
        depth[depth > self.config.max_depth] = np.nan

        return depth

    def get_depth_map(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> np.ndarray:
        """
        Compute depth map from stereo pair.

        Args:
            left_frame: Left camera frame (H, W) numpy array
            right_frame: Right camera frame (H, W) numpy array

        Returns:
            Depth map as (H, W) numpy array with depth in cm
        """
        start_time = time.perf_counter()

        disparity = self.compute_disparity(left_frame, right_frame)
        depth = self.disparity_to_depth(disparity)

        self._last_processing_time_ms = (time.perf_counter() - start_time) * 1000

        return depth

    def process(self, left_frame: np.ndarray, right_frame: np.ndarray) -> PointCloud3D:
        """
        Generate 3D point cloud from stereo image pair.

        This is the main processing method that combines rectification,
        stereo matching, and 3D reconstruction.

        Args:
            left_frame: Left camera frame (H, W) numpy array
            right_frame: Right camera frame (H, W) numpy array

        Returns:
            PointCloud3D object with 3D coordinates
        """
        start_time = time.perf_counter()
        timestamp = time.time()

        # Ensure grayscale
        if len(left_frame.shape) == 3:
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_frame

        if len(right_frame.shape) == 3:
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_frame

        # Rectify images
        left_rect, right_rect = self.rectify_images(left_gray, right_gray)

        # Compute disparity
        disparity = self.compute_disparity(left_rect, right_rect, rectify=False)

        # Reproject to 3D using Q matrix
        points_3d = cv2.reprojectImageTo3D(
            disparity.astype(np.float32),
            self.calibration_data.disparity_to_depth,
            handleMissingValues=True,
        )

        # Create valid point mask
        valid_mask = ~np.isnan(disparity) & ~np.isinf(points_3d[:, :, 2])

        # Apply depth range filter
        z_coords = points_3d[:, :, 2]
        valid_mask &= (z_coords >= self.config.min_depth) & (
            z_coords <= self.config.max_depth
        )

        # Extract valid 3D points
        valid_points = points_3d[valid_mask]

        # Extract intensities from rectified left image
        intensities = left_rect[valid_mask]

        self._frame_count += 1
        self._last_processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Frame {self._frame_count}: {len(valid_points)} valid points, "
            f"{self._last_processing_time_ms:.1f}ms"
        )

        return PointCloud3D(
            points=valid_points,
            intensities=intensities,
            mask=valid_mask,
            timestamp=timestamp,
            frame_id=self._frame_count,
        )

    def update_calibration(self, calibration_data: CalibrationData) -> None:
        """
        Update calibration parameters.

        This will recompute rectification maps.

        Args:
            calibration_data: New calibration data
        """
        self.calibration_data = calibration_data
        self._compute_rectification_maps()
        logger.info("Updated calibration data and rectification maps")

    def set_depth_range(self, min_depth: float, max_depth: float) -> None:
        """
        Set valid depth range for filtering.

        Args:
            min_depth: Minimum depth in cm (e.g., 0 for touch surface)
            max_depth: Maximum depth in cm (e.g., 35 for max height)
        """
        if max_depth <= min_depth:
            raise ValueError(
                f"max_depth ({max_depth}) must be greater than min_depth ({min_depth})"
            )

        self.config.min_depth = min_depth
        self.config.max_depth = max_depth
        logger.info(f"Depth range set to [{min_depth}, {max_depth}] cm")

    def set_algorithm(self, algorithm: StereoAlgorithm) -> None:
        """
        Change stereo matching algorithm.

        Args:
            algorithm: StereoAlgorithm.BM or StereoAlgorithm.SGBM
        """
        self.config.algorithm = algorithm
        self._stereo_matcher = self._create_stereo_matcher()
        logger.info(f"Switched to {algorithm.value} algorithm")

    def set_num_disparities(self, num_disparities: int) -> None:
        """
        Set number of disparities to search.

        Args:
            num_disparities: Must be positive and divisible by 16
        """
        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities must be positive and divisible by 16, got {num_disparities}"
            )

        self.config.num_disparities = num_disparities
        self._stereo_matcher = self._create_stereo_matcher()
        logger.info(f"Set num_disparities to {num_disparities}")

    def set_block_size(self, block_size: int) -> None:
        """
        Set block size for stereo matching.

        Args:
            block_size: Odd number >= 5 for BM, >= 3 for SGBM
        """
        min_size = 5 if self.config.algorithm == StereoAlgorithm.BM else 3
        if block_size < min_size or block_size % 2 == 0:
            raise ValueError(
                f"block_size must be odd and >= {min_size}, got {block_size}"
            )

        self.config.block_size = block_size
        self._stereo_matcher = self._create_stereo_matcher()
        logger.info(f"Set block_size to {block_size}")

    def get_processing_time_ms(self) -> float:
        """Return processing time of last frame in milliseconds."""
        return self._last_processing_time_ms

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "algorithm": self.config.algorithm.value,
            "num_disparities": self.config.num_disparities,
            "block_size": self.config.block_size,
            "depth_range": [self.config.min_depth, self.config.max_depth],
            "frame_count": self._frame_count,
            "last_processing_time_ms": self._last_processing_time_ms,
        }

    def visualize_disparity(
        self, disparity: np.ndarray, colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create visualization of disparity map.

        Args:
            disparity: Disparity map from compute_disparity
            colormap: OpenCV colormap to use

        Returns:
            Color-mapped disparity visualization (H, W, 3)
        """
        # Normalize to 0-255 range
        valid_mask = ~np.isnan(disparity) & (disparity > 0)
        if not np.any(valid_mask):
            h, w = disparity.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        disp_normalized = np.zeros_like(disparity)
        disp_normalized[valid_mask] = disparity[valid_mask]

        min_val = np.min(disp_normalized[valid_mask])
        max_val = np.max(disp_normalized[valid_mask])

        if max_val > min_val:
            disp_normalized = (
                (disp_normalized - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)
        else:
            disp_normalized = np.zeros_like(disparity, dtype=np.uint8)

        # Apply colormap
        return cv2.applyColorMap(disp_normalized, colormap)

    def visualize_depth(
        self, depth: np.ndarray, colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create visualization of depth map.

        Args:
            depth: Depth map from get_depth_map
            colormap: OpenCV colormap to use

        Returns:
            Color-mapped depth visualization (H, W, 3)
        """
        # Normalize to 0-255 range based on configured depth range
        valid_mask = ~np.isnan(depth)
        if not np.any(valid_mask):
            h, w = depth.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        depth_normalized = np.zeros_like(depth)
        depth_normalized[valid_mask] = np.clip(
            depth[valid_mask], self.config.min_depth, self.config.max_depth
        )

        # Scale to 0-255 (inverted so closer = brighter)
        depth_range = self.config.max_depth - self.config.min_depth
        if depth_range > 0:
            depth_normalized = (
                (self.config.max_depth - depth_normalized) / depth_range * 255
            ).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)

        depth_normalized[~valid_mask] = 0

        # Apply colormap
        return cv2.applyColorMap(depth_normalized, colormap)
