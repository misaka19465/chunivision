"""
Stereo camera calibration system for ChunIVision.

Provides tools for calibrating stereo camera setup, including
intrinsic and extrinsic parameter estimation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration_data import CalibrationData
from ..utils.logger import Logger

logger = Logger.get_logger(__name__)


class CalibrationError(Exception):
    """Raised when calibration fails."""

    pass


class Calibrator:
    """
    Stereo camera calibration manager.

    Handles the calibration process for dual cameras, computing:
    - Individual camera intrinsics (focal length, principal point)
    - Distortion coefficients
    - Stereo extrinsics (rotation/translation between cameras)
    - Rectification parameters for stereo matching

    Typical usage:
        calibrator = Calibrator(board_size=(9, 6), square_size=2.5)
        for left_img, right_img in image_pairs:
            calibrator.add_image_pair(left_img, right_img)
        calibration_data = calibrator.calibrate()
    """

    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 2.5,
        image_size: Tuple[int, int] = (640, 480),
    ):
        """
        Initialize calibrator with checkerboard parameters.

        Args:
            board_size: Number of inner corners (cols, rows) on checkerboard
            square_size: Size of each square in cm
            image_size: Image resolution (width, height)
        """
        self.board_size = board_size
        self.square_size = square_size
        self.image_size = image_size

        # Storage for calibration images
        self._left_images: List[np.ndarray] = []
        self._right_images: List[np.ndarray] = []
        self._object_points: List[np.ndarray] = []
        self._left_image_points: List[np.ndarray] = []
        self._right_image_points: List[np.ndarray] = []

        # Prepare object points (3D points of checkerboard corners)
        self._object_pattern = np.zeros(
            (board_size[0] * board_size[1], 3), dtype=np.float32
        )
        self._object_pattern[:, :2] = (
            np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
            * square_size
        )

        logger.info(
            f"Calibrator initialized: board_size={board_size}, "
            f"square_size={square_size}cm, image_size={image_size}"
        )

    def add_image_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> bool:
        """
        Add a stereo image pair for calibration.

        The checkerboard must be visible in both images.

        Args:
            left_image: Left camera image (grayscale or BGR)
            right_image: Right camera image (grayscale or BGR)

        Returns:
            True if checkerboard was found in both images
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image

        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image

        # Find checkerboard corners
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )

        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray, self.board_size, flags
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray, self.board_size, flags
        )

        if not ret_left or not ret_right:
            logger.debug("Checkerboard not found in one or both images")
            return False

        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(
            left_gray, corners_left, (11, 11), (-1, -1), criteria
        )
        corners_right = cv2.cornerSubPix(
            right_gray, corners_right, (11, 11), (-1, -1), criteria
        )

        # Store points
        self._object_points.append(self._object_pattern.copy())
        self._left_image_points.append(corners_left)
        self._right_image_points.append(corners_right)
        self._left_images.append(left_gray.copy())
        self._right_images.append(right_gray.copy())

        logger.info(f"Added calibration pair #{len(self._left_images)}")
        return True

    def get_num_pairs(self) -> int:
        """Return number of valid image pairs collected."""
        return len(self._left_images)

    def calibrate(self) -> CalibrationData:
        """
        Run stereo calibration with collected image pairs.

        Requires at least 10 image pairs for reliable calibration.

        Returns:
            CalibrationData with computed parameters

        Raises:
            CalibrationError: If calibration fails or insufficient data
        """
        if len(self._left_images) < 3:
            raise CalibrationError(
                f"Need at least 3 image pairs, have {len(self._left_images)}"
            )

        logger.info(
            f"Starting stereo calibration with {len(self._left_images)} image pairs"
        )

        # Calibration flags
        calib_flags = (
            cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        )

        # Calibrate individual cameras first
        logger.info("Calibrating left camera...")
        ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
            self._object_points,
            self._left_image_points,
            self.image_size,
            None,
            None,
            flags=calib_flags,
        )

        logger.info("Calibrating right camera...")
        ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
            self._object_points,
            self._right_image_points,
            self.image_size,
            None,
            None,
            flags=calib_flags,
        )

        logger.info(f"Left camera RMS error: {ret_left:.4f}")
        logger.info(f"Right camera RMS error: {ret_right:.4f}")

        # Stereo calibration
        logger.info("Performing stereo calibration...")
        stereo_flags = (
            cv2.CALIB_FIX_INTRINSIC  # Use intrinsics from individual calibration
        )

        (
            ret_stereo,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            self._object_points,
            self._left_image_points,
            self._right_image_points,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            self.image_size,
            flags=stereo_flags,
        )

        logger.info(f"Stereo calibration RMS error: {ret_stereo:.4f}")

        # Compute rectification transforms
        logger.info("Computing stereo rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            self.image_size,
            R,
            T,
            alpha=0,  # Only valid pixels
        )

        # Compute baseline from translation vector
        baseline = np.linalg.norm(T)

        logger.info(f"Stereo baseline: {baseline:.2f} cm")
        logger.info("Calibration complete")

        return CalibrationData(
            camera_left_matrix=mtx_left,
            camera_right_matrix=mtx_right,
            dist_coeffs_left=dist_left.flatten(),
            dist_coeffs_right=dist_right.flatten(),
            rotation_matrix=R,
            translation_vector=T.flatten(),
            rectify_left=R1,
            rectify_right=R2,
            projection_left=P1,
            projection_right=P2,
            disparity_to_depth=Q,
            stereo_baseline=baseline,
            image_size=self.image_size,
        )

    def visualize_detection(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize checkerboard detection in image pair.

        Args:
            left_image: Left camera image
            right_image: Right camera image

        Returns:
            Tuple of images with detected corners drawn
        """
        # Convert to color for drawing
        if len(left_image.shape) == 2:
            left_vis = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        else:
            left_vis = left_image.copy()

        if len(right_image.shape) == 2:
            right_vis = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
        else:
            right_vis = right_image.copy()

        # Detect corners
        ret_left, corners_left = cv2.findChessboardCorners(
            left_image, self.board_size, None
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            right_image, self.board_size, None
        )

        # Draw corners
        cv2.drawChessboardCorners(left_vis, self.board_size, corners_left, ret_left)
        cv2.drawChessboardCorners(right_vis, self.board_size, corners_right, ret_right)

        return left_vis, right_vis

    def clear(self) -> None:
        """Clear all collected calibration data."""
        self._left_images.clear()
        self._right_images.clear()
        self._object_points.clear()
        self._left_image_points.clear()
        self._right_image_points.clear()
        logger.info("Calibration data cleared")
