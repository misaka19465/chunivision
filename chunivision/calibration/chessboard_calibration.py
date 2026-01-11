"""
Chessboard calibration for stereo camera system.

This module provides chessboard-based calibration for the stereo camera
system. An A4 printed chessboard is placed on a horizontal plane and
detected to compute camera intrinsic and extrinsic parameters.

The calibration process:
1. Detect chessboard corners in multiple frames from each camera
2. Compute individual camera intrinsics (camera matrix, distortion)
3. Compute stereo calibration (rotation, translation between cameras)
4. Compute rectification transforms for stereo matching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import Logger
from .lens_distortion import LensDistortion, LensDistortionParams

logger = Logger.get_logger(__name__)


@dataclass
class ChessboardConfig:
    """
    Configuration for chessboard calibration.

    Default values match the printable calibration pattern:
    docs/calibration_chessboard.svg

    Pattern specifications:
    - Inner corners: 9 × 6 (10 × 7 squares)
    - Square size: 25mm (2.5cm)
    - Pattern size: 250mm × 175mm
    - Paper: A4 (297mm × 210mm)

    Print the SVG at 100% scale (no fit-to-page) for accurate dimensions.

    Attributes:
        pattern_size: Number of inner corners (columns, rows)
        square_size: Physical size of each square in cm
        a4_paper_size: A4 paper size (width, height) in cm
        min_captures: Minimum number of valid captures required
        min_quality_threshold: Minimum quality score to accept a capture
        corner_refinement_window: Window size for corner refinement
        corner_refinement_iterations: Max iterations for corner refinement
        corner_refinement_epsilon: Convergence epsilon for corner refinement
    """

    # Pattern matches docs/calibration_chessboard.svg
    pattern_size: Tuple[int, int] = (9, 6)  # 9×6 inner corners (10×7 squares)
    square_size: float = 2.5  # 25mm = 2.5cm per square
    a4_paper_size: Tuple[float, float] = (29.7, 21.0)  # A4 in cm (landscape)
    min_captures: int = 5  # Minimum captures needed
    min_quality_threshold: float = 0.5
    corner_refinement_window: Tuple[int, int] = (11, 11)
    corner_refinement_iterations: int = 30
    corner_refinement_epsilon: float = 0.001

    def get_object_points(self) -> np.ndarray:
        """
        Generate 3D object points for the chessboard pattern.

        Returns:
            Array of shape (rows*cols, 3) with (x, y, 0) coordinates
        """
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp


@dataclass
class ChessboardCapture:
    """
    A single chessboard capture result.

    Attributes:
        frame: The captured frame (grayscale)
        corners: Detected corner points, shape (N, 1, 2)
        object_points: Corresponding 3D object points
        quality: Detection quality score (0-1)
        timestamp: Capture timestamp
    """

    frame: np.ndarray
    corners: np.ndarray
    object_points: np.ndarray
    quality: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChessboardCalibrationResult:
    """
    Result of chessboard calibration for a single camera.

    Attributes:
        camera_matrix: 3x3 intrinsic camera matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors for each capture
        tvecs: Translation vectors for each capture
        reprojection_error: Mean reprojection error
        num_captures: Number of captures used
        image_size: Image size (width, height)
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    reprojection_error: float
    num_captures: int
    image_size: Tuple[int, int]

    def get_quality_score(self) -> float:
        """
        Get overall quality score based on reprojection error.

        Returns:
            Quality score between 0 and 1
        """
        # Lower reprojection error = higher quality
        # Error < 0.5 pixels = 1.0, error > 5 pixels = ~0
        return float(np.exp(-self.reprojection_error / 2.0))


@dataclass
class StereoCalibrationResult:
    """
    Result of stereo calibration.

    Attributes:
        left_result: Left camera calibration result
        right_result: Right camera calibration result
        rotation_matrix: 3x3 rotation from left to right camera
        translation_vector: Translation from left to right camera
        essential_matrix: 3x3 essential matrix
        fundamental_matrix: 3x3 fundamental matrix
        rectify_left: 3x3 rectification transform for left camera
        rectify_right: 3x3 rectification transform for right camera
        projection_left: 3x4 projection matrix for left camera
        projection_right: 3x4 projection matrix for right camera
        disparity_to_depth: 4x4 Q matrix for disparity-to-depth
        roi_left: Valid ROI for left camera after rectification
        roi_right: Valid ROI for right camera after rectification
        stereo_error: Stereo calibration reprojection error
    """

    left_result: ChessboardCalibrationResult
    right_result: ChessboardCalibrationResult
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    rectify_left: np.ndarray
    rectify_right: np.ndarray
    projection_left: np.ndarray
    projection_right: np.ndarray
    disparity_to_depth: np.ndarray
    roi_left: Tuple[int, int, int, int]
    roi_right: Tuple[int, int, int, int]
    stereo_error: float

    @property
    def baseline(self) -> float:
        """Get stereo baseline in cm (distance between cameras)."""
        return float(np.linalg.norm(self.translation_vector))

    def get_quality_score(self) -> float:
        """Get overall stereo calibration quality score."""
        left_q = self.left_result.get_quality_score()
        right_q = self.right_result.get_quality_score()
        stereo_q = float(np.exp(-self.stereo_error / 2.0))
        return (left_q + right_q + stereo_q) / 3.0


class ChessboardCalibrator:
    """
    Chessboard-based camera calibration.

    This class handles:
    1. Automatic chessboard detection and corner refinement
    2. Single camera intrinsic calibration
    3. Stereo camera calibration
    4. Integration with lens distortion correction

    The calibration workflow:
    1. User places A4 chessboard on horizontal surface
    2. System captures multiple frames from each camera
    3. Chessboard corners are detected and refined
    4. Camera intrinsics and stereo extrinsics are computed

    Example:
        config = ChessboardConfig(pattern_size=(9, 6), square_size=2.5)
        calibrator = ChessboardCalibrator(config)

        # Capture frames and detect chessboards
        for frame in get_frames():
            calibrator.add_capture("left", frame)

        # Perform calibration
        result = calibrator.calibrate_camera("left")
    """

    def __init__(
        self,
        config: Optional[ChessboardConfig] = None,
        lens_distortion: Optional[LensDistortion] = None,
    ):
        """
        Initialize chessboard calibrator.

        Args:
            config: Chessboard configuration. If None, uses defaults.
            lens_distortion: Lens distortion handler for pre-undistortion.
                            If provided, frames are undistorted before detection.
        """
        self.config = config or ChessboardConfig()
        self.lens_distortion = lens_distortion

        # Storage for captures
        self._left_captures: List[ChessboardCapture] = []
        self._right_captures: List[ChessboardCapture] = []

        # Object points (same for all captures)
        self._object_points = self.config.get_object_points()

        # Callbacks
        self._on_progress: Optional[Callable[[str, float], None]] = None
        self._on_message: Optional[Callable[[str], None]] = None

        logger.info(
            f"ChessboardCalibrator initialized: pattern={self.config.pattern_size}, "
            f"square_size={self.config.square_size}cm"
        )

    def set_callbacks(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Set callback functions for UI feedback."""
        self._on_progress = on_progress
        self._on_message = on_message

    def _notify_progress(self, step: str, progress: float) -> None:
        """Notify progress callback."""
        if self._on_progress:
            self._on_progress(step, progress)

    def _notify_message(self, message: str) -> None:
        """Notify message callback."""
        if self._on_message:
            self._on_message(message)
        logger.info(message)

    def detect_chessboard(
        self,
        frame: np.ndarray,
        undistort_first: bool = True,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detect chessboard corners in a frame.

        Args:
            frame: Input frame (grayscale or BGR)
            undistort_first: Whether to undistort the frame first

        Returns:
            Tuple of (corners, quality) if found, None otherwise
            - corners: shape (N, 1, 2) refined corner positions
            - quality: detection quality score (0-1)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Undistort if lens distortion handler is available
        if undistort_first and self.lens_distortion is not None:
            gray = self.lens_distortion.undistort_frame(gray)

        # Find chessboard corners
        cols, rows = self.config.pattern_size
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

        if not found or corners is None:
            return None

        # Refine corner positions
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            self.config.corner_refinement_iterations,
            self.config.corner_refinement_epsilon,
        )
        corners = cv2.cornerSubPix(
            gray,
            corners,
            self.config.corner_refinement_window,
            (-1, -1),
            criteria,
        )

        # Compute quality based on corner regularity
        quality = self._compute_detection_quality(corners)

        return corners, quality

    def _compute_detection_quality(self, corners: np.ndarray) -> float:
        """
        Compute detection quality based on corner arrangement.

        Checks for:
        - Regular spacing between corners
        - Reasonable perspective distortion
        - No missing or misaligned corners

        Args:
            corners: Detected corners, shape (N, 1, 2)

        Returns:
            Quality score between 0 and 1
        """
        cols, rows = self.config.pattern_size
        corners_2d = corners.reshape(-1, 2)

        # Check spacing consistency
        # Compute horizontal and vertical distances
        quality_scores = []

        # Horizontal spacing
        for row in range(rows):
            row_corners = corners_2d[row * cols : (row + 1) * cols]
            if len(row_corners) >= 2:
                distances = np.linalg.norm(np.diff(row_corners, axis=0), axis=1)
                if len(distances) > 0:
                    # Coefficient of variation (lower is better)
                    cv_score = 1.0 - min(
                        1.0, np.std(distances) / (np.mean(distances) + 1e-6)
                    )
                    quality_scores.append(cv_score)

        # Vertical spacing
        for col in range(cols):
            col_corners = corners_2d[col::cols]
            if len(col_corners) >= 2:
                distances = np.linalg.norm(np.diff(col_corners, axis=0), axis=1)
                if len(distances) > 0:
                    cv_score = 1.0 - min(
                        1.0, np.std(distances) / (np.mean(distances) + 1e-6)
                    )
                    quality_scores.append(cv_score)

        if not quality_scores:
            return 0.5

        return float(np.mean(quality_scores))

    def add_capture(
        self,
        camera: str,
        frame: np.ndarray,
        undistort_first: bool = True,
    ) -> Optional[ChessboardCapture]:
        """
        Add a capture from a camera.

        Detects chessboard and stores if found.

        Args:
            camera: Camera identifier ("left" or "right")
            frame: Input frame
            undistort_first: Whether to undistort before detection

        Returns:
            ChessboardCapture if chessboard was found, None otherwise
        """
        result = self.detect_chessboard(frame, undistort_first)

        if result is None:
            logger.debug(f"No chessboard found in {camera} camera frame")
            return None

        corners, quality = result

        if quality < self.config.min_quality_threshold:
            logger.debug(
                f"Chessboard quality too low: {quality:.2f} < {self.config.min_quality_threshold}"
            )
            return None

        # Store grayscale version
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        capture = ChessboardCapture(
            frame=gray,
            corners=corners,
            object_points=self._object_points.copy(),
            quality=quality,
        )

        if camera == "left":
            self._left_captures.append(capture)
        elif camera == "right":
            self._right_captures.append(capture)
        else:
            raise ValueError(f"Unknown camera: {camera}")

        logger.info(
            f"Added chessboard capture for {camera} camera "
            f"(quality={quality:.2f}, total={self.get_capture_count(camera)})"
        )

        return capture

    def get_capture_count(self, camera: str) -> int:
        """Get number of captures for a camera."""
        if camera == "left":
            return len(self._left_captures)
        elif camera == "right":
            return len(self._right_captures)
        else:
            raise ValueError(f"Unknown camera: {camera}")

    def clear_captures(self, camera: Optional[str] = None) -> None:
        """
        Clear stored captures.

        Args:
            camera: Camera to clear, or None for all cameras
        """
        if camera is None or camera == "left":
            self._left_captures.clear()
        if camera is None or camera == "right":
            self._right_captures.clear()
        logger.info(f"Cleared captures for: {camera or 'all cameras'}")

    def calibrate_camera(
        self,
        camera: str,
        flags: int = 0,
    ) -> ChessboardCalibrationResult:
        """
        Calibrate a single camera.

        Args:
            camera: Camera identifier ("left" or "right")
            flags: OpenCV calibration flags

        Returns:
            ChessboardCalibrationResult with intrinsic parameters

        Raises:
            ValueError: If insufficient captures
        """
        captures = self._left_captures if camera == "left" else self._right_captures

        if len(captures) < self.config.min_captures:
            raise ValueError(
                f"Insufficient captures for {camera} camera: "
                f"{len(captures)} < {self.config.min_captures}"
            )

        # Prepare calibration data
        object_points = [c.object_points for c in captures]
        image_points = [c.corners for c in captures]
        image_size = (captures[0].frame.shape[1], captures[0].frame.shape[0])

        self._notify_message(
            f"Calibrating {camera} camera with {len(captures)} captures..."
        )

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=flags,
        )

        # Compute mean reprojection error
        total_error = 0.0
        for i, (objp, imgp, rvec, tvec) in enumerate(
            zip(object_points, image_points, rvecs, tvecs)
        ):
            projected, _ = cv2.projectPoints(
                objp, rvec, tvec, camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
            total_error += error
        mean_error = total_error / len(object_points)

        result = ChessboardCalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs.flatten(),
            rvecs=list(rvecs),
            tvecs=list(tvecs),
            reprojection_error=mean_error,
            num_captures=len(captures),
            image_size=image_size,
        )

        self._notify_message(
            f"{camera.capitalize()} camera calibrated: "
            f"reprojection error = {mean_error:.4f} pixels, "
            f"quality = {result.get_quality_score():.1%}"
        )

        return result

    def calibrate_stereo(
        self,
        left_result: Optional[ChessboardCalibrationResult] = None,
        right_result: Optional[ChessboardCalibrationResult] = None,
        flags: int = cv2.CALIB_FIX_INTRINSIC,
    ) -> StereoCalibrationResult:
        """
        Perform stereo calibration.

        Uses paired captures from both cameras to compute the geometric
        relationship between them.

        Args:
            left_result: Pre-computed left camera calibration (optional)
            right_result: Pre-computed right camera calibration (optional)
            flags: OpenCV stereo calibration flags

        Returns:
            StereoCalibrationResult with stereo parameters

        Raises:
            ValueError: If insufficient paired captures
        """
        # Calibrate individual cameras if not provided
        if left_result is None:
            left_result = self.calibrate_camera("left")
        if right_result is None:
            right_result = self.calibrate_camera("right")

        # Use minimum of both capture counts
        min_captures = min(len(self._left_captures), len(self._right_captures))

        if min_captures < self.config.min_captures:
            raise ValueError(
                f"Insufficient paired captures: {min_captures} < {self.config.min_captures}"
            )

        # Use paired captures (assuming synchronized capture)
        object_points = []
        left_image_points = []
        right_image_points = []

        for i in range(min_captures):
            object_points.append(self._left_captures[i].object_points)
            left_image_points.append(self._left_captures[i].corners)
            right_image_points.append(self._right_captures[i].corners)

        image_size = left_result.image_size

        self._notify_message(
            f"Performing stereo calibration with {min_captures} pairs..."
        )

        # Stereo calibration
        ret, cm1, dc1, cm2, dc2, R, T, E, F = cv2.stereoCalibrate(
            object_points,
            left_image_points,
            right_image_points,
            left_result.camera_matrix,
            left_result.dist_coeffs,
            right_result.camera_matrix,
            right_result.dist_coeffs,
            image_size,
            flags=flags,
        )

        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_result.camera_matrix,
            left_result.dist_coeffs,
            right_result.camera_matrix,
            right_result.dist_coeffs,
            image_size,
            R,
            T,
            alpha=0,  # Crop invalid regions
        )

        result = StereoCalibrationResult(
            left_result=left_result,
            right_result=right_result,
            rotation_matrix=R,
            translation_vector=T.flatten(),
            essential_matrix=E,
            fundamental_matrix=F,
            rectify_left=R1,
            rectify_right=R2,
            projection_left=P1,
            projection_right=P2,
            disparity_to_depth=Q,
            roi_left=tuple(roi1),
            roi_right=tuple(roi2),
            stereo_error=ret,
        )

        self._notify_message(
            f"Stereo calibration complete: "
            f"error = {ret:.4f} pixels, "
            f"baseline = {result.baseline:.2f} cm, "
            f"quality = {result.get_quality_score():.1%}"
        )

        return result

    def run_interactive_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
        target_captures: int = 10,
        auto_capture: bool = True,
        capture_delay: float = 1.0,
    ) -> StereoCalibrationResult:
        """
        Run interactive stereo calibration workflow.

        Displays camera views and guides the user through the calibration process.

        Args:
            left_camera: Left camera instance (must have get_frame())
            right_camera: Right camera instance (must have get_frame())
            target_captures: Target number of captures per camera
            auto_capture: Automatically capture when chessboard is stable
            capture_delay: Minimum delay between captures in seconds

        Returns:
            StereoCalibrationResult

        Raises:
            RuntimeError: If calibration fails
        """
        import time

        self._notify_message("Starting interactive chessboard calibration")
        self._notify_message(
            f"Place A4 chessboard ({self.config.pattern_size[0]}x{self.config.pattern_size[1]}) "
            f"on horizontal surface"
        )
        self._notify_progress("Initializing", 0.0)

        # Create display windows
        window_left = "Left Camera - Chessboard Calibration"
        window_right = "Right Camera - Chessboard Calibration"
        cv2.namedWindow(window_left, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_right, cv2.WINDOW_AUTOSIZE)

        last_capture_time = 0.0
        last_left_corners = None
        last_right_corners = None
        stable_frames = 0
        stable_threshold = 5  # Need 5 stable frames to auto-capture

        try:
            while True:
                # Get frames
                left_frame = self._get_camera_frame(left_camera)
                right_frame = self._get_camera_frame(right_camera)

                # Detect chessboard
                left_result = self.detect_chessboard(left_frame)
                right_result = self.detect_chessboard(right_frame)

                # Prepare display frames
                left_display = self._prepare_display(left_frame, left_result)
                right_display = self._prepare_display(right_frame, right_result)

                # Add capture count overlay
                self._add_status_overlay(left_display, "left", target_captures)
                self._add_status_overlay(right_display, "right", target_captures)

                # Check for stable detection (for auto-capture)
                if auto_capture and left_result and right_result:
                    left_corners, left_q = left_result
                    right_corners, right_q = right_result

                    # Check stability
                    is_stable = True
                    if last_left_corners is not None:
                        movement = np.mean(np.abs(left_corners - last_left_corners))
                        if movement > 2.0:  # More than 2 pixels movement
                            is_stable = False

                    if is_stable:
                        stable_frames += 1
                    else:
                        stable_frames = 0

                    last_left_corners = left_corners.copy()
                    last_right_corners = right_corners.copy()

                    # Auto-capture if stable and enough time passed
                    current_time = time.time()
                    if (
                        stable_frames >= stable_threshold
                        and current_time - last_capture_time > capture_delay
                    ):
                        self.add_capture("left", left_frame)
                        self.add_capture("right", right_frame)
                        last_capture_time = current_time
                        stable_frames = 0

                        left_count = self.get_capture_count("left")
                        self._notify_message(
                            f"Captured pair {left_count}/{target_captures}"
                        )
                        self._notify_progress(
                            "Capturing",
                            min(1.0, left_count / target_captures * 0.5),
                        )

                        # Check if we have enough
                        if (
                            self.get_capture_count("left") >= target_captures
                            and self.get_capture_count("right") >= target_captures
                        ):
                            break
                else:
                    stable_frames = 0
                    last_left_corners = None
                    last_right_corners = None

                # Display
                cv2.imshow(window_left, left_display)
                cv2.imshow(window_right, right_display)

                # Handle key press
                key = cv2.waitKey(30) & 0xFF

                if key == 27:  # ESC - cancel
                    raise RuntimeError("Calibration cancelled by user")

                elif key == ord(" "):  # Space - manual capture
                    if left_result and right_result:
                        self.add_capture("left", left_frame)
                        self.add_capture("right", right_frame)
                        last_capture_time = time.time()
                        self._notify_message(
                            f"Manual capture: {self.get_capture_count('left')}/{target_captures}"
                        )

                elif key == ord("c"):  # C - calibrate now
                    if self.get_capture_count("left") >= self.config.min_captures:
                        break

                elif key == ord("r"):  # R - reset
                    self.clear_captures()
                    self._notify_message("Captures cleared")

        finally:
            cv2.destroyWindow(window_left)
            cv2.destroyWindow(window_right)

        # Perform calibration
        self._notify_progress("Calibrating", 0.6)
        return self.calibrate_stereo()

    def _get_camera_frame(self, camera: Any) -> np.ndarray:
        """Get frame from camera object."""
        if hasattr(camera, "get_frame"):
            return camera.get_frame()
        elif hasattr(camera, "read"):
            ret, frame = camera.read()
            if not ret:
                raise RuntimeError("Failed to read from camera")
            return frame
        else:
            raise RuntimeError("Camera must have get_frame() or read() method")

    def _prepare_display(
        self,
        frame: np.ndarray,
        detection_result: Optional[Tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """
        Prepare frame for display with chessboard overlay.

        Args:
            frame: Input frame
            detection_result: Detection result (corners, quality) or None

        Returns:
            BGR display frame with overlays
        """
        # Convert to BGR
        if len(frame.shape) == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        if detection_result is not None:
            corners, quality = detection_result
            cols, rows = self.config.pattern_size

            # Draw chessboard corners
            cv2.drawChessboardCorners(display, (cols, rows), corners, True)

            # Quality indicator
            color = (
                (0, 255, 0)
                if quality > 0.7
                else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
            )
            cv2.putText(
                display,
                f"Quality: {quality:.1%}",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        else:
            # No detection
            cv2.putText(
                display,
                "No chessboard detected",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        return display

    def _add_status_overlay(
        self,
        display: np.ndarray,
        camera: str,
        target: int,
    ) -> None:
        """Add status overlay to display frame."""
        count = self.get_capture_count(camera)
        text = f"{camera.upper()}: {count}/{target} captures"
        color = (0, 255, 0) if count >= target else (255, 255, 255)

        cv2.putText(
            display,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # Instructions
        cv2.putText(
            display,
            "SPACE: Manual capture | C: Calibrate | R: Reset | ESC: Cancel",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    def draw_debug_info(
        self,
        frame: np.ndarray,
        capture: ChessboardCapture,
    ) -> np.ndarray:
        """
        Draw debug visualization for a capture.

        Args:
            frame: Frame to draw on
            capture: Capture to visualize

        Returns:
            Frame with debug overlays
        """
        if len(frame.shape) == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        # Draw corners
        cols, rows = self.config.pattern_size
        cv2.drawChessboardCorners(display, (cols, rows), capture.corners, True)

        # Draw axes at first corner
        if len(capture.corners) > 0:
            origin = tuple(capture.corners[0].ravel().astype(int))
            # X axis (red)
            if cols > 1:
                x_pt = tuple(capture.corners[1].ravel().astype(int))
                cv2.arrowedLine(display, origin, x_pt, (0, 0, 255), 2)
            # Y axis (green)
            if rows > 1:
                y_pt = tuple(capture.corners[cols].ravel().astype(int))
                cv2.arrowedLine(display, origin, y_pt, (0, 255, 0), 2)

        return display
