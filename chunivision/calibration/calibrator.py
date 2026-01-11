"""
Complete calibration workflow coordinator for ChunIVision.

This module provides a comprehensive two-stage calibration process:

Stage 1: Chessboard Calibration
- User places A4 printed chessboard on horizontal surface
- System captures multiple views and detects chessboard corners
- Computes lens distortion parameters and stereo geometry
- No iteration limit - continues until quality thresholds are met

Stage 2: Touch Zone Calibration
- User places white paper matching the game touch area size
- System detects paper edges or user manually selects corners
- Computes perspective transform for touch zone mapping
- Validates by overlaying zone grid on camera view

The calibration prioritizes quality over speed - it will continue
capturing and refining until acceptable accuracy is achieved.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import Logger
from .calibration_data import CalibrationData, CalibrationDataError
from .chessboard_calibration import (
    ChessboardCalibrator,
    ChessboardConfig,
    StereoCalibrationResult,
)
from .lens_distortion import LensDistortion, LensDistortionParams
from .touchzone_calibration import (
    TouchZoneCalibrator,
    TouchZoneCalibrationResult,
    TouchZoneConfig,
)
from .transform_calculator import TransformCalculator

logger = Logger.get_logger(__name__)


class CalibrationError(Exception):
    """Raised when calibration fails."""

    pass


@dataclass
class CalibrationConfig:
    """
    Configuration for the complete calibration workflow.

    Attributes:
        chessboard_config: Chessboard detection configuration
        touchzone_config: Touch zone calibration configuration
        lens_distortion_params: Fixed lens distortion parameters
        target_stereo_quality: Target quality score for stereo calibration
        target_zone_quality: Target quality score for zone calibration
        min_stereo_quality: Minimum acceptable stereo quality
        min_zone_quality: Minimum acceptable zone quality
        max_stereo_captures: Maximum captures for stereo calibration
        stereo_capture_delay: Delay between stereo captures (seconds)
        auto_accept_timeout: Timeout for auto-accepting calibration (seconds)
    """

    chessboard_config: ChessboardConfig = None
    touchzone_config: TouchZoneConfig = None
    lens_distortion_params: LensDistortionParams = None
    target_stereo_quality: float = 0.9
    target_zone_quality: float = 0.95
    min_stereo_quality: float = 0.7
    min_zone_quality: float = 0.8
    max_stereo_captures: int = 50  # No artificial limit, but prevent infinite loops
    stereo_capture_delay: float = 1.0
    auto_accept_timeout: float = 10.0

    def __post_init__(self):
        if self.chessboard_config is None:
            self.chessboard_config = ChessboardConfig()
        if self.touchzone_config is None:
            self.touchzone_config = TouchZoneConfig()
        if self.lens_distortion_params is None:
            self.lens_distortion_params = LensDistortionParams.get_default()


@dataclass
class FullCalibrationResult:
    """
    Complete result from the two-stage calibration process.

    Attributes:
        stereo_result: Result from chessboard stereo calibration
        left_zone_result: Left camera touch zone calibration
        right_zone_result: Right camera touch zone calibration
        lens_distortion: Lens distortion handler
        timestamp: Calibration timestamp
        overall_quality: Combined quality score
    """

    stereo_result: StereoCalibrationResult
    left_zone_result: TouchZoneCalibrationResult
    right_zone_result: TouchZoneCalibrationResult
    lens_distortion: LensDistortion
    timestamp: datetime
    overall_quality: float

    def to_calibration_data(self) -> CalibrationData:
        """
        Convert to CalibrationData format for storage.

        Returns:
            CalibrationData instance with all calibration parameters
        """
        # Combine zone boundaries from both cameras
        zone_boundaries = {
            "grid": list(self.left_zone_result.zone_boundaries[0].corners.shape),
            "left_zones": [
                {
                    "id": z.zone_id,
                    "corners": z.corners.tolist(),
                    "center": z.center.tolist(),
                }
                for z in self.left_zone_result.zone_boundaries
            ],
            "right_zones": [
                {
                    "id": z.zone_id,
                    "corners": z.corners.tolist(),
                    "center": z.center.tolist(),
                }
                for z in self.right_zone_result.zone_boundaries
            ],
        }

        return CalibrationData(
            version="2.0",  # New version for two-stage calibration
            timestamp=self.timestamp,
            camera_left_matrix=self.stereo_result.left_result.camera_matrix,
            camera_right_matrix=self.stereo_result.right_result.camera_matrix,
            dist_coeffs_left=self.stereo_result.left_result.dist_coeffs,
            dist_coeffs_right=self.stereo_result.right_result.dist_coeffs,
            rotation_matrix=self.stereo_result.rotation_matrix,
            translation_vector=self.stereo_result.translation_vector,
            rectify_left=self.stereo_result.rectify_left,
            rectify_right=self.stereo_result.rectify_right,
            projection_left=self.stereo_result.projection_left,
            projection_right=self.stereo_result.projection_right,
            disparity_to_depth=self.stereo_result.disparity_to_depth,
            stereo_baseline=self.stereo_result.baseline,
            image_size=self.stereo_result.left_result.image_size,
            zone_boundaries=zone_boundaries,
            height_thresholds=np.array([17.9, 21.3, 24.7, 28.1, 31.5, 34.9]),
            reference_board_size=self.left_zone_result.physical_corners.max(
                axis=0
            ).tolist(),
            camera_left_transform=self.left_zone_result.perspective_transform,
            camera_right_transform=self.right_zone_result.perspective_transform,
            calibration_quality={
                "stereo_quality": self.stereo_result.get_quality_score(),
                "left_zone_quality": self.left_zone_result.quality_score,
                "right_zone_quality": self.right_zone_result.quality_score,
                "overall_quality": self.overall_quality,
                "stereo_error": self.stereo_result.stereo_error,
                "left_zone_error": self.left_zone_result.reprojection_error,
                "right_zone_error": self.right_zone_result.reprojection_error,
            },
            left_calibration_points=self.left_zone_result.image_corners,
            right_calibration_points=self.right_zone_result.image_corners,
        )


class Calibrator:
    """
    Complete two-stage calibration workflow coordinator.

    This calibrator implements a quality-first approach:
    1. Lens distortion is handled using fixed factory parameters
    2. Chessboard calibration establishes stereo geometry
    3. Touch zone calibration maps camera views to game zones

    The calibration will continue until quality thresholds are met,
    not based on a fixed number of iterations.

    Example:
        calibrator = Calibrator()
        result = calibrator.run_full_calibration(left_camera, right_camera)
        calibrator.save_calibration(result, "calibration.yaml")
    """

    # Default height thresholds for 6 air sensor levels (cm above surface)
    DEFAULT_HEIGHT_THRESHOLDS = [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]

    # Quality thresholds for legacy compatibility
    MIN_QUALITY_THRESHOLD = 0.7
    RECOMMENDED_QUALITY_THRESHOLD = 0.85

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize calibrator with configuration.

        Args:
            config: Calibration configuration. If None, uses defaults.
        """
        self.config = config or CalibrationConfig()

        # Initialize lens distortion with fixed parameters
        self.lens_distortion = LensDistortion(self.config.lens_distortion_params)

        # Initialize sub-calibrators
        self.chessboard_calibrator = ChessboardCalibrator(
            config=self.config.chessboard_config,
            lens_distortion=self.lens_distortion,
        )
        self.touchzone_calibrator = TouchZoneCalibrator(
            config=self.config.touchzone_config,
            lens_distortion=self.lens_distortion,
        )

        # Legacy attributes for backward compatibility
        self.board_physical_size = (
            self.config.touchzone_config.paper_size_cm[0],
            self.config.touchzone_config.paper_size_cm[1],
        )
        self.zone_grid = self.config.touchzone_config.zone_grid
        self.image_size = (640, 480)
        self.stereo_baseline = 20.0

        # Define physical corner positions for legacy compatibility
        self._world_points = np.array(
            [
                [0, 0],
                [self.board_physical_size[0], 0],
                [self.board_physical_size[0], self.board_physical_size[1]],
                [0, self.board_physical_size[1]],
            ],
            dtype=np.float32,
        )

        # Callbacks
        self._on_progress: Optional[Callable[[str, float], None]] = None
        self._on_message: Optional[Callable[[str], None]] = None

        logger.info("Calibrator initialized with two-stage calibration workflow")

    def set_callbacks(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Set callback functions for UI feedback.

        Args:
            on_progress: Called with (step_name, progress_0_to_1)
            on_message: Called with message string
        """
        self._on_progress = on_progress
        self._on_message = on_message

        # Pass to sub-calibrators
        self.chessboard_calibrator.set_callbacks(on_progress, on_message)
        self.touchzone_calibrator.set_callbacks(on_progress, on_message)

    def _notify_progress(self, step: str, progress: float) -> None:
        """Notify progress callback."""
        if self._on_progress:
            self._on_progress(step, progress)

    def _notify_message(self, message: str) -> None:
        """Notify message callback."""
        if self._on_message:
            self._on_message(message)
        logger.info(message)

    def run_full_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
    ) -> FullCalibrationResult:
        """
        Run the complete two-stage calibration process.

        Stage 1: Chessboard stereo calibration
        Stage 2: Touch zone calibration for each camera

        Args:
            left_camera: Left camera instance
            right_camera: Right camera instance

        Returns:
            FullCalibrationResult with complete calibration

        Raises:
            CalibrationError: If calibration fails
        """
        self._notify_message("=" * 60)
        self._notify_message(
            "ChunIVision Calibration - Two-Stage Quality-First Process"
        )
        self._notify_message("=" * 60)

        timestamp = datetime.now()

        # Stage 1: Chessboard stereo calibration
        self._notify_progress("Stage 1: Stereo Calibration", 0.0)
        self._notify_message("\nSTAGE 1: CHESSBOARD STEREO CALIBRATION")
        self._notify_message("-" * 40)
        stereo_result = self._run_stage1_stereo_calibration(left_camera, right_camera)

        # Stage 2: Touch zone calibration
        self._notify_progress("Stage 2: Zone Calibration", 0.5)
        self._notify_message("\nSTAGE 2: TOUCH ZONE CALIBRATION")
        self._notify_message("-" * 40)
        left_zone, right_zone = self._run_stage2_zone_calibration(
            left_camera, right_camera
        )

        # Compute overall quality
        overall_quality = (
            stereo_result.get_quality_score() * 0.3
            + left_zone.quality_score * 0.35
            + right_zone.quality_score * 0.35
        )

        result = FullCalibrationResult(
            stereo_result=stereo_result,
            left_zone_result=left_zone,
            right_zone_result=right_zone,
            lens_distortion=self.lens_distortion,
            timestamp=timestamp,
            overall_quality=overall_quality,
        )

        self._notify_progress("Calibration Complete", 1.0)
        self._notify_message("\n" + "=" * 60)
        self._notify_message("CALIBRATION COMPLETE")
        self._notify_message(f"Overall Quality: {overall_quality:.1%}")
        self._notify_message(
            f"  Stereo Quality: {stereo_result.get_quality_score():.1%}"
        )
        self._notify_message(f"  Left Zone Quality: {left_zone.quality_score:.1%}")
        self._notify_message(f"  Right Zone Quality: {right_zone.quality_score:.1%}")
        self._notify_message("=" * 60)

        return result

    def _run_stage1_stereo_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
    ) -> StereoCalibrationResult:
        """
        Run Stage 1: Chessboard stereo calibration.

        Captures chessboard images until quality threshold is met.
        No iteration limit - quality first!

        Args:
            left_camera: Left camera
            right_camera: Right camera

        Returns:
            StereoCalibrationResult

        Raises:
            CalibrationError: If calibration fails
        """
        import time

        self._notify_message(
            f"Place A4 chessboard ({self.config.chessboard_config.pattern_size[0]}x"
            f"{self.config.chessboard_config.pattern_size[1]}) on a horizontal surface"
        )
        self._notify_message("Move the chessboard to different positions and angles")
        self._notify_message(
            f"Target quality: {self.config.target_stereo_quality:.0%}, "
            f"Minimum: {self.config.min_stereo_quality:.0%}"
        )

        # Create display windows
        window_left = "Left Camera - Chessboard (Stage 1)"
        window_right = "Right Camera - Chessboard (Stage 1)"
        window_status = "Calibration Status"
        cv2.namedWindow(window_left, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_right, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_status, cv2.WINDOW_AUTOSIZE)

        last_capture_time = 0.0
        last_quality = 0.0
        best_result = None

        try:
            while True:
                # Get frames
                left_frame = self._get_camera_frame(left_camera)
                right_frame = self._get_camera_frame(right_camera)

                # Detect chessboard
                left_result = self.chessboard_calibrator.detect_chessboard(left_frame)
                right_result = self.chessboard_calibrator.detect_chessboard(right_frame)

                # Prepare display
                left_display = self._prepare_chessboard_display(left_frame, left_result)
                right_display = self._prepare_chessboard_display(
                    right_frame, right_result
                )

                # Status display
                status_display = self._create_status_display(
                    self.chessboard_calibrator.get_capture_count("left"),
                    last_quality,
                    self.config.target_stereo_quality,
                    self.config.min_stereo_quality,
                )

                # Check for stable detection and auto-capture
                current_time = time.time()
                if (
                    left_result is not None
                    and right_result is not None
                    and current_time - last_capture_time
                    > self.config.stereo_capture_delay
                ):
                    # Add captures
                    self.chessboard_calibrator.add_capture("left", left_frame)
                    self.chessboard_calibrator.add_capture("right", right_frame)
                    last_capture_time = current_time

                    count = self.chessboard_calibrator.get_capture_count("left")
                    self._notify_message(f"Captured pair {count}")

                    # Try calibration if we have enough captures
                    if count >= self.config.chessboard_config.min_captures:
                        try:
                            result = self.chessboard_calibrator.calibrate_stereo()
                            last_quality = result.get_quality_score()

                            if (
                                best_result is None
                                or last_quality > best_result.get_quality_score()
                            ):
                                best_result = result

                            # Check if quality is sufficient
                            if last_quality >= self.config.target_stereo_quality:
                                self._notify_message(
                                    f"Target quality achieved: {last_quality:.1%}"
                                )
                                break

                            # Check for maximum captures
                            if count >= self.config.max_stereo_captures:
                                if last_quality >= self.config.min_stereo_quality:
                                    self._notify_message(
                                        f"Maximum captures reached, using best quality: {last_quality:.1%}"
                                    )
                                    break
                                else:
                                    self._notify_message(
                                        f"Quality too low ({last_quality:.1%}), continuing..."
                                    )

                        except Exception as e:
                            logger.debug(f"Calibration attempt failed: {e}")

                # Display
                cv2.imshow(window_left, left_display)
                cv2.imshow(window_right, right_display)
                cv2.imshow(window_status, status_display)

                # Handle key press
                key = cv2.waitKey(30) & 0xFF

                if key == 27:  # ESC
                    raise CalibrationError("Calibration cancelled by user")

                elif key == ord("c"):  # Force calibration
                    count = self.chessboard_calibrator.get_capture_count("left")
                    if count >= self.config.chessboard_config.min_captures:
                        result = self.chessboard_calibrator.calibrate_stereo()
                        if result.get_quality_score() >= self.config.min_stereo_quality:
                            return result
                        self._notify_message(
                            f"Quality too low: {result.get_quality_score():.1%}"
                        )

                elif key == ord("r"):  # Reset
                    self.chessboard_calibrator.clear_captures()
                    best_result = None
                    last_quality = 0.0
                    self._notify_message("Captures cleared")

        finally:
            cv2.destroyWindow(window_left)
            cv2.destroyWindow(window_right)
            cv2.destroyWindow(window_status)

        if best_result is None:
            raise CalibrationError("No valid stereo calibration achieved")

        return best_result

    def _run_stage2_zone_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
    ) -> Tuple[TouchZoneCalibrationResult, TouchZoneCalibrationResult]:
        """
        Run Stage 2: Touch zone calibration for both cameras.

        Args:
            left_camera: Left camera
            right_camera: Right camera

        Returns:
            Tuple of (left_result, right_result)

        Raises:
            CalibrationError: If calibration fails
        """
        self._notify_message(
            f"\nPlace white paper ({self.config.touchzone_config.paper_size_cm[0]}x"
            f"{self.config.touchzone_config.paper_size_cm[1]}cm) on the touch surface"
        )

        # Calibrate left camera
        self._notify_message("\nCalibrating LEFT camera touch zones...")
        left_result = self._calibrate_single_camera_zones(left_camera, "LEFT")

        # Calibrate right camera
        self._notify_message("\nCalibrating RIGHT camera touch zones...")
        right_result = self._calibrate_single_camera_zones(right_camera, "RIGHT")

        return left_result, right_result

    def _calibrate_single_camera_zones(
        self,
        camera: Any,
        camera_name: str,
    ) -> TouchZoneCalibrationResult:
        """
        Calibrate touch zones for a single camera with quality-first approach.

        Will retry until quality threshold is met.

        Args:
            camera: Camera instance
            camera_name: Camera identifier

        Returns:
            TouchZoneCalibrationResult

        Raises:
            CalibrationError: If calibration fails
        """
        best_result = None

        while True:
            try:
                result = self.touchzone_calibrator.run_interactive_calibration(
                    camera,
                    camera_name=camera_name,
                    auto_detect=True,
                )

                if result.quality_score >= self.config.target_zone_quality:
                    self._notify_message(
                        f"{camera_name}: Target quality achieved: {result.quality_score:.1%}"
                    )
                    return result

                if result.quality_score >= self.config.min_zone_quality:
                    if (
                        best_result is None
                        or result.quality_score > best_result.quality_score
                    ):
                        best_result = result

                    # Ask user if they want to retry
                    if self._ask_retry_calibration(camera, result):
                        continue
                    else:
                        return result

                # Quality too low, must retry
                self._notify_message(
                    f"{camera_name}: Quality too low ({result.quality_score:.1%}), "
                    f"minimum required: {self.config.min_zone_quality:.1%}"
                )

            except RuntimeError as e:
                if "cancelled" in str(e).lower():
                    if best_result is not None:
                        self._notify_message("Using best achieved calibration")
                        return best_result
                    raise CalibrationError(f"{camera_name} calibration cancelled")
                raise

        return best_result

    def _ask_retry_calibration(
        self,
        camera: Any,
        result: TouchZoneCalibrationResult,
    ) -> bool:
        """
        Ask user if they want to retry calibration.

        Args:
            camera: Camera instance
            result: Current calibration result

        Returns:
            True if user wants to retry, False to accept current result
        """
        return not self.touchzone_calibrator.verify_calibration(
            result,
            camera,
            display_time=self.config.auto_accept_timeout,
        )

    def _get_camera_frame(self, camera: Any) -> np.ndarray:
        """Get frame from camera object."""
        if hasattr(camera, "get_frame"):
            return camera.get_frame()
        elif hasattr(camera, "read"):
            ret, frame = camera.read()
            if not ret:
                raise CalibrationError("Failed to read from camera")
            return frame
        else:
            raise CalibrationError("Camera must have get_frame() or read() method")

    def _prepare_chessboard_display(
        self,
        frame: np.ndarray,
        detection_result: Optional[Tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """Prepare chessboard display frame."""
        if len(frame.shape) == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        if detection_result is not None:
            corners, quality = detection_result
            cols, rows = self.config.chessboard_config.pattern_size
            cv2.drawChessboardCorners(display, (cols, rows), corners, True)

            color = (0, 255, 0) if quality > 0.7 else (0, 255, 255)
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

    def _create_status_display(
        self,
        capture_count: int,
        current_quality: float,
        target_quality: float,
        min_quality: float,
    ) -> np.ndarray:
        """Create status display panel."""
        display = np.zeros((200, 400, 3), dtype=np.uint8)

        # Title
        cv2.putText(
            display,
            "Stage 1: Stereo Calibration",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Capture count
        cv2.putText(
            display,
            f"Captures: {capture_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Quality bar
        bar_x = 10
        bar_y = 100
        bar_width = 380
        bar_height = 30

        # Background
        cv2.rectangle(
            display,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1,
        )

        # Minimum threshold marker
        min_x = int(bar_x + bar_width * min_quality)
        cv2.line(display, (min_x, bar_y), (min_x, bar_y + bar_height), (0, 255, 255), 2)

        # Target threshold marker
        target_x = int(bar_x + bar_width * target_quality)
        cv2.line(
            display, (target_x, bar_y), (target_x, bar_y + bar_height), (0, 255, 0), 2
        )

        # Current quality bar
        if current_quality > 0:
            quality_width = int(bar_width * min(current_quality, 1.0))
            color = (
                (0, 255, 0)
                if current_quality >= target_quality
                else (0, 255, 255) if current_quality >= min_quality else (0, 0, 255)
            )
            cv2.rectangle(
                display,
                (bar_x, bar_y),
                (bar_x + quality_width, bar_y + bar_height),
                color,
                -1,
            )

        cv2.putText(
            display,
            f"Quality: {current_quality:.1%}",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Instructions
        cv2.putText(
            display,
            "C: Force calibrate | R: Reset | ESC: Cancel",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

        return display

    def save_calibration(
        self,
        result: FullCalibrationResult,
        path: str,
    ) -> None:
        """
        Save calibration result to file.

        Args:
            result: Calibration result to save
            path: Output file path (YAML format)
        """
        data = result.to_calibration_data()
        data.save(path)
        self._notify_message(f"Calibration saved to: {path}")

    def load_calibration(self, path: str) -> CalibrationData:
        """
        Load calibration data from file.

        Args:
            path: Path to calibration YAML file

        Returns:
            Loaded CalibrationData object
        """
        try:
            data = CalibrationData.load(path)
            self._notify_message(f"Calibration loaded from: {path}")
            return data
        except FileNotFoundError:
            raise CalibrationError(f"Calibration file not found: {path}")
        except CalibrationDataError as e:
            raise CalibrationError(f"Failed to load calibration: {e}")

    # ========== Legacy compatibility methods ==========

    def run_interactive_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
        zone_selector: Any = None,
        height_thresholds: Optional[List[float]] = None,
    ) -> CalibrationData:
        """
        Legacy method for backward compatibility.

        Runs the full two-stage calibration and returns CalibrationData.
        """
        result = self.run_full_calibration(left_camera, right_camera)
        return result.to_calibration_data()

    def calibrate_from_points(
        self,
        left_image_points: np.ndarray,
        right_image_points: np.ndarray,
        height_thresholds: Optional[List[float]] = None,
    ) -> CalibrationData:
        """
        Create calibration data from pre-selected corner points.

        This is a simplified calibration for testing or when chessboard
        calibration has already been done separately.

        Args:
            left_image_points: 4 corner points from left camera
            right_image_points: 4 corner points from right camera
            height_thresholds: Optional height thresholds

        Returns:
            CalibrationData
        """
        # Use touch zone calibrator for perspective transforms
        left_result = self.touchzone_calibrator.calibrate(
            left_image_points.astype(np.float32),
            image_size=self.image_size,
        )
        right_result = self.touchzone_calibrator.calibrate(
            right_image_points.astype(np.float32),
            image_size=self.image_size,
        )

        if height_thresholds is None:
            height_thresholds = self.DEFAULT_HEIGHT_THRESHOLDS.copy()

        # Calculate zone boundaries for legacy format
        zone_boundaries = self._calculate_zone_boundaries()

        return CalibrationData(
            timestamp=datetime.now(),
            camera_left_transform=left_result.perspective_transform,
            camera_right_transform=right_result.perspective_transform,
            zone_boundaries=zone_boundaries,
            height_thresholds=np.array(height_thresholds, dtype=np.float64),
            stereo_baseline=self.stereo_baseline,
            reference_board_size=self.board_physical_size,
            image_size=self.image_size,
            calibration_quality={
                "left_quality_score": left_result.quality_score,
                "right_quality_score": right_result.quality_score,
                "left_mean_error": left_result.reprojection_error,
                "right_mean_error": right_result.reprojection_error,
                "left_max_error": left_result.reprojection_error,
                "right_max_error": right_result.reprojection_error,
            },
            left_calibration_points=left_image_points.astype(np.float64),
            right_calibration_points=right_image_points.astype(np.float64),
        )

    def _calculate_zone_boundaries(self) -> Dict[str, Any]:
        """Calculate zone boundaries based on grid configuration."""
        cols, rows = self.zone_grid
        zone_width = self.board_physical_size[0] / cols
        zone_height = self.board_physical_size[1] / rows

        zones = []
        for row in range(rows):
            for col in range(cols):
                x_min = col * zone_width
                y_min = row * zone_height
                x_max = (col + 1) * zone_width
                y_max = (row + 1) * zone_height

                if row == 0:
                    zone_id = 2 * (cols - 1 - col) + 1
                else:
                    zone_id = 2 * (cols - 1 - col) + 2

                zones.append(
                    {
                        "id": zone_id,
                        "grid_row": row,
                        "grid_col": col,
                        "bounds": [x_min, y_min, x_max, y_max],
                        "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
                    }
                )

        zones.sort(key=lambda z: z["id"])

        return {
            "grid": list(self.zone_grid),
            "zone_size": [zone_width, zone_height],
            "board_size": list(self.board_physical_size),
            "zones": zones,
        }

    def validate_calibration(
        self,
        data: CalibrationData,
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate calibration data quality.

        Args:
            data: CalibrationData to validate

        Returns:
            Tuple of (is_valid, quality_score, list_of_issues)
        """
        issues = data.validate()
        quality = data.get_quality_score()
        is_valid = len(issues) == 0 and quality >= self.MIN_QUALITY_THRESHOLD
        return is_valid, quality, issues

    def calibrate_height_thresholds(
        self,
        get_hand_height: Callable[[], Optional[float]],
        target_heights: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Interactively calibrate height thresholds.

        Args:
            get_hand_height: Function that returns current hand height
            target_heights: Target heights for each level

        Returns:
            List of 6 calibrated height thresholds
        """
        if target_heights is None:
            target_heights = self.DEFAULT_HEIGHT_THRESHOLDS

        if len(target_heights) != 6:
            raise CalibrationError(
                f"target_heights must have 6 values, got {len(target_heights)}"
            )

        calibrated_thresholds = []

        for level, target in enumerate(target_heights):
            self._notify_message(
                f"Place hand at height level {level} ({target:.1f}cm) and hold steady..."
            )

            samples = []
            max_samples = 30
            stable_count = 0
            stable_threshold = 10

            for _ in range(max_samples):
                height = get_hand_height()
                if height is not None:
                    samples.append(height)

                    if len(samples) >= 5:
                        recent = samples[-5:]
                        if max(recent) - min(recent) < 2.0:
                            stable_count += 1
                            if stable_count >= stable_threshold:
                                break
                        else:
                            stable_count = 0

            if len(samples) < 5:
                raise CalibrationError(f"Not enough hand detections for level {level}")

            threshold = float(np.median(samples[-10:]))
            calibrated_thresholds.append(threshold)

            self._notify_message(f"Level {level} calibrated at {threshold:.1f}cm")

        for i in range(len(calibrated_thresholds) - 1):
            if calibrated_thresholds[i] >= calibrated_thresholds[i + 1]:
                avg = (calibrated_thresholds[i] + calibrated_thresholds[i + 1]) / 2
                calibrated_thresholds[i] = avg - 0.5
                calibrated_thresholds[i + 1] = avg + 0.5

        return calibrated_thresholds
