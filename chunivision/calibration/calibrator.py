"""
Calibration workflow coordinator for ChunIVision.

Manages the interactive calibration process for mapping camera views
to physical touch zones. Note that lens distortion correction is handled
automatically by the Oculus camera library using factory calibration parameters.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .calibration_data import CalibrationData, CalibrationDataError
from .transform_calculator import TransformCalculator
from .zone_selector import ZoneSelector
from ..utils.logger import Logger

logger = Logger.get_logger(__name__)


class CalibrationError(Exception):
    """Raised when calibration fails."""

    pass


class Calibrator:
    """
    Calibration workflow coordinator.

    Orchestrates the interactive calibration process for ChunIVision:
    1. User selects 4 corner points of calibration board for each camera
    2. Computes perspective transformation matrices (image -> physical coordinates)
    3. User calibrates height thresholds for 6 air sensor levels
    4. Validates and saves calibration data

    Note: Lens distortion correction is handled by Oculus camera library.
    This calibration is ONLY for establishing the mapping between
    camera image coordinates and physical touch zone coordinates.

    Typical usage:
        calibrator = Calibrator(
            board_physical_size=(44.0, 9.0),  # cm
            zone_grid=(16, 2)  # columns, rows
        )
        calibration_data = calibrator.run_interactive_calibration(
            left_camera, right_camera
        )
        calibrator.save_calibration(calibration_data, "calibration.yaml")
    """

    # Default height thresholds for 6 air sensor levels (cm above surface)
    DEFAULT_HEIGHT_THRESHOLDS = [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]

    # Minimum quality threshold for calibration to be accepted
    MIN_QUALITY_THRESHOLD = 0.7

    # Recommended quality threshold
    RECOMMENDED_QUALITY_THRESHOLD = 0.85

    def __init__(
        self,
        board_physical_size: Tuple[float, float] = (44.0, 9.0),
        zone_grid: Tuple[int, int] = (16, 2),
        image_size: Tuple[int, int] = (640, 480),
        stereo_baseline: float = 20.0,
    ):
        """
        Initialize calibrator.

        Args:
            board_physical_size: Physical size of calibration board (width, height) in cm
            zone_grid: Number of touch zones (columns, rows)
            image_size: Camera image resolution (width, height)
            stereo_baseline: Distance between cameras in cm
        """
        self.board_physical_size = board_physical_size
        self.zone_grid = zone_grid
        self.image_size = image_size
        self.stereo_baseline = stereo_baseline

        # Define physical corner positions of the calibration board
        # These correspond to the 4 corners in clockwise order from bottom-left
        self._world_points = np.array(
            [
                [0, 0],  # Bottom-left
                [board_physical_size[0], 0],  # Bottom-right
                [board_physical_size[0], board_physical_size[1]],  # Top-right
                [0, board_physical_size[1]],  # Top-left
            ],
            dtype=np.float32,
        )

        # Callbacks for UI events
        self._on_progress: Optional[Callable[[str, float], None]] = None
        self._on_message: Optional[Callable[[str], None]] = None

        logger.info(
            f"Calibrator initialized: board_size={board_physical_size}cm, "
            f"zone_grid={zone_grid}, image_size={image_size}, baseline={stereo_baseline}cm"
        )

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

    def _notify_progress(self, step: str, progress: float) -> None:
        """Notify progress callback if set."""
        if self._on_progress:
            self._on_progress(step, progress)

    def _notify_message(self, message: str) -> None:
        """Notify message callback if set."""
        if self._on_message:
            self._on_message(message)
        logger.info(message)

    def run_interactive_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
        zone_selector: Optional[ZoneSelector] = None,
        height_thresholds: Optional[List[float]] = None,
    ) -> CalibrationData:
        """
        Run interactive calibration workflow.

        Steps:
        1. Prompt user to place calibration board
        2. User selects 4 corner points for left camera
        3. User selects 4 corner points for right camera
        4. Calculate perspective transforms
        5. Set height thresholds (interactive or use provided values)
        6. Validate and return calibration data

        Args:
            left_camera: Left camera instance (must provide get_frame())
            right_camera: Right camera instance (must provide get_frame())
            zone_selector: Optional ZoneSelector instance (creates default if None)
            height_thresholds: Optional list of 6 height thresholds in cm
                              (uses default if not provided)

        Returns:
            CalibrationData with computed parameters

        Raises:
            CalibrationError: If calibration fails
        """
        self._notify_message("Starting interactive calibration workflow")
        self._notify_progress("Initializing", 0.0)

        if zone_selector is None:
            zone_selector = ZoneSelector("ChunIVision Calibration")

        owns_selector = zone_selector is not None

        try:
            # Step 1: Get frames from cameras (already lens-distortion-corrected by Oculus lib)
            self._notify_progress("Capturing frames", 0.1)
            self._notify_message("Capturing frames from cameras...")
            left_frame = self._get_camera_frame(left_camera, "left")
            right_frame = self._get_camera_frame(right_camera, "right")

            # Step 2: Select points for left camera
            self._notify_progress("Left camera calibration", 0.2)
            self._notify_message(
                "Please select 4 corners of calibration board (left camera)"
            )
            left_image_points = zone_selector.select_points(
                left_frame,
                num_points=4,
                instructions="Select 4 corners: bottom-left, bottom-right, top-right, top-left (clockwise)",
            )

            if left_image_points is None or len(left_image_points) != 4:
                raise CalibrationError("Failed to select 4 points for left camera")

            left_image_points = np.array(left_image_points, dtype=np.float32)

            # Step 3: Select points for right camera
            self._notify_progress("Right camera calibration", 0.4)
            self._notify_message(
                "Please select 4 corners of calibration board (right camera)"
            )
            right_image_points = zone_selector.select_points(
                right_frame,
                num_points=4,
                instructions="Select 4 corners: bottom-left, bottom-right, top-right, top-left (clockwise)",
            )

            if right_image_points is None or len(right_image_points) != 4:
                raise CalibrationError("Failed to select 4 points for right camera")

            right_image_points = np.array(right_image_points, dtype=np.float32)

            # Step 4: Calculate perspective transforms
            self._notify_progress("Computing transforms", 0.6)
            self._notify_message("Computing perspective transformations...")

            left_transform = TransformCalculator.calculate_perspective_transform(
                left_image_points, self._world_points
            )
            right_transform = TransformCalculator.calculate_perspective_transform(
                right_image_points, self._world_points
            )

            # Step 5: Validate transform quality
            self._notify_progress("Validating quality", 0.7)
            left_quality, left_mean_err, left_max_err = (
                TransformCalculator.validate_transform_quality(
                    left_transform, left_image_points, self._world_points
                )
            )
            right_quality, right_mean_err, right_max_err = (
                TransformCalculator.validate_transform_quality(
                    right_transform, right_image_points, self._world_points
                )
            )

            self._notify_message(
                f"Left camera transform quality: {left_quality:.1%} "
                f"(mean error: {left_mean_err:.2f}cm, max error: {left_max_err:.2f}cm)"
            )
            self._notify_message(
                f"Right camera transform quality: {right_quality:.1%} "
                f"(mean error: {right_mean_err:.2f}cm, max error: {right_max_err:.2f}cm)"
            )

            # Check minimum quality
            if left_quality < self.MIN_QUALITY_THRESHOLD:
                raise CalibrationError(
                    f"Left camera calibration quality too low: {left_quality:.1%} "
                    f"(minimum: {self.MIN_QUALITY_THRESHOLD:.1%})"
                )
            if right_quality < self.MIN_QUALITY_THRESHOLD:
                raise CalibrationError(
                    f"Right camera calibration quality too low: {right_quality:.1%} "
                    f"(minimum: {self.MIN_QUALITY_THRESHOLD:.1%})"
                )

            # Warn if below recommended
            if left_quality < self.RECOMMENDED_QUALITY_THRESHOLD:
                logger.warning(
                    f"Left camera quality {left_quality:.1%} is below recommended "
                    f"{self.RECOMMENDED_QUALITY_THRESHOLD:.1%}. Consider recalibrating."
                )
            if right_quality < self.RECOMMENDED_QUALITY_THRESHOLD:
                logger.warning(
                    f"Right camera quality {right_quality:.1%} is below recommended "
                    f"{self.RECOMMENDED_QUALITY_THRESHOLD:.1%}. Consider recalibrating."
                )

            # Step 6: Set height thresholds
            self._notify_progress("Setting height thresholds", 0.8)
            if height_thresholds is None:
                height_thresholds = self.DEFAULT_HEIGHT_THRESHOLDS.copy()
                self._notify_message(
                    f"Using default height thresholds: {height_thresholds}"
                )
            else:
                if len(height_thresholds) != 6:
                    raise CalibrationError(
                        f"height_thresholds must have 6 values, got {len(height_thresholds)}"
                    )
                self._notify_message(
                    f"Using provided height thresholds: {height_thresholds}"
                )

            # Step 7: Calculate zone boundaries
            self._notify_progress("Computing zone boundaries", 0.9)
            zone_boundaries = self._calculate_zone_boundaries()

            # Step 8: Create calibration data
            self._notify_progress("Finalizing", 0.95)
            calibration_quality = {
                "left_quality_score": float(left_quality),
                "right_quality_score": float(right_quality),
                "left_mean_error": float(left_mean_err),
                "right_mean_error": float(right_mean_err),
                "left_max_error": float(left_max_err),
                "right_max_error": float(right_max_err),
            }

            calibration_data = CalibrationData(
                timestamp=datetime.now(),
                camera_left_transform=left_transform,
                camera_right_transform=right_transform,
                zone_boundaries=zone_boundaries,
                height_thresholds=np.array(height_thresholds, dtype=np.float64),
                stereo_baseline=self.stereo_baseline,
                reference_board_size=self.board_physical_size,
                image_size=self.image_size,
                calibration_quality=calibration_quality,
                left_calibration_points=left_image_points.astype(np.float64),
                right_calibration_points=right_image_points.astype(np.float64),
            )

            # Validate the calibration data
            errors = calibration_data.validate()
            if errors:
                # Log warnings but don't fail - some validation errors may be acceptable
                for error in errors:
                    logger.warning(f"Calibration validation warning: {error}")

            self._notify_progress("Complete", 1.0)
            self._notify_message("Calibration completed successfully")
            return calibration_data

        except CalibrationError:
            raise
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise CalibrationError(f"Interactive calibration failed: {e}") from e
        finally:
            if owns_selector:
                zone_selector.close()

    def _get_camera_frame(self, camera: Any, camera_name: str) -> np.ndarray:
        """
        Get a frame from camera.

        Args:
            camera: Camera instance
            camera_name: Camera identifier for logging

        Returns:
            Frame as numpy array

        Raises:
            CalibrationError: If frame capture fails
        """
        try:
            if hasattr(camera, "get_frame"):
                frame = camera.get_frame()
            elif hasattr(camera, "read"):
                ret, frame = camera.read()
                if not ret:
                    raise CalibrationError(f"Failed to read from {camera_name} camera")
            else:
                raise CalibrationError(
                    f"{camera_name} camera does not have get_frame() or read() method"
                )

            if frame is None:
                raise CalibrationError(f"Got None frame from {camera_name} camera")

            return frame

        except Exception as e:
            raise CalibrationError(
                f"Failed to get frame from {camera_name} camera: {e}"
            ) from e

    def _calculate_zone_boundaries(self) -> Dict[str, Any]:
        """
        Calculate zone boundaries based on grid configuration.

        Returns:
            Dictionary containing zone boundary information
        """
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

                # Calculate zone ID based on ChunIVision numbering:
                # Bottom row (row 0): odd numbers from right (1,3,5...31)
                # Top row (row 1): even numbers from right (2,4,6...32)
                if row == 0:
                    zone_id = 2 * (cols - 1 - col) + 1  # 1,3,5...31 from right
                else:
                    zone_id = 2 * (cols - 1 - col) + 2  # 2,4,6...32 from right

                zones.append(
                    {
                        "id": zone_id,
                        "grid_row": row,
                        "grid_col": col,
                        "bounds": [x_min, y_min, x_max, y_max],
                        "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
                    }
                )

        # Sort by zone ID for consistency
        zones.sort(key=lambda z: z["id"])

        return {
            "grid": list(self.zone_grid),
            "zone_size": [zone_width, zone_height],
            "board_size": list(self.board_physical_size),
            "zones": zones,
        }

    def calibrate_from_points(
        self,
        left_image_points: np.ndarray,
        right_image_points: np.ndarray,
        height_thresholds: Optional[List[float]] = None,
    ) -> CalibrationData:
        """
        Create calibration data from pre-selected points (non-interactive).

        Useful for testing, batch calibration, or when points are known.

        Args:
            left_image_points: 4 corner points from left camera (4, 2)
            right_image_points: 4 corner points from right camera (4, 2)
            height_thresholds: Optional list of 6 height thresholds in cm

        Returns:
            CalibrationData with computed parameters

        Raises:
            CalibrationError: If calibration fails
        """
        logger.info("Creating calibration from provided points")

        try:
            # Validate input points
            left_image_points = np.array(left_image_points, dtype=np.float32)
            right_image_points = np.array(right_image_points, dtype=np.float32)

            if left_image_points.shape != (4, 2):
                raise CalibrationError(
                    f"left_image_points must be (4, 2), got {left_image_points.shape}"
                )
            if right_image_points.shape != (4, 2):
                raise CalibrationError(
                    f"right_image_points must be (4, 2), got {right_image_points.shape}"
                )

            # Calculate perspective transforms
            left_transform = TransformCalculator.calculate_perspective_transform(
                left_image_points, self._world_points
            )
            right_transform = TransformCalculator.calculate_perspective_transform(
                right_image_points, self._world_points
            )

            # Validate transform quality
            left_quality, left_mean_err, left_max_err = (
                TransformCalculator.validate_transform_quality(
                    left_transform, left_image_points, self._world_points
                )
            )
            right_quality, right_mean_err, right_max_err = (
                TransformCalculator.validate_transform_quality(
                    right_transform, right_image_points, self._world_points
                )
            )

            logger.info(
                f"Left camera quality: {left_quality:.1%} "
                f"(mean error: {left_mean_err:.2f}cm)"
            )
            logger.info(
                f"Right camera quality: {right_quality:.1%} "
                f"(mean error: {right_mean_err:.2f}cm)"
            )

            # Set height thresholds
            if height_thresholds is None:
                height_thresholds = self.DEFAULT_HEIGHT_THRESHOLDS.copy()

            # Create calibration data
            calibration_quality = {
                "left_quality_score": float(left_quality),
                "right_quality_score": float(right_quality),
                "left_mean_error": float(left_mean_err),
                "right_mean_error": float(right_mean_err),
                "left_max_error": float(left_max_err),
                "right_max_error": float(right_max_err),
            }

            calibration_data = CalibrationData(
                timestamp=datetime.now(),
                camera_left_transform=left_transform,
                camera_right_transform=right_transform,
                zone_boundaries=self._calculate_zone_boundaries(),
                height_thresholds=np.array(height_thresholds, dtype=np.float64),
                stereo_baseline=self.stereo_baseline,
                reference_board_size=self.board_physical_size,
                image_size=self.image_size,
                calibration_quality=calibration_quality,
                left_calibration_points=left_image_points.astype(np.float64),
                right_calibration_points=right_image_points.astype(np.float64),
            )

            return calibration_data

        except CalibrationError:
            raise
        except Exception as e:
            logger.error(f"Calibration from points failed: {e}")
            raise CalibrationError(f"Calibration from points failed: {e}") from e

    def save_calibration(self, data: CalibrationData, path: str) -> None:
        """
        Save calibration data to file.

        Args:
            data: CalibrationData to save
            path: Output file path (YAML format)

        Raises:
            CalibrationError: If save fails
        """
        try:
            data.save(path)
            logger.info(f"Calibration saved to: {path}")
        except CalibrationDataError as e:
            raise CalibrationError(f"Failed to save calibration: {e}") from e
        except Exception as e:
            raise CalibrationError(f"Failed to save calibration: {e}") from e

    def load_calibration(self, path: str) -> CalibrationData:
        """
        Load calibration data from file.

        Args:
            path: Path to calibration YAML file

        Returns:
            Loaded CalibrationData object

        Raises:
            CalibrationError: If load fails or file not found
        """
        try:
            data = CalibrationData.load(path)
            logger.info(f"Calibration loaded from: {path}")

            # Validate loaded data
            errors = data.validate()
            if errors:
                for error in errors:
                    logger.warning(f"Loaded calibration has issue: {error}")

            return data

        except FileNotFoundError:
            raise CalibrationError(f"Calibration file not found: {path}")
        except CalibrationDataError as e:
            raise CalibrationError(f"Failed to load calibration: {e}") from e
        except Exception as e:
            raise CalibrationError(f"Failed to load calibration: {e}") from e

    def validate_calibration(
        self, data: CalibrationData
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate calibration quality.

        Args:
            data: CalibrationData to validate

        Returns:
            Tuple of (is_valid, quality_score, list_of_issues)
        """
        issues = []

        # Check basic validation
        errors = data.validate()
        issues.extend(errors)

        # Get quality score
        quality_score = data.get_quality_score()

        # Check quality thresholds
        left_quality = data.calibration_quality.get("left_quality_score", 0.0)
        right_quality = data.calibration_quality.get("right_quality_score", 0.0)

        if left_quality < self.MIN_QUALITY_THRESHOLD:
            issues.append(
                f"Left camera quality {left_quality:.1%} below minimum {self.MIN_QUALITY_THRESHOLD:.1%}"
            )
        if right_quality < self.MIN_QUALITY_THRESHOLD:
            issues.append(
                f"Right camera quality {right_quality:.1%} below minimum {self.MIN_QUALITY_THRESHOLD:.1%}"
            )

        # Check if transforms are reasonable (not too extreme)
        for name, transform in [
            ("left", data.camera_left_transform),
            ("right", data.camera_right_transform),
        ]:
            try:
                decomp = TransformCalculator.decompose_transform(transform)
                scale_x, scale_y = decomp["scale"]

                # Check for extreme scaling (>20x or <0.05x)
                if scale_x > 20 or scale_x < 0.05:
                    issues.append(f"{name} camera has extreme X scale: {scale_x:.2f}")
                if scale_y > 20 or scale_y < 0.05:
                    issues.append(f"{name} camera has extreme Y scale: {scale_y:.2f}")

                # Check for extreme perspective
                p1, p2 = decomp["perspective"]
                if abs(p1) > 0.1 or abs(p2) > 0.1:
                    issues.append(
                        f"{name} camera has extreme perspective: ({p1:.4f}, {p2:.4f})"
                    )

            except Exception as e:
                issues.append(f"Failed to analyze {name} transform: {e}")

        is_valid = len(issues) == 0 and quality_score >= self.MIN_QUALITY_THRESHOLD

        return is_valid, quality_score, issues

    def calibrate_height_thresholds(
        self,
        get_hand_height: Callable[[], Optional[float]],
        target_heights: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Interactively calibrate height thresholds.

        Prompts user to place hand at each height level and records the
        actual Z coordinates.

        Args:
            get_hand_height: Function that returns current hand height in cm,
                            or None if no hand detected
            target_heights: Optional target heights for each level.
                           If not provided, uses default air sensor heights.

        Returns:
            List of 6 calibrated height thresholds

        Raises:
            CalibrationError: If calibration fails
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

            # Collect multiple samples for stability
            samples = []
            max_samples = 30
            stable_count = 0
            stable_threshold = 10  # Need 10 stable readings

            for _ in range(max_samples):
                height = get_hand_height()
                if height is not None:
                    samples.append(height)

                    # Check stability (last 5 samples within 2cm)
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

            # Use median of stable samples
            threshold = float(np.median(samples[-10:]))
            calibrated_thresholds.append(threshold)

            self._notify_message(f"Level {level} calibrated at {threshold:.1f}cm")

        # Ensure thresholds are in ascending order
        for i in range(len(calibrated_thresholds) - 1):
            if calibrated_thresholds[i] >= calibrated_thresholds[i + 1]:
                # Force ascending by averaging
                avg = (calibrated_thresholds[i] + calibrated_thresholds[i + 1]) / 2
                calibrated_thresholds[i] = avg - 0.5
                calibrated_thresholds[i + 1] = avg + 0.5

        return calibrated_thresholds
