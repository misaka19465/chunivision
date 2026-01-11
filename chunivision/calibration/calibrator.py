"""
Calibration workflow coordinator for ChunIVision.

Manages the interactive calibration process for mapping camera views
to physical touch zones. Note that lens distortion correction is handled
automatically by the Oculus camera library using factory calibration parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .calibration_data import CalibrationData
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
    """

    def __init__(
        self,
        board_physical_size: Tuple[float, float] = (44.0, 9.0),
        zone_grid: Tuple[int, int] = (16, 2),
        image_size: Tuple[int, int] = (640, 480),
    ):
        """
        Initialize calibrator.

        Args:
            board_physical_size: Physical size of calibration board (width, height) in cm
            zone_grid: Number of touch zones (columns, rows)
            image_size: Camera image resolution (width, height)
        """
        self.board_physical_size = board_physical_size
        self.zone_grid = zone_grid
        self.image_size = image_size

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

        logger.info(
            f"Calibrator initialized: board_size={board_physical_size}cm, "
            f"zone_grid={zone_grid}, image_size={image_size}"
        )

    def run_interactive_calibration(
        self,
        left_camera: Any,
        right_camera: Any,
        zone_selector: Optional[ZoneSelector] = None,
    ) -> CalibrationData:
        """
        Run interactive calibration workflow.

        Steps:
        1. Prompt user to place calibration board
        2. User selects 4 corner points for left camera
        3. User selects 4 corner points for right camera
        4. Calculate perspective transforms
        5. User calibrates height thresholds
        6. Return calibration data

        Args:
            left_camera: Left camera instance (must provide get_frame())
            right_camera: Right camera instance (must provide get_frame())
            zone_selector: Optional ZoneSelector instance (creates default if None)

        Returns:
            CalibrationData with computed parameters

        Raises:
            CalibrationError: If calibration fails
        """
        logger.info("Starting interactive calibration workflow")

        if zone_selector is None:
            zone_selector = ZoneSelector("ChunIVision Calibration")

        try:
            # Step 1: Get frames from cameras (already lens-distortion-corrected by Oculus lib)
            logger.info("Capturing frames from cameras...")
            left_frame = self._get_camera_frame(left_camera, "left")
            right_frame = self._get_camera_frame(right_camera, "right")

            # Step 2: Select points for left camera
            logger.info("Please select 4 corners of calibration board (left camera)")
            left_image_points = zone_selector.select_points(
                left_frame,
                num_points=4,
                instructions="Select 4 corners: bottom-left, bottom-right, top-right, top-left (clockwise)",
            )

            if left_image_points is None or len(left_image_points) != 4:
                raise CalibrationError("Failed to select 4 points for left camera")

            left_image_points = np.array(left_image_points, dtype=np.float32)

            # Step 3: Select points for right camera
            logger.info("Please select 4 corners of calibration board (right camera)")
            right_image_points = zone_selector.select_points(
                right_frame,
                num_points=4,
                instructions="Select 4 corners: bottom-left, bottom-right, top-right, top-left (clockwise)",
            )

            if right_image_points is None or len(right_image_points) != 4:
                raise CalibrationError("Failed to select 4 points for right camera")

            right_image_points = np.array(right_image_points, dtype=np.float32)

            # Step 4: Calculate perspective transforms
            logger.info("Computing perspective transformations...")
            left_transform = TransformCalculator.calculate_perspective_transform(
                left_image_points, self._world_points
            )
            right_transform = TransformCalculator.calculate_perspective_transform(
                right_image_points, self._world_points
            )

            # Step 5: Validate transform quality
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
                f"Left camera transform quality: {left_quality:.3f} "
                f"(mean error: {left_mean_err:.2f}cm, max error: {left_max_err:.2f}cm)"
            )
            logger.info(
                f"Right camera transform quality: {right_quality:.3f} "
                f"(mean error: {right_mean_err:.2f}cm, max error: {right_max_err:.2f}cm)"
            )

            if left_quality < 0.85 or right_quality < 0.85:
                logger.warning(
                    "Transform quality is below recommended threshold (0.85). "
                    "Consider recalibrating for better accuracy."
                )

            # Step 6: Height calibration (would need UI implementation)
            logger.info(
                "Height calibration would be performed here (UI not implemented)"
            )
            height_thresholds = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]  # Placeholder

            # Step 7: Create calibration data
            calibration_data = CalibrationData(
                camera_left_transform=left_transform,
                camera_right_transform=right_transform,
                zone_boundaries=self._calculate_zone_boundaries(),
                height_thresholds=height_thresholds,
                stereo_baseline=20.0,  # Placeholder, should be measured
                reference_board_size=self.board_physical_size,
            )

            logger.info("Calibration completed successfully")
            return calibration_data

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise CalibrationError(f"Interactive calibration failed: {e}") from e
        finally:
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

                zones.append(
                    {
                        "id": row * cols + col,
                        "bounds": [x_min, y_min, x_max, y_max],
                        "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
                    }
                )

        return {
            "grid": self.zone_grid,
            "zone_size": [zone_width, zone_height],
            "zones": zones,
        }
