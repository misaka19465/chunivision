"""
Touch zone calibration using white paper reference.

This module handles the second stage of calibration where a white paper
matching the game's touch area size is used to establish the mapping
between camera coordinates and physical touch zones.

The calibration process:
1. User places white paper (same size as touch area) on the play surface
2. System detects the white paper edges
3. User confirms or adjusts the detected corners
4. Perspective transform is computed to map camera view to touch zones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import Logger
from .lens_distortion import LensDistortion
from .transform_calculator import TransformCalculator
from .zone_selector import ZoneSelector

logger = Logger.get_logger(__name__)


@dataclass
class TouchZoneConfig:
    """
    Configuration for touch zone calibration.

    Attributes:
        zone_grid: Number of touch zones (columns, rows)
        zone_width_cm: Width of each zone in cm
        zone_height_cm: Height of each zone in cm
        paper_size_cm: Physical size of calibration paper (width, height) in cm
        detection_method: Method for paper detection ("contour", "corners", "manual")
        min_paper_area_ratio: Minimum paper area as ratio of frame area
        max_paper_area_ratio: Maximum paper area as ratio of frame area
        canny_threshold1: Canny edge detection threshold 1
        canny_threshold2: Canny edge detection threshold 2
        blur_kernel_size: Gaussian blur kernel size for preprocessing
    """

    zone_grid: Tuple[int, int] = (16, 2)  # 16 columns, 2 rows
    zone_width_cm: float = 2.75  # 27.5mm per zone
    zone_height_cm: float = 4.5  # 45mm per zone
    paper_size_cm: Tuple[float, float] = (44.0, 9.0)  # Total touch area size
    detection_method: str = "contour"
    min_paper_area_ratio: float = 0.05  # Paper must be at least 5% of frame
    max_paper_area_ratio: float = 0.9  # Paper can't be more than 90% of frame
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    blur_kernel_size: int = 5

    @property
    def total_width_cm(self) -> float:
        """Get total touch area width in cm."""
        return self.zone_grid[0] * self.zone_width_cm

    @property
    def total_height_cm(self) -> float:
        """Get total touch area height in cm."""
        return self.zone_grid[1] * self.zone_height_cm


@dataclass
class ZoneBoundary:
    """
    Boundary information for a single touch zone.

    Attributes:
        zone_id: Zone identifier (1-32)
        grid_row: Row in grid (0=bottom, 1=top)
        grid_col: Column in grid (0-15, 0=leftmost)
        corners: Corner points in physical coordinates (4x2 array)
        center: Center point in physical coordinates
    """

    zone_id: int
    grid_row: int
    grid_col: int
    corners: np.ndarray  # (4, 2) corners in physical coords
    center: np.ndarray  # (2,) center in physical coords

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside this zone."""
        # Simple bounding box check
        x, y = point
        x_min = self.corners[:, 0].min()
        x_max = self.corners[:, 0].max()
        y_min = self.corners[:, 1].min()
        y_max = self.corners[:, 1].max()
        return x_min <= x <= x_max and y_min <= y <= y_max


@dataclass
class TouchZoneCalibrationResult:
    """
    Result of touch zone calibration for a camera.

    Attributes:
        perspective_transform: 3x3 homography matrix (image -> physical)
        inverse_transform: 3x3 inverse homography (physical -> image)
        image_corners: Detected/selected corner points in image coords (4x2)
        physical_corners: Corresponding physical corner points (4x2)
        zone_boundaries: List of zone boundary information
        reprojection_error: Calibration reprojection error
        quality_score: Overall quality score (0-1)
        image_size: Image size (width, height)
    """

    perspective_transform: np.ndarray
    inverse_transform: np.ndarray
    image_corners: np.ndarray
    physical_corners: np.ndarray
    zone_boundaries: List[ZoneBoundary]
    reprojection_error: float
    quality_score: float
    image_size: Tuple[int, int]

    def transform_point_to_physical(self, image_point: np.ndarray) -> np.ndarray:
        """Transform a point from image coordinates to physical coordinates."""
        return TransformCalculator.apply_transform(
            image_point, self.perspective_transform
        )

    def transform_point_to_image(self, physical_point: np.ndarray) -> np.ndarray:
        """Transform a point from physical coordinates to image coordinates."""
        return TransformCalculator.apply_transform(
            physical_point, self.inverse_transform
        )

    def get_zone_at_point(self, physical_point: np.ndarray) -> Optional[int]:
        """
        Get the zone ID at a physical coordinate.

        Args:
            physical_point: (x, y) in physical coordinates (cm)

        Returns:
            Zone ID (1-32) or None if outside all zones
        """
        for zone in self.zone_boundaries:
            if zone.contains_point(physical_point):
                return zone.zone_id
        return None


class TouchZoneCalibrator:
    """
    Touch zone calibration using white paper reference.

    This calibrator establishes the mapping between camera image coordinates
    and the physical touch zone grid. It uses a white paper (same size as
    the game's touch area) as a calibration reference.

    The process:
    1. Detect the white paper in the camera view
    2. Identify the 4 corners of the paper
    3. Compute perspective transform to map image to physical coordinates
    4. Generate zone boundary information

    Example:
        config = TouchZoneConfig(zone_grid=(16, 2), paper_size_cm=(44.0, 9.0))
        calibrator = TouchZoneCalibrator(config)

        # Auto-detect paper
        corners = calibrator.detect_paper(frame)

        # Or manual selection
        corners = calibrator.select_corners_interactive(frame)

        # Compute calibration
        result = calibrator.calibrate(corners, image_size=(640, 480))
    """

    def __init__(
        self,
        config: Optional[TouchZoneConfig] = None,
        lens_distortion: Optional[LensDistortion] = None,
    ):
        """
        Initialize touch zone calibrator.

        Args:
            config: Touch zone configuration
            lens_distortion: Lens distortion handler for pre-undistortion
        """
        self.config = config or TouchZoneConfig()
        self.lens_distortion = lens_distortion

        # Physical corner positions (same for all calibrations)
        # Order: bottom-left, bottom-right, top-right, top-left
        self._physical_corners = np.array(
            [
                [0, 0],
                [self.config.paper_size_cm[0], 0],
                [self.config.paper_size_cm[0], self.config.paper_size_cm[1]],
                [0, self.config.paper_size_cm[1]],
            ],
            dtype=np.float32,
        )

        # Callbacks
        self._on_progress: Optional[Callable[[str, float], None]] = None
        self._on_message: Optional[Callable[[str], None]] = None

        logger.info(
            f"TouchZoneCalibrator initialized: "
            f"grid={self.config.zone_grid}, paper_size={self.config.paper_size_cm}cm"
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

    def preprocess_frame(
        self,
        frame: np.ndarray,
        undistort: bool = True,
    ) -> np.ndarray:
        """
        Preprocess frame for paper detection.

        Args:
            frame: Input frame
            undistort: Whether to apply lens distortion correction

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Undistort if handler available
        if undistort and self.lens_distortion is not None:
            gray = self.lens_distortion.undistort_frame(gray)

        return gray

    def detect_paper(
        self,
        frame: np.ndarray,
        undistort: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Automatically detect white paper in the frame.

        Uses contour detection to find the largest bright rectangular
        region in the frame.

        Args:
            frame: Input frame
            undistort: Whether to undistort the frame first

        Returns:
            4 corner points as (4, 2) array in clockwise order starting
            from bottom-left, or None if detection fails
        """
        gray = self.preprocess_frame(frame, undistort)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            gray,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            0,
        )

        # Threshold to find white regions
        # Use adaptive threshold for varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -5,  # Negative value to favor bright regions
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            logger.debug("No contours found")
            return None

        # Filter contours by area
        frame_area = gray.shape[0] * gray.shape[1]
        min_area = frame_area * self.config.min_paper_area_ratio
        max_area = frame_area * self.config.max_paper_area_ratio

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if it's roughly quadrilateral
                if 4 <= len(approx) <= 6:
                    valid_contours.append((contour, approx, area))

        if not valid_contours:
            logger.debug("No valid quadrilateral contours found")
            return None

        # Select the largest valid contour
        valid_contours.sort(key=lambda x: x[2], reverse=True)
        _, best_approx, _ = valid_contours[0]

        # Get exactly 4 corners using convex hull and corner detection
        corners = self._extract_corners(best_approx)

        if corners is None or len(corners) != 4:
            logger.debug(
                f"Could not extract 4 corners, got {len(corners) if corners is not None else 0}"
            )
            return None

        # Order corners: bottom-left, bottom-right, top-right, top-left
        corners = self._order_corners(corners)

        return corners.astype(np.float32)

    def _extract_corners(self, approx: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract exactly 4 corners from an approximate polygon.

        Args:
            approx: Approximate polygon points

        Returns:
            4 corner points or None
        """
        if len(approx) == 4:
            return approx.reshape(-1, 2)

        # If more points, use convex hull and find 4 extreme points
        hull = cv2.convexHull(approx)
        if len(hull) < 4:
            return None

        # Find the 4 corners by looking for extreme points
        points = hull.reshape(-1, 2)

        # Get bounding rect corners
        x, y, w, h = cv2.boundingRect(points)

        # Find points closest to each corner of bounding rect
        target_corners = np.array(
            [
                [x, y + h],  # Bottom-left
                [x + w, y + h],  # Bottom-right
                [x + w, y],  # Top-right
                [x, y],  # Top-left
            ]
        )

        corners = []
        for target in target_corners:
            distances = np.linalg.norm(points - target, axis=1)
            closest_idx = np.argmin(distances)
            corners.append(points[closest_idx])

        return np.array(corners)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners: bottom-left, bottom-right, top-right, top-left.

        Args:
            corners: 4 corner points in arbitrary order

        Returns:
            Ordered corners (4, 2)
        """
        # Sort by sum of coordinates (top-left has smallest, bottom-right has largest)
        rect = np.zeros((4, 2), dtype=np.float32)

        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()

        # Top-left: smallest sum
        # Bottom-right: largest sum
        # Top-right: smallest difference
        # Bottom-left: largest difference

        tl_idx = np.argmin(s)
        br_idx = np.argmax(s)
        tr_idx = np.argmin(diff)
        bl_idx = np.argmax(diff)

        rect[0] = corners[bl_idx]  # Bottom-left
        rect[1] = corners[br_idx]  # Bottom-right
        rect[2] = corners[tr_idx]  # Top-right
        rect[3] = corners[tl_idx]  # Top-left

        return rect

    def select_corners_interactive(
        self,
        frame: np.ndarray,
        window_title: str = "Touch Zone Calibration",
        initial_corners: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Interactive corner selection using ZoneSelector.

        Args:
            frame: Camera frame
            window_title: Window title
            initial_corners: Pre-detected corners to refine

        Returns:
            4 corner points or None if cancelled
        """
        selector = ZoneSelector(window_title)

        instructions = (
            "Select 4 corners of white paper: "
            "bottom-left, bottom-right, top-right, top-left"
        )

        try:
            points = selector.select_points(
                frame,
                num_points=4,
                instructions=instructions,
                initial_points=(
                    [p for p in initial_corners]
                    if initial_corners is not None
                    else None
                ),
            )

            if len(points) != 4:
                return None

            return np.array(points, dtype=np.float32)

        finally:
            selector.close()

    def calibrate(
        self,
        image_corners: np.ndarray,
        image_size: Tuple[int, int],
    ) -> TouchZoneCalibrationResult:
        """
        Compute touch zone calibration from corner points.

        Args:
            image_corners: 4 corner points in image coordinates (4, 2)
                          Order: bottom-left, bottom-right, top-right, top-left
            image_size: Image size (width, height)

        Returns:
            TouchZoneCalibrationResult with computed parameters

        Raises:
            ValueError: If corners are invalid
        """
        if image_corners.shape != (4, 2):
            raise ValueError(f"Expected 4 corners, got shape {image_corners.shape}")

        # Compute perspective transform
        perspective_transform = TransformCalculator.calculate_perspective_transform(
            image_corners,
            self._physical_corners,
        )

        # Compute inverse transform
        inverse_transform = TransformCalculator.calculate_inverse_transform(
            perspective_transform
        )

        # Validate transform quality
        quality, mean_error, max_error = TransformCalculator.validate_transform_quality(
            perspective_transform,
            image_corners,
            self._physical_corners,
        )

        # Generate zone boundaries
        zone_boundaries = self._generate_zone_boundaries()

        self._notify_message(
            f"Touch zone calibration: quality={quality:.1%}, "
            f"mean_error={mean_error:.3f}cm, max_error={max_error:.3f}cm"
        )

        return TouchZoneCalibrationResult(
            perspective_transform=perspective_transform,
            inverse_transform=inverse_transform,
            image_corners=image_corners,
            physical_corners=self._physical_corners.copy(),
            zone_boundaries=zone_boundaries,
            reprojection_error=mean_error,
            quality_score=quality,
            image_size=image_size,
        )

    def _generate_zone_boundaries(self) -> List[ZoneBoundary]:
        """
        Generate zone boundary information.

        Returns:
            List of ZoneBoundary for all 32 zones
        """
        cols, rows = self.config.zone_grid
        zone_width = self.config.zone_width_cm
        zone_height = self.config.zone_height_cm

        boundaries = []

        for row in range(rows):
            for col in range(cols):
                # Calculate corner positions
                x0 = col * zone_width
                y0 = row * zone_height
                x1 = (col + 1) * zone_width
                y1 = (row + 1) * zone_height

                corners = np.array(
                    [
                        [x0, y0],  # Bottom-left
                        [x1, y0],  # Bottom-right
                        [x1, y1],  # Top-right
                        [x0, y1],  # Top-left
                    ],
                    dtype=np.float32,
                )

                center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)

                # Calculate zone ID based on game's numbering convention:
                # Bottom row (row 0): odd numbers from right (1,3,5...31)
                # Top row (row 1): even numbers from right (2,4,6...32)
                if row == 0:
                    zone_id = 2 * (cols - 1 - col) + 1  # Odd numbers
                else:
                    zone_id = 2 * (cols - 1 - col) + 2  # Even numbers

                boundaries.append(
                    ZoneBoundary(
                        zone_id=zone_id,
                        grid_row=row,
                        grid_col=col,
                        corners=corners,
                        center=center,
                    )
                )

        # Sort by zone ID
        boundaries.sort(key=lambda z: z.zone_id)

        return boundaries

    def run_interactive_calibration(
        self,
        camera: Any,
        camera_name: str = "camera",
        auto_detect: bool = True,
    ) -> TouchZoneCalibrationResult:
        """
        Run interactive touch zone calibration.

        Args:
            camera: Camera instance with get_frame() or read()
            camera_name: Camera identifier for display
            auto_detect: Whether to try auto-detection first

        Returns:
            TouchZoneCalibrationResult

        Raises:
            RuntimeError: If calibration fails or is cancelled
        """
        self._notify_message(f"Starting touch zone calibration for {camera_name}")
        self._notify_message(
            f"Place white paper ({self.config.paper_size_cm[0]}x{self.config.paper_size_cm[1]}cm) "
            f"on the touch surface"
        )

        window_name = f"Touch Zone Calibration - {camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        detected_corners = None
        confirmed_corners = None
        calibration_result = None

        try:
            while True:
                # Get frame
                frame = self._get_camera_frame(camera)
                gray = self.preprocess_frame(frame, undistort=True)

                # Auto-detect paper
                if auto_detect and detected_corners is None:
                    detected_corners = self.detect_paper(frame, undistort=False)

                # Prepare display
                display = self._prepare_display(
                    frame, detected_corners, confirmed_corners
                )

                # Add instructions
                self._add_instructions(display)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == 27:  # ESC - cancel
                    raise RuntimeError("Calibration cancelled by user")

                elif key == ord(" "):  # Space - confirm detection
                    if detected_corners is not None:
                        confirmed_corners = detected_corners.copy()
                        self._notify_message(
                            "Corners confirmed, computing calibration..."
                        )

                elif key == ord("m"):  # M - manual selection
                    cv2.destroyWindow(window_name)
                    confirmed_corners = self.select_corners_interactive(
                        frame,
                        f"Manual Selection - {camera_name}",
                        initial_corners=detected_corners,
                    )
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

                    if confirmed_corners is None:
                        self._notify_message(
                            "Manual selection cancelled, continuing..."
                        )
                    else:
                        self._notify_message("Manual corners selected")

                elif key == ord("r"):  # R - reset
                    detected_corners = None
                    confirmed_corners = None
                    self._notify_message("Reset detection")

                elif key == ord("c") or key == 13:  # C or Enter - calibrate
                    if confirmed_corners is not None:
                        image_size = (frame.shape[1], frame.shape[0])
                        calibration_result = self.calibrate(
                            confirmed_corners, image_size
                        )
                        break

        finally:
            cv2.destroyWindow(window_name)

        if calibration_result is None:
            raise RuntimeError("Calibration not completed")

        return calibration_result

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
        detected_corners: Optional[np.ndarray],
        confirmed_corners: Optional[np.ndarray],
    ) -> np.ndarray:
        """Prepare display frame with overlays."""
        if len(frame.shape) == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        # Draw detected corners (yellow)
        if detected_corners is not None and confirmed_corners is None:
            self._draw_corners(display, detected_corners, (0, 255, 255), "Detected")

        # Draw confirmed corners (green)
        if confirmed_corners is not None:
            self._draw_corners(display, confirmed_corners, (0, 255, 0), "Confirmed")

            # Draw zone grid preview
            self._draw_zone_preview(display, confirmed_corners)

        return display

    def _draw_corners(
        self,
        display: np.ndarray,
        corners: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw corner markers and polygon."""
        pts = corners.astype(np.int32)

        # Draw polygon
        cv2.polylines(display, [pts], True, color, 2)

        # Draw corner points with numbers
        for i, (x, y) in enumerate(pts):
            cv2.circle(display, (x, y), 6, color, -1)
            cv2.circle(display, (x, y), 8, (0, 0, 0), 2)
            cv2.putText(
                display,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Label
        cv2.putText(
            display,
            label,
            (int(corners[:, 0].mean()), int(corners[:, 1].mean())),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    def _draw_zone_preview(
        self,
        display: np.ndarray,
        corners: np.ndarray,
    ) -> None:
        """Draw zone grid preview on the detected paper area."""
        cols, rows = self.config.zone_grid

        # Create perspective transform from physical to image
        transform = cv2.getPerspectiveTransform(
            self._physical_corners,
            corners.astype(np.float32),
        )

        # Draw zone lines
        zone_width = self.config.zone_width_cm
        zone_height = self.config.zone_height_cm

        # Vertical lines
        for col in range(cols + 1):
            x = col * zone_width
            pt1 = cv2.perspectiveTransform(
                np.array([[[x, 0]]], dtype=np.float32),
                transform,
            )[0, 0]
            pt2 = cv2.perspectiveTransform(
                np.array([[[x, self.config.paper_size_cm[1]]]], dtype=np.float32),
                transform,
            )[0, 0]
            cv2.line(
                display,
                tuple(pt1.astype(int)),
                tuple(pt2.astype(int)),
                (255, 0, 255),
                1,
            )

        # Horizontal lines
        for row in range(rows + 1):
            y = row * zone_height
            pt1 = cv2.perspectiveTransform(
                np.array([[[0, y]]], dtype=np.float32),
                transform,
            )[0, 0]
            pt2 = cv2.perspectiveTransform(
                np.array([[[self.config.paper_size_cm[0], y]]], dtype=np.float32),
                transform,
            )[0, 0]
            cv2.line(
                display,
                tuple(pt1.astype(int)),
                tuple(pt2.astype(int)),
                (255, 0, 255),
                1,
            )

    def _add_instructions(self, display: np.ndarray) -> None:
        """Add instruction overlay to display."""
        h = display.shape[0]

        # Background bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 60), (40, 40, 40), -1)

        # Instructions
        cv2.putText(
            display,
            "Touch Zone Calibration - Position white paper matching touch area size",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display,
            "SPACE: Confirm detection | M: Manual select | R: Reset | C/Enter: Calibrate | ESC: Cancel",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    def verify_calibration(
        self,
        result: TouchZoneCalibrationResult,
        camera: Any,
        display_time: float = 5.0,
    ) -> bool:
        """
        Display calibration verification visualization.

        Shows the zone grid overlaid on the camera view in real-time.

        Args:
            result: Calibration result to verify
            camera: Camera instance
            display_time: Time to display verification

        Returns:
            True if user confirms, False otherwise
        """
        import time

        window_name = "Calibration Verification"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        start_time = time.time()
        confirmed = False

        try:
            while True:
                elapsed = time.time() - start_time

                frame = self._get_camera_frame(camera)
                display = self._prepare_verification_display(frame, result)

                # Add countdown
                remaining = max(0, display_time - elapsed)
                cv2.putText(
                    display,
                    f"Auto-confirm in {remaining:.1f}s (Press Y to confirm, N to redo)",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("y") or key == ord("Y"):
                    confirmed = True
                    break
                elif key == ord("n") or key == ord("N"):
                    confirmed = False
                    break
                elif elapsed >= display_time:
                    confirmed = True
                    break

        finally:
            cv2.destroyWindow(window_name)

        return confirmed

    def _prepare_verification_display(
        self,
        frame: np.ndarray,
        result: TouchZoneCalibrationResult,
    ) -> np.ndarray:
        """Prepare verification display with zone overlay."""
        if len(frame.shape) == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        # Transform zone boundaries to image coordinates
        for zone in result.zone_boundaries:
            # Transform corners to image coordinates
            img_corners = cv2.perspectiveTransform(
                zone.corners.reshape(1, -1, 2),
                result.inverse_transform,
            )[0]

            # Draw zone boundary
            pts = img_corners.astype(np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 1)

            # Draw zone number at center
            img_center = cv2.perspectiveTransform(
                zone.center.reshape(1, 1, 2),
                result.inverse_transform,
            )[0, 0]

            cv2.putText(
                display,
                str(zone.zone_id),
                tuple(img_center.astype(int) - [5, -5]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 255),
                1,
            )

        # Add quality info
        cv2.putText(
            display,
            f"Quality: {result.quality_score:.1%} | Error: {result.reprojection_error:.3f}cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return display
