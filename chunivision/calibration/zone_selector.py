"""
Interactive zone calibration UI for ChunIVision.

Provides a graphical interface for selecting calibration points on camera frames,
with visual feedback, zoom functionality, and preview capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import Logger

logger = Logger.get_logger(__name__)


@dataclass
class SelectionState:
    """
    Internal state for point selection.

    Attributes:
        points: List of selected points as (x, y) coordinates
        hover_point: Current mouse position for visual feedback
        zoom_enabled: Whether zoom mode is active
        zoom_center: Center point of zoom region
        zoom_factor: Current zoom magnification
    """

    points: List[np.ndarray]
    hover_point: Optional[Tuple[int, int]]
    zoom_enabled: bool
    zoom_center: Optional[Tuple[int, int]]
    zoom_factor: float

    @classmethod
    def create_empty(cls) -> "SelectionState":
        """Create empty selection state."""
        return cls(
            points=[],
            hover_point=None,
            zoom_enabled=False,
            zoom_center=None,
            zoom_factor=2.0,
        )


class ZoneSelector:
    """
    GUI for user to select calibration points on camera view.

    Provides an interactive OpenCV-based interface for selecting
    calibration points with visual feedback and precision tools.

    Features:
        - Click to select points with visual markers
        - Real-time hover feedback
        - Zoom mode for precise selection (toggle with 'z')
        - Undo last point (press 'u')
        - Clear all points (press 'c')
        - Preview transformed view with zone overlay
        - Instructions displayed on screen

    Key Bindings:
        - Left Click: Select point
        - Right Click: Undo last point
        - 'z': Toggle zoom mode
        - 'u': Undo last point
        - 'c': Clear all points
        - 'Enter/Space': Confirm selection
        - 'Escape': Cancel selection

    Example:
        selector = ZoneSelector("Calibration")
        points = selector.select_points(frame, num_points=4,
                                        instructions="Select 4 corners")
        if points:
            # Process selected points
            pass
        selector.close()
    """

    # Colors (BGR format)
    COLOR_POINT = (0, 255, 0)  # Green for selected points
    COLOR_POINT_NUMBER = (255, 255, 255)  # White for point numbers
    COLOR_HOVER = (255, 255, 0)  # Cyan for hover position
    COLOR_LINE = (0, 200, 0)  # Dark green for connecting lines
    COLOR_CROSSHAIR = (200, 200, 200)  # Gray for crosshairs
    COLOR_INSTRUCTION = (255, 255, 255)  # White for instructions
    COLOR_ZOOM_BORDER = (0, 0, 255)  # Red for zoom window border
    COLOR_ZONE_GRID = (255, 0, 255)  # Magenta for zone grid overlay
    COLOR_ZONE_TEXT = (255, 255, 0)  # Cyan for zone numbers

    # UI constants
    POINT_RADIUS = 6
    HOVER_RADIUS = 4
    LINE_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    ZOOM_WINDOW_SIZE = 200
    DEFAULT_ZOOM_FACTOR = 2.0
    MAX_ZOOM_FACTOR = 8.0
    MIN_ZOOM_FACTOR = 1.5

    def __init__(
        self,
        window_title: str = "Zone Calibration",
        zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    ):
        """
        Initialize selector with window configuration.

        Args:
            window_title: Title for the OpenCV window
            zoom_factor: Initial zoom magnification (1.5-8.0)
        """
        self.window_title = window_title
        self._state = SelectionState.create_empty()
        self._state.zoom_factor = max(
            self.MIN_ZOOM_FACTOR, min(self.MAX_ZOOM_FACTOR, zoom_factor)
        )

        self._frame: Optional[np.ndarray] = None
        self._display_frame: Optional[np.ndarray] = None
        self._num_points: int = 4
        self._instructions: str = ""
        self._selection_complete: bool = False
        self._selection_cancelled: bool = False
        self._window_created: bool = False
        self._on_point_selected: Optional[Callable[[np.ndarray], None]] = None

        logger.debug(f"ZoneSelector initialized: window_title='{window_title}'")

    def _create_window(self) -> None:
        """Create OpenCV window with mouse callback."""
        if not self._window_created:
            cv2.namedWindow(self.window_title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_title, self._mouse_callback)
            self._window_created = True
            logger.debug(f"Created window: {self.window_title}")

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: object
    ) -> None:
        """
        Handle mouse events for point selection.

        Args:
            event: OpenCV mouse event type
            x: X coordinate of mouse position
            y: Y coordinate of mouse position
            flags: Additional flags
            param: User parameter (unused)
        """
        if self._frame is None:
            return

        # Get actual coordinates (accounting for zoom if active)
        actual_x, actual_y = self._get_actual_coordinates(x, y)

        # Handle hover
        if event == cv2.EVENT_MOUSEMOVE:
            self._state.hover_point = (actual_x, actual_y)

        # Handle left click - select point
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(self._state.points) < self._num_points:
                point = np.array([actual_x, actual_y], dtype=np.float64)
                self._state.points.append(point)

                if self._on_point_selected:
                    self._on_point_selected(point)

                logger.debug(
                    f"Point {len(self._state.points)} selected at ({actual_x}, {actual_y})"
                )

                if len(self._state.points) >= self._num_points:
                    self._selection_complete = True
                    logger.info(f"All {self._num_points} points selected")

        # Handle right click - undo last point
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._undo_last_point()

        # Update zoom center on middle click
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._state.zoom_center = (actual_x, actual_y)
            self._state.zoom_enabled = True
            logger.debug(f"Zoom center set at ({actual_x}, {actual_y})")

    def _get_actual_coordinates(
        self, display_x: int, display_y: int
    ) -> Tuple[int, int]:
        """
        Convert display coordinates to actual frame coordinates.

        Accounts for zoom if enabled.

        Args:
            display_x: X coordinate in display window
            display_y: Y coordinate in display window

        Returns:
            Tuple of (actual_x, actual_y) in frame coordinates
        """
        # If not zoomed, coordinates are the same
        # Zoom is handled in preview, not by transforming the main view
        return display_x, display_y

    def _undo_last_point(self) -> None:
        """Remove the last selected point."""
        if self._state.points:
            removed = self._state.points.pop()
            self._selection_complete = False
            logger.debug(
                f"Undid point at ({removed[0]:.1f}, {removed[1]:.1f}), "
                f"{len(self._state.points)} points remaining"
            )

    def _clear_all_points(self) -> None:
        """Clear all selected points."""
        self._state.points.clear()
        self._selection_complete = False
        logger.debug("Cleared all points")

    def _toggle_zoom(self) -> None:
        """Toggle zoom mode on/off."""
        self._state.zoom_enabled = not self._state.zoom_enabled
        if self._state.zoom_enabled and self._state.hover_point:
            self._state.zoom_center = self._state.hover_point
        logger.debug(
            f"Zoom mode: {'enabled' if self._state.zoom_enabled else 'disabled'}"
        )

    def _increase_zoom(self) -> None:
        """Increase zoom factor."""
        self._state.zoom_factor = min(
            self.MAX_ZOOM_FACTOR, self._state.zoom_factor + 0.5
        )
        logger.debug(f"Zoom factor: {self._state.zoom_factor:.1f}x")

    def _decrease_zoom(self) -> None:
        """Decrease zoom factor."""
        self._state.zoom_factor = max(
            self.MIN_ZOOM_FACTOR, self._state.zoom_factor - 0.5
        )
        logger.debug(f"Zoom factor: {self._state.zoom_factor:.1f}x")

    def _render_frame(self) -> np.ndarray:
        """
        Render the frame with all overlays.

        Returns:
            Rendered frame with points, lines, and instructions
        """
        if self._frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Create display copy (convert to BGR if grayscale)
        if len(self._frame.shape) == 2:
            display = cv2.cvtColor(self._frame, cv2.COLOR_GRAY2BGR)
        else:
            display = self._frame.copy()

        # Draw instructions at top
        self._draw_instructions(display)

        # Draw status info at bottom
        self._draw_status(display)

        # Draw crosshairs at hover position
        if self._state.hover_point:
            self._draw_crosshairs(display, self._state.hover_point)

        # Draw connecting lines between points
        if len(self._state.points) >= 2:
            self._draw_point_lines(display)

        # Draw selected points with numbers
        for i, point in enumerate(self._state.points):
            self._draw_point(display, point, i + 1)

        # Draw hover indicator
        if self._state.hover_point and len(self._state.points) < self._num_points:
            self._draw_hover(display, self._state.hover_point)

        # Draw zoom window if enabled
        if self._state.zoom_enabled and self._state.hover_point:
            self._draw_zoom_window(display)

        return display

    def _draw_instructions(self, frame: np.ndarray) -> None:
        """Draw instructions at top of frame."""
        # Background bar for better readability
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (40, 40, 40), -1)

        # Main instruction text
        instruction_text = (
            self._instructions if self._instructions else "Click to select points"
        )
        cv2.putText(
            frame,
            instruction_text,
            (10, 20),
            self.FONT,
            self.FONT_SCALE,
            self.COLOR_INSTRUCTION,
            self.FONT_THICKNESS,
        )

        # Control hints
        hints = "Z:Zoom | U/RClick:Undo | C:Clear | Enter:Confirm | Esc:Cancel"
        cv2.putText(
            frame,
            hints,
            (10, 40),
            self.FONT,
            self.FONT_SCALE * 0.8,
            (180, 180, 180),
            self.FONT_THICKNESS,
        )

    def _draw_status(self, frame: np.ndarray) -> None:
        """Draw status information at bottom of frame."""
        h = frame.shape[0]

        # Background bar
        cv2.rectangle(frame, (0, h - 30), (frame.shape[1], h), (40, 40, 40), -1)

        # Point counter
        status = f"Points: {len(self._state.points)}/{self._num_points}"
        if self._state.zoom_enabled:
            status += f" | Zoom: {self._state.zoom_factor:.1f}x (+/-)"

        cv2.putText(
            frame,
            status,
            (10, h - 10),
            self.FONT,
            self.FONT_SCALE,
            self.COLOR_INSTRUCTION,
            self.FONT_THICKNESS,
        )

        # Hover coordinates
        if self._state.hover_point:
            x, y = self._state.hover_point
            coord_text = f"({x}, {y})"
            text_size = cv2.getTextSize(
                coord_text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
            )[0]
            cv2.putText(
                frame,
                coord_text,
                (frame.shape[1] - text_size[0] - 10, h - 10),
                self.FONT,
                self.FONT_SCALE,
                self.COLOR_HOVER,
                self.FONT_THICKNESS,
            )

    def _draw_crosshairs(self, frame: np.ndarray, point: Tuple[int, int]) -> None:
        """Draw crosshairs at the given point."""
        x, y = point
        h, w = frame.shape[:2]

        # Vertical line (avoiding status bars)
        cv2.line(frame, (x, 51), (x, h - 31), self.COLOR_CROSSHAIR, 1)
        # Horizontal line
        cv2.line(frame, (0, y), (w, y), self.COLOR_CROSSHAIR, 1)

    def _draw_point(self, frame: np.ndarray, point: np.ndarray, number: int) -> None:
        """Draw a selected point with its number."""
        x, y = int(point[0]), int(point[1])

        # Outer circle (border)
        cv2.circle(frame, (x, y), self.POINT_RADIUS + 2, (0, 0, 0), -1)
        # Inner circle (filled)
        cv2.circle(frame, (x, y), self.POINT_RADIUS, self.COLOR_POINT, -1)

        # Point number
        text = str(number)
        text_size = cv2.getTextSize(
            text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
        )[0]
        text_x = x - text_size[0] // 2
        text_y = y - self.POINT_RADIUS - 5
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            self.FONT,
            self.FONT_SCALE,
            self.COLOR_POINT_NUMBER,
            self.FONT_THICKNESS,
        )

    def _draw_hover(self, frame: np.ndarray, point: Tuple[int, int]) -> None:
        """Draw hover indicator at mouse position."""
        x, y = point
        cv2.circle(frame, (x, y), self.HOVER_RADIUS, self.COLOR_HOVER, 2)

    def _draw_point_lines(self, frame: np.ndarray) -> None:
        """Draw lines connecting selected points."""
        points = self._state.points
        n = len(points)

        for i in range(n):
            p1 = (int(points[i][0]), int(points[i][1]))
            p2 = (int(points[(i + 1) % n][0]), int(points[(i + 1) % n][1]))

            # Draw line if we have more than current connection
            # or if all points selected (close the polygon)
            if i < n - 1 or self._selection_complete:
                cv2.line(frame, p1, p2, self.COLOR_LINE, self.LINE_THICKNESS)

    def _draw_zoom_window(self, frame: np.ndarray) -> None:
        """Draw magnified zoom window."""
        if self._frame is None or self._state.hover_point is None:
            return

        h, w = frame.shape[:2]
        cx, cy = self._state.hover_point
        zoom = self._state.zoom_factor
        half_size = int(self.ZOOM_WINDOW_SIZE / (2 * zoom))

        # Source region in original frame
        src_x1 = max(0, cx - half_size)
        src_y1 = max(0, cy - half_size)
        src_x2 = min(w, cx + half_size)
        src_y2 = min(h, cy + half_size)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return

        # Get source region from original frame
        if len(self._frame.shape) == 2:
            src_region = cv2.cvtColor(
                self._frame[src_y1:src_y2, src_x1:src_x2],
                cv2.COLOR_GRAY2BGR,
            )
        else:
            src_region = self._frame[src_y1:src_y2, src_x1:src_x2].copy()

        # Resize to zoom window size
        zoomed = cv2.resize(
            src_region,
            (self.ZOOM_WINDOW_SIZE, self.ZOOM_WINDOW_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # Draw crosshair in center of zoom window
        zc = self.ZOOM_WINDOW_SIZE // 2
        cv2.line(zoomed, (zc, 0), (zc, self.ZOOM_WINDOW_SIZE), self.COLOR_CROSSHAIR, 1)
        cv2.line(zoomed, (0, zc), (self.ZOOM_WINDOW_SIZE, zc), self.COLOR_CROSSHAIR, 1)

        # Add border
        cv2.rectangle(
            zoomed,
            (0, 0),
            (self.ZOOM_WINDOW_SIZE - 1, self.ZOOM_WINDOW_SIZE - 1),
            self.COLOR_ZOOM_BORDER,
            2,
        )

        # Position zoom window in corner opposite to mouse
        margin = 10
        if cx < w // 2:
            zoom_x = w - self.ZOOM_WINDOW_SIZE - margin
        else:
            zoom_x = margin

        if cy < h // 2:
            zoom_y = h - self.ZOOM_WINDOW_SIZE - margin - 30  # Account for status bar
        else:
            zoom_y = 51 + margin  # Account for instruction bar

        # Draw zoom window on frame
        frame[
            zoom_y : zoom_y + self.ZOOM_WINDOW_SIZE,
            zoom_x : zoom_x + self.ZOOM_WINDOW_SIZE,
        ] = zoomed

    def select_points(
        self,
        frame: np.ndarray,
        num_points: int = 4,
        instructions: str = "",
        initial_points: Optional[List[np.ndarray]] = None,
        on_point_selected: Optional[Callable[[np.ndarray], None]] = None,
    ) -> List[np.ndarray]:
        """
        Display frame and let user select points by clicking.

        Creates an interactive window where the user can click to select
        calibration points. The window includes visual feedback, zoom
        capability, and keyboard controls.

        Args:
            frame: Camera frame to display (grayscale or BGR)
            num_points: Number of points to select (default 4)
            instructions: Text to display to user
            initial_points: Optional list of pre-selected points
            on_point_selected: Optional callback when a point is selected

        Returns:
            List of selected points as numpy arrays of shape (2,) with [x, y]
            Returns empty list if selection was cancelled

        Example:
            selector = ZoneSelector()
            corners = selector.select_points(
                frame,
                num_points=4,
                instructions="Click the 4 corners of the calibration board"
            )
        """
        self._frame = frame.copy()
        self._num_points = num_points
        self._instructions = instructions
        self._on_point_selected = on_point_selected
        self._selection_complete = False
        self._selection_cancelled = False

        # Initialize state
        self._state = SelectionState.create_empty()
        if initial_points:
            self._state.points = [p.copy() for p in initial_points]
            if len(self._state.points) >= num_points:
                self._selection_complete = True

        logger.info(
            f"Starting point selection: num_points={num_points}, "
            f"initial_points={len(self._state.points)}"
        )

        self._create_window()

        # Main selection loop
        while not self._selection_cancelled:
            display = self._render_frame()
            cv2.imshow(self.window_title, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Escape - cancel
                self._selection_cancelled = True
                logger.info("Selection cancelled by user")
                break

            elif key in (13, 32):  # Enter or Space - confirm
                if self._selection_complete or len(self._state.points) > 0:
                    logger.info(
                        f"Selection confirmed: {len(self._state.points)} points"
                    )
                    break

            elif key == ord("u") or key == ord("U"):  # Undo
                self._undo_last_point()

            elif key == ord("c") or key == ord("C"):  # Clear
                self._clear_all_points()

            elif key == ord("z") or key == ord("Z"):  # Toggle zoom
                self._toggle_zoom()

            elif key == ord("+") or key == ord("="):  # Increase zoom
                self._increase_zoom()

            elif key == ord("-") or key == ord("_"):  # Decrease zoom
                self._decrease_zoom()

        if self._selection_cancelled:
            return []

        return self._state.points.copy()

    def show_preview(
        self,
        frame: np.ndarray,
        transform: Optional[np.ndarray] = None,
        zone_grid: Optional[np.ndarray] = None,
        wait_key: bool = True,
        window_title: Optional[str] = None,
    ) -> int:
        """
        Show preview of calibrated view with zone overlay.

        Displays the frame with optional perspective transform applied
        and zone grid overlay for verification.

        Args:
            frame: Camera frame to display
            transform: Optional 3x3 perspective transform matrix
            zone_grid: Optional zone boundary array of shape (num_zones, 4, 2)
            wait_key: If True, wait for key press before returning
            window_title: Optional custom window title

        Returns:
            Key code pressed (if wait_key=True) or -1

        Example:
            selector.show_preview(frame, transform=calibration.transform,
                                  zone_grid=zone_boundaries)
        """
        # Apply perspective transform if provided
        if transform is not None:
            h, w = frame.shape[:2]
            preview = cv2.warpPerspective(frame, transform, (w, h))
        else:
            preview = frame.copy()

        # Convert to BGR if grayscale
        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # Draw zone grid overlay
        if zone_grid is not None:
            self._draw_zone_grid(preview, zone_grid)

        # Display
        title = window_title or f"{self.window_title} - Preview"
        cv2.imshow(title, preview)

        if wait_key:
            key = cv2.waitKey(0) & 0xFF
            return key
        else:
            return cv2.waitKey(1) & 0xFF

    def _draw_zone_grid(self, frame: np.ndarray, zone_grid: np.ndarray) -> None:
        """
        Draw zone boundaries on frame.

        Args:
            frame: Frame to draw on (modified in place)
            zone_grid: Zone boundaries array of shape (num_zones, 4, 2)
        """
        for i, zone in enumerate(zone_grid):
            # Draw zone boundary polygon
            pts = zone.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, self.COLOR_ZONE_GRID, 1)

            # Draw zone number at center
            center = zone.mean(axis=0).astype(int)
            zone_id = i + 1  # 1-indexed
            text = str(zone_id)
            text_size = cv2.getTextSize(
                text, self.FONT, self.FONT_SCALE * 0.8, self.FONT_THICKNESS
            )[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                self.FONT,
                self.FONT_SCALE * 0.8,
                self.COLOR_ZONE_TEXT,
                self.FONT_THICKNESS,
            )

    def get_selected_points(self) -> List[np.ndarray]:
        """
        Get currently selected points.

        Returns:
            Copy of list of selected points
        """
        return self._state.points.copy()

    def set_zoom_factor(self, factor: float) -> None:
        """
        Set zoom magnification factor.

        Args:
            factor: Zoom factor (1.5-8.0)
        """
        self._state.zoom_factor = max(
            self.MIN_ZOOM_FACTOR, min(self.MAX_ZOOM_FACTOR, factor)
        )

    def close(self) -> None:
        """Close the selector window and release resources."""
        if self._window_created:
            cv2.destroyWindow(self.window_title)
            self._window_created = False
            logger.debug(f"Closed window: {self.window_title}")

    def __enter__(self) -> "ZoneSelector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - close window."""
        self.close()
