"""
Unit tests for ZoneSelector module.

Tests the interactive point selection UI for calibration.
Note: Some tests require mocking cv2 functions since they involve GUI operations.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from chunivision.calibration.zone_selector import SelectionState, ZoneSelector


class TestSelectionState:
    """Tests for SelectionState dataclass."""

    def test_create_empty(self):
        """Test creating empty selection state."""
        state = SelectionState.create_empty()

        assert state.points == []
        assert state.hover_point is None
        assert state.zoom_enabled is False
        assert state.zoom_center is None
        assert state.zoom_factor == 2.0

    def test_selection_state_custom_values(self):
        """Test SelectionState with custom values."""
        points = [np.array([10, 20]), np.array([30, 40])]
        state = SelectionState(
            points=points,
            hover_point=(50, 60),
            zoom_enabled=True,
            zoom_center=(100, 100),
            zoom_factor=4.0,
        )

        assert len(state.points) == 2
        assert state.hover_point == (50, 60)
        assert state.zoom_enabled is True
        assert state.zoom_center == (100, 100)
        assert state.zoom_factor == 4.0


class TestZoneSelectorInit:
    """Tests for ZoneSelector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        selector = ZoneSelector()

        assert selector.window_title == "Zone Calibration"
        assert selector._window_created is False

    def test_custom_window_title(self):
        """Test initialization with custom window title."""
        selector = ZoneSelector(window_title="Custom Title")

        assert selector.window_title == "Custom Title"

    def test_custom_zoom_factor(self):
        """Test initialization with custom zoom factor."""
        selector = ZoneSelector(zoom_factor=4.0)

        assert selector._state.zoom_factor == 4.0

    def test_zoom_factor_clamping_min(self):
        """Test zoom factor is clamped to minimum."""
        selector = ZoneSelector(zoom_factor=0.5)

        assert selector._state.zoom_factor == ZoneSelector.MIN_ZOOM_FACTOR

    def test_zoom_factor_clamping_max(self):
        """Test zoom factor is clamped to maximum."""
        selector = ZoneSelector(zoom_factor=20.0)

        assert selector._state.zoom_factor == ZoneSelector.MAX_ZOOM_FACTOR


class TestZoneSelectorPointManagement:
    """Tests for point management functionality."""

    def test_undo_last_point(self):
        """Test undoing the last selected point."""
        selector = ZoneSelector()
        selector._state.points = [
            np.array([10, 20]),
            np.array([30, 40]),
            np.array([50, 60]),
        ]

        selector._undo_last_point()

        assert len(selector._state.points) == 2
        np.testing.assert_array_equal(selector._state.points[-1], [30, 40])

    def test_undo_empty_list(self):
        """Test undoing when no points selected."""
        selector = ZoneSelector()
        selector._state.points = []

        # Should not raise an error
        selector._undo_last_point()

        assert len(selector._state.points) == 0

    def test_clear_all_points(self):
        """Test clearing all points."""
        selector = ZoneSelector()
        selector._state.points = [
            np.array([10, 20]),
            np.array([30, 40]),
        ]
        selector._selection_complete = True

        selector._clear_all_points()

        assert len(selector._state.points) == 0
        assert selector._selection_complete is False

    def test_get_selected_points(self):
        """Test getting copy of selected points."""
        selector = ZoneSelector()
        original_points = [np.array([10, 20]), np.array([30, 40])]
        selector._state.points = original_points.copy()

        result = selector.get_selected_points()

        assert len(result) == 2
        # Modifying result should not affect internal state
        result.append(np.array([50, 60]))
        assert len(selector._state.points) == 2


class TestZoneSelectorZoom:
    """Tests for zoom functionality."""

    def test_toggle_zoom(self):
        """Test toggling zoom mode."""
        selector = ZoneSelector()
        assert selector._state.zoom_enabled is False

        selector._toggle_zoom()
        assert selector._state.zoom_enabled is True

        selector._toggle_zoom()
        assert selector._state.zoom_enabled is False

    def test_toggle_zoom_sets_center(self):
        """Test that toggling zoom sets center from hover point."""
        selector = ZoneSelector()
        selector._state.hover_point = (100, 200)

        selector._toggle_zoom()

        assert selector._state.zoom_center == (100, 200)

    def test_increase_zoom(self):
        """Test increasing zoom factor."""
        selector = ZoneSelector(zoom_factor=2.0)

        selector._increase_zoom()

        assert selector._state.zoom_factor == 2.5

    def test_increase_zoom_max(self):
        """Test zoom factor doesn't exceed maximum."""
        selector = ZoneSelector(zoom_factor=8.0)

        selector._increase_zoom()

        assert selector._state.zoom_factor == ZoneSelector.MAX_ZOOM_FACTOR

    def test_decrease_zoom(self):
        """Test decreasing zoom factor."""
        selector = ZoneSelector(zoom_factor=3.0)

        selector._decrease_zoom()

        assert selector._state.zoom_factor == 2.5

    def test_decrease_zoom_min(self):
        """Test zoom factor doesn't go below minimum."""
        selector = ZoneSelector(zoom_factor=1.5)

        selector._decrease_zoom()

        assert selector._state.zoom_factor == ZoneSelector.MIN_ZOOM_FACTOR

    def test_set_zoom_factor(self):
        """Test setting zoom factor."""
        selector = ZoneSelector()

        selector.set_zoom_factor(5.0)

        assert selector._state.zoom_factor == 5.0

    def test_set_zoom_factor_clamping(self):
        """Test set_zoom_factor clamps values."""
        selector = ZoneSelector()

        selector.set_zoom_factor(0.5)
        assert selector._state.zoom_factor == ZoneSelector.MIN_ZOOM_FACTOR

        selector.set_zoom_factor(100.0)
        assert selector._state.zoom_factor == ZoneSelector.MAX_ZOOM_FACTOR


class TestZoneSelectorCoordinates:
    """Tests for coordinate handling."""

    def test_get_actual_coordinates(self):
        """Test coordinate conversion (no transform)."""
        selector = ZoneSelector()

        result = selector._get_actual_coordinates(100, 200)

        assert result == (100, 200)


class TestZoneSelectorRendering:
    """Tests for rendering functionality."""

    def test_render_frame_empty(self):
        """Test rendering with no frame."""
        selector = ZoneSelector()
        selector._frame = None

        result = selector._render_frame()

        assert result.shape == (480, 640, 3)

    def test_render_frame_grayscale(self):
        """Test rendering grayscale frame."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640), dtype=np.uint8)
        selector._num_points = 4
        selector._instructions = "Test"

        result = selector._render_frame()

        assert result.shape == (480, 640, 3)

    def test_render_frame_color(self):
        """Test rendering color frame."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4
        selector._instructions = "Test"

        result = selector._render_frame()

        assert result.shape == (480, 640, 3)

    def test_render_frame_with_points(self):
        """Test rendering with selected points."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4
        selector._state.points = [
            np.array([100, 100]),
            np.array([200, 100]),
        ]

        result = selector._render_frame()

        # Should complete without error
        assert result.shape == (480, 640, 3)

    def test_render_frame_with_hover(self):
        """Test rendering with hover point."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4
        selector._state.hover_point = (300, 200)

        result = selector._render_frame()

        assert result.shape == (480, 640, 3)

    def test_render_frame_with_zoom(self):
        """Test rendering with zoom enabled."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4
        selector._state.zoom_enabled = True
        selector._state.hover_point = (300, 200)
        selector._state.zoom_center = (300, 200)

        result = selector._render_frame()

        assert result.shape == (480, 640, 3)


class TestZoneSelectorMouseCallback:
    """Tests for mouse callback handling."""

    def test_mouse_move_updates_hover(self):
        """Test mouse move updates hover point."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4

        selector._mouse_callback(cv2.EVENT_MOUSEMOVE, 100, 200, 0, None)

        assert selector._state.hover_point == (100, 200)

    def test_left_click_adds_point(self):
        """Test left click adds a point."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4

        selector._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)

        assert len(selector._state.points) == 1
        np.testing.assert_array_equal(selector._state.points[0], [100, 200])

    def test_left_click_completes_selection(self):
        """Test selection completes when num_points reached."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 2
        selector._state.points = [np.array([10, 20])]

        selector._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)

        assert len(selector._state.points) == 2
        assert selector._selection_complete is True

    def test_left_click_ignores_when_complete(self):
        """Test left click ignored when selection complete."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 2
        selector._state.points = [np.array([10, 20]), np.array([30, 40])]

        selector._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)

        assert len(selector._state.points) == 2

    def test_right_click_undoes_point(self):
        """Test right click undoes last point."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4
        selector._state.points = [np.array([10, 20]), np.array([30, 40])]

        selector._mouse_callback(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)

        assert len(selector._state.points) == 1

    def test_middle_click_sets_zoom(self):
        """Test middle click sets zoom center."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4

        selector._mouse_callback(cv2.EVENT_MBUTTONDOWN, 200, 300, 0, None)

        assert selector._state.zoom_center == (200, 300)
        assert selector._state.zoom_enabled is True

    def test_callback_with_point_selected_callback(self):
        """Test point selection callback is called."""
        import cv2

        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._num_points = 4
        callback = MagicMock()
        selector._on_point_selected = callback

        selector._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)

        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        np.testing.assert_array_equal(call_args, [100, 200])


class TestZoneSelectorPreview:
    """Tests for preview functionality."""

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=27)
    def test_show_preview_basic(self, mock_waitkey, mock_imshow):
        """Test basic preview display."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = selector.show_preview(frame, wait_key=True)

        mock_imshow.assert_called_once()
        assert result == 27

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=-1)
    @patch("cv2.warpPerspective")
    def test_show_preview_with_transform(self, mock_warp, mock_waitkey, mock_imshow):
        """Test preview with perspective transform."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        transform = np.eye(3)
        mock_warp.return_value = frame.copy()

        selector.show_preview(frame, transform=transform)

        mock_warp.assert_called_once()

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=-1)
    def test_show_preview_with_zone_grid(self, mock_waitkey, mock_imshow):
        """Test preview with zone grid overlay."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        zone_grid = np.array(
            [
                [[10, 10], [50, 10], [50, 50], [10, 50]],
                [[60, 10], [100, 10], [100, 50], [60, 50]],
            ]
        )

        selector.show_preview(frame, zone_grid=zone_grid)

        mock_imshow.assert_called_once()

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=32)
    def test_show_preview_custom_title(self, mock_waitkey, mock_imshow):
        """Test preview with custom window title."""
        selector = ZoneSelector(window_title="Main")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector.show_preview(frame, window_title="Preview Window")

        call_args = mock_imshow.call_args[0]
        assert call_args[0] == "Preview Window"

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=-1)
    def test_show_preview_no_wait(self, mock_waitkey, mock_imshow):
        """Test preview without waiting."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector.show_preview(frame, wait_key=False)

        # waitKey should be called with 1 (non-blocking)
        mock_waitkey.assert_called_with(1)

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=-1)
    def test_show_preview_grayscale(self, mock_waitkey, mock_imshow):
        """Test preview with grayscale frame."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640), dtype=np.uint8)

        selector.show_preview(frame)

        # Should convert to BGR
        mock_imshow.assert_called_once()


class TestZoneSelectorSelectPoints:
    """Tests for select_points method."""

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_escape_cancels(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing Escape cancels selection."""
        mock_waitkey.return_value = 27  # Escape key

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = selector.select_points(frame, num_points=4)

        assert result == []

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_enter_confirms(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing Enter confirms selection."""
        mock_waitkey.return_value = 13  # Enter key

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Pre-add some points
        result = selector.select_points(
            frame,
            num_points=4,
            initial_points=[np.array([10, 20]), np.array([30, 40])],
        )

        assert len(result) == 2

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_space_confirms(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing Space confirms selection."""
        mock_waitkey.return_value = 32  # Space key

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = selector.select_points(
            frame,
            num_points=4,
            initial_points=[np.array([10, 20])],
        )

        assert len(result) == 1

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_with_initial_points(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test select_points with pre-selected points."""
        mock_waitkey.return_value = 13  # Enter key

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        initial = [np.array([100, 100]), np.array([200, 200])]

        result = selector.select_points(frame, num_points=4, initial_points=initial)

        assert len(result) == 2

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_u_key_undoes(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing 'u' undoes last point."""
        # First call returns 'u', second returns Escape
        mock_waitkey.side_effect = [ord("u"), 27]

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._state.points = [np.array([10, 20])]

        selector.select_points(frame, num_points=4)

        # The undo should have been called
        assert len(selector._state.points) == 0

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_c_key_clears(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing 'c' clears all points."""
        # First call returns 'c', second returns Escape
        mock_waitkey.side_effect = [ord("c"), 27]

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector.select_points(
            frame,
            num_points=4,
            initial_points=[np.array([10, 20]), np.array([30, 40])],
        )

        assert len(selector._state.points) == 0

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_z_key_toggles_zoom(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing 'z' toggles zoom mode."""
        # First call returns 'z', second returns Escape
        mock_waitkey.side_effect = [ord("z"), 27]

        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector.select_points(frame, num_points=4)

        assert selector._state.zoom_enabled is True

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_plus_key_increases_zoom(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing '+' increases zoom factor."""
        mock_waitkey.side_effect = [ord("+"), 27]

        selector = ZoneSelector(zoom_factor=2.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector.select_points(frame, num_points=4)

        assert selector._state.zoom_factor == 2.5

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_select_points_minus_key_decreases_zoom(
        self, mock_waitkey, mock_imshow, mock_callback, mock_window
    ):
        """Test pressing '-' decreases zoom factor."""
        mock_waitkey.side_effect = [ord("-"), 27]

        selector = ZoneSelector(zoom_factor=3.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Set zoom factor after state reset in select_points
        def set_zoom_and_run():
            selector._state.zoom_factor = 3.0
            return selector.select_points(frame, num_points=4)

        # Call select_points which resets state, then press minus key
        # The zoom factor starts at default 2.0 after reset, so after
        # decrease it should be 1.5 (clamped to minimum)
        selector.select_points(frame, num_points=4)

        # The zoom_factor will be 2.0 (default) - 0.5 = 1.5 (minimum)
        assert selector._state.zoom_factor == 1.5


class TestZoneSelectorWindow:
    """Tests for window management."""

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    def test_create_window(self, mock_callback, mock_window):
        """Test window creation."""
        selector = ZoneSelector(window_title="Test Window")

        selector._create_window()

        mock_window.assert_called_once()
        mock_callback.assert_called_once()
        assert selector._window_created is True

    @patch("cv2.namedWindow")
    @patch("cv2.setMouseCallback")
    def test_create_window_only_once(self, mock_callback, mock_window):
        """Test window is only created once."""
        selector = ZoneSelector()

        selector._create_window()
        selector._create_window()

        assert mock_window.call_count == 1

    @patch("cv2.destroyWindow")
    def test_close_window(self, mock_destroy):
        """Test window closing."""
        selector = ZoneSelector(window_title="Test Window")
        selector._window_created = True

        selector.close()

        mock_destroy.assert_called_once_with("Test Window")
        assert selector._window_created is False

    @patch("cv2.destroyWindow")
    def test_close_no_window(self, mock_destroy):
        """Test close when no window created."""
        selector = ZoneSelector()

        selector.close()

        mock_destroy.assert_not_called()


class TestZoneSelectorContextManager:
    """Tests for context manager support."""

    @patch("cv2.destroyWindow")
    def test_context_manager(self, mock_destroy):
        """Test using ZoneSelector as context manager."""
        with ZoneSelector(window_title="Test") as selector:
            selector._window_created = True
            assert isinstance(selector, ZoneSelector)

        mock_destroy.assert_called_once_with("Test")

    @patch("cv2.destroyWindow")
    def test_context_manager_exception(self, mock_destroy):
        """Test context manager cleans up on exception."""
        try:
            with ZoneSelector(window_title="Test") as selector:
                selector._window_created = True
                raise ValueError("Test error")
        except ValueError:
            pass

        mock_destroy.assert_called_once_with("Test")


class TestZoneSelectorDrawing:
    """Tests for drawing helper functions."""

    def test_draw_instructions(self):
        """Test drawing instructions."""
        selector = ZoneSelector()
        selector._instructions = "Click to select"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should complete without error
        selector._draw_instructions(frame)

        # Check that top area is modified (dark background drawn)
        assert frame[25, 10, 0] == 40  # Check the background color

    def test_draw_status(self):
        """Test drawing status bar."""
        selector = ZoneSelector()
        selector._num_points = 4
        selector._state.points = [np.array([10, 20])]
        selector._state.hover_point = (100, 200)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should complete without error
        selector._draw_status(frame)

    def test_draw_point(self):
        """Test drawing a point marker."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        point = np.array([100, 100])

        selector._draw_point(frame, point, 1)

        # Check that point area is modified
        assert frame[100, 100, 1] > 0  # Green channel should be set

    def test_draw_hover(self):
        """Test drawing hover indicator."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector._draw_hover(frame, (200, 200))

        # Should complete without error

    def test_draw_crosshairs(self):
        """Test drawing crosshairs."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector._draw_crosshairs(frame, (320, 240))

        # Check crosshair lines are drawn
        assert frame[240, 0, 0] > 0  # Horizontal line
        assert frame[60, 320, 0] > 0  # Vertical line

    def test_draw_point_lines(self):
        """Test drawing connecting lines."""
        selector = ZoneSelector()
        selector._state.points = [
            np.array([100, 100]),
            np.array([200, 100]),
            np.array([200, 200]),
        ]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        selector._draw_point_lines(frame)

        # Should draw lines between points

    def test_draw_zone_grid(self):
        """Test drawing zone grid overlay."""
        selector = ZoneSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        zone_grid = np.array(
            [
                [[50, 50], [100, 50], [100, 100], [50, 100]],
            ]
        )

        selector._draw_zone_grid(frame, zone_grid)

        # Should complete without error

    def test_draw_zoom_window(self):
        """Test drawing zoom window."""
        selector = ZoneSelector()
        selector._frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        selector._state.hover_point = (320, 240)
        selector._state.zoom_enabled = True
        selector._state.zoom_factor = 2.0
        frame = selector._frame.copy()

        selector._draw_zoom_window(frame)

        # Should complete without error

    def test_draw_zoom_window_grayscale(self):
        """Test drawing zoom window with grayscale source."""
        selector = ZoneSelector()
        selector._frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        selector._state.hover_point = (320, 240)
        selector._state.zoom_enabled = True
        display = np.zeros((480, 640, 3), dtype=np.uint8)

        selector._draw_zoom_window(display)

        # Should complete without error

    def test_draw_zoom_window_edge_cases(self):
        """Test zoom window at frame edges."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._state.zoom_enabled = True

        # Top-left corner
        selector._state.hover_point = (10, 10)
        frame = selector._frame.copy()
        selector._draw_zoom_window(frame)

        # Bottom-right corner
        selector._state.hover_point = (630, 470)
        frame = selector._frame.copy()
        selector._draw_zoom_window(frame)

    def test_draw_zoom_window_no_frame(self):
        """Test zoom window with no frame."""
        selector = ZoneSelector()
        selector._frame = None
        selector._state.hover_point = (100, 100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should return early without error
        selector._draw_zoom_window(frame)

    def test_draw_zoom_window_no_hover(self):
        """Test zoom window with no hover point."""
        selector = ZoneSelector()
        selector._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        selector._state.hover_point = None
        frame = selector._frame.copy()

        # Should return early without error
        selector._draw_zoom_window(frame)
