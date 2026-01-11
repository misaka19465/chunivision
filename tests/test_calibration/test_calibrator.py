"""
Tests for calibrator module.

Updated to work with the new two-stage calibration API.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from chunivision.calibration.calibrator import (
    Calibrator,
    CalibrationConfig,
    CalibrationError,
    FullCalibrationResult,
)
from chunivision.calibration.calibration_data import CalibrationData
from chunivision.calibration.chessboard_calibration import ChessboardConfig
from chunivision.calibration.touchzone_calibration import TouchZoneConfig


class TestCalibratorCreation:
    """Tests for Calibrator instantiation."""

    def test_default_creation(self):
        """Test creating Calibrator with defaults."""
        calibrator = Calibrator()

        # Legacy attributes should still work
        assert calibrator.board_physical_size == (44.0, 9.0)
        assert calibrator.zone_grid == (16, 2)
        assert calibrator.image_size == (640, 480)
        assert calibrator.stereo_baseline == 20.0

    def test_custom_creation(self):
        """Test creating Calibrator with custom config."""
        config = CalibrationConfig(
            touchzone_config=TouchZoneConfig(
                paper_size_cm=(50.0, 10.0),
                zone_grid=(20, 2),
            ),
        )
        calibrator = Calibrator(config=config)

        assert calibrator.config.touchzone_config.paper_size_cm == (50.0, 10.0)
        assert calibrator.config.touchzone_config.zone_grid == (20, 2)

    def test_world_points_computed(self):
        """Test that world points are computed from board size."""
        config = CalibrationConfig(
            touchzone_config=TouchZoneConfig(paper_size_cm=(100.0, 20.0))
        )
        calibrator = Calibrator(config=config)

        # World points should be corners of the board
        expected = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], dtype=np.float32)
        np.testing.assert_array_equal(calibrator._world_points, expected)


class TestCalibratorCallbacks:
    """Tests for Calibrator callback functionality."""

    def test_set_callbacks(self):
        """Test setting callback functions."""
        calibrator = Calibrator()

        progress_callback = Mock()
        message_callback = Mock()

        calibrator.set_callbacks(
            on_progress=progress_callback, on_message=message_callback
        )

        # Trigger callbacks
        calibrator._notify_progress("test", 0.5)
        calibrator._notify_message("test message")

        progress_callback.assert_called_once_with("test", 0.5)
        message_callback.assert_called_once_with("test message")

    def test_notify_without_callbacks(self):
        """Test that notify methods work without callbacks set."""
        calibrator = Calibrator()

        # Should not raise
        calibrator._notify_progress("test", 0.5)
        calibrator._notify_message("test message")


class TestCalibrateFromPoints:
    """Tests for calibrate_from_points method."""

    def test_valid_points(self):
        """Test calibration with valid points."""
        calibrator = Calibrator()

        # Simulate realistic points (corners of board in image)
        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )

        result = calibrator.calibrate_from_points(left_points, right_points)

        assert isinstance(result, CalibrationData)
        assert result.camera_left_transform.shape == (3, 3)
        assert result.camera_right_transform.shape == (3, 3)
        assert result.stereo_baseline == 20.0

    def test_with_custom_height_thresholds(self):
        """Test calibration with custom height thresholds."""
        calibrator = Calibrator()

        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )
        custom_thresholds = [5, 10, 15, 20, 25, 30]

        result = calibrator.calibrate_from_points(
            left_points, right_points, height_thresholds=custom_thresholds
        )

        np.testing.assert_array_equal(result.height_thresholds, custom_thresholds)

    def test_invalid_left_points_shape(self):
        """Test that wrong left points shape raises error."""
        calibrator = Calibrator()

        left_points = np.array([[50, 400], [590, 400]], dtype=np.float32)  # Only 2
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )

        with pytest.raises(ValueError, match="Expected 4 corners"):
            calibrator.calibrate_from_points(left_points, right_points)

    def test_invalid_right_points_shape(self):
        """Test that wrong right points shape raises error."""
        calibrator = Calibrator()

        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array([[60, 390], [600, 390], [590, 90]], dtype=np.float32)

        with pytest.raises(ValueError, match="Expected 4 corners"):
            calibrator.calibrate_from_points(left_points, right_points)

    def test_calibration_quality_stored(self):
        """Test that calibration quality metrics are stored."""
        calibrator = Calibrator()

        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )

        result = calibrator.calibrate_from_points(left_points, right_points)

        assert "left_quality_score" in result.calibration_quality
        assert "right_quality_score" in result.calibration_quality
        assert "left_mean_error" in result.calibration_quality
        assert "right_mean_error" in result.calibration_quality

    def test_calibration_points_stored(self):
        """Test that selected calibration points are stored."""
        calibrator = Calibrator()

        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )

        result = calibrator.calibrate_from_points(left_points, right_points)

        np.testing.assert_array_almost_equal(
            result.left_calibration_points, left_points
        )
        np.testing.assert_array_almost_equal(
            result.right_calibration_points, right_points
        )


class TestZoneBoundaries:
    """Tests for zone boundary calculation."""

    def test_zone_boundaries_structure(self):
        """Test zone boundaries dictionary structure."""
        calibrator = Calibrator()

        boundaries = calibrator._calculate_zone_boundaries()

        assert "grid" in boundaries
        assert "zone_size" in boundaries
        assert "board_size" in boundaries
        assert "zones" in boundaries

        assert boundaries["grid"] == [16, 2]
        assert len(boundaries["zones"]) == 32

    def test_zone_size_calculation(self):
        """Test that zone sizes are calculated correctly."""
        calibrator = Calibrator()

        boundaries = calibrator._calculate_zone_boundaries()

        assert boundaries["zone_size"][0] == pytest.approx(44.0 / 16)  # 2.75
        assert boundaries["zone_size"][1] == pytest.approx(9.0 / 2)  # 4.5

    def test_zone_id_numbering(self):
        """Test zone ID numbering convention."""
        calibrator = Calibrator()

        boundaries = calibrator._calculate_zone_boundaries()
        zones = boundaries["zones"]

        # Should be sorted by ID
        zone_ids = [z["id"] for z in zones]
        assert zone_ids == list(range(1, 33))

        # Check specific zones
        # Zone 1 should be bottom-right (row 0, col 15)
        zone_1 = next(z for z in zones if z["id"] == 1)
        assert zone_1["grid_row"] == 0
        assert zone_1["grid_col"] == 15

        # Zone 2 should be top-right (row 1, col 15)
        zone_2 = next(z for z in zones if z["id"] == 2)
        assert zone_2["grid_row"] == 1
        assert zone_2["grid_col"] == 15

        # Zone 31 should be bottom-left (row 0, col 0)
        zone_31 = next(z for z in zones if z["id"] == 31)
        assert zone_31["grid_row"] == 0
        assert zone_31["grid_col"] == 0

        # Zone 32 should be top-left (row 1, col 0)
        zone_32 = next(z for z in zones if z["id"] == 32)
        assert zone_32["grid_row"] == 1
        assert zone_32["grid_col"] == 0


class TestSaveLoadCalibration:
    """Tests for save/load calibration functionality."""

    def test_save_and_load_via_calibration_data(self):
        """Test saving and loading calibration using CalibrationData directly."""
        calibrator = Calibrator()

        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )

        data = calibrator.calibrate_from_points(left_points, right_points)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"
            # Save directly via CalibrationData
            data.save(str(path))

            assert path.exists()

            # Load via calibrator
            loaded = calibrator.load_calibration(str(path))

            assert loaded.stereo_baseline == data.stereo_baseline
            np.testing.assert_allclose(
                loaded.camera_left_transform, data.camera_left_transform
            )

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent file raises error."""
        calibrator = Calibrator()

        with pytest.raises(CalibrationError, match="not found"):
            calibrator.load_calibration("/nonexistent/path/calibration.yaml")


class TestValidateCalibration:
    """Tests for calibration validation."""

    def test_valid_calibration(self):
        """Test validating a good calibration."""
        calibrator = Calibrator()

        left_points = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        right_points = np.array(
            [[60, 390], [600, 390], [590, 90], [70, 90]], dtype=np.float32
        )

        data = calibrator.calibrate_from_points(left_points, right_points)

        is_valid, quality, issues = calibrator.validate_calibration(data)

        assert isinstance(is_valid, bool)
        assert 0.0 <= quality <= 1.0
        assert isinstance(issues, list)

    def test_invalid_data_reported(self):
        """Test that invalid data is reported."""
        calibrator = Calibrator()

        data = CalibrationData()
        # Set invalid stereo baseline
        data.stereo_baseline = 0.0

        is_valid, quality, issues = calibrator.validate_calibration(data)

        assert not is_valid
        # Should have some validation issue
        assert quality < calibrator.MIN_QUALITY_THRESHOLD or len(issues) > 0


class TestHeightCalibration:
    """Tests for height threshold calibration."""

    def test_calibrate_height_thresholds(self):
        """Test height calibration with mock height getter."""
        calibrator = Calibrator()

        # Mock height getter that returns slightly noisy values
        heights = iter(
            [17.5, 17.8, 17.9, 18.0, 17.9] * 6  # Level 0
            + [21.0, 21.2, 21.3, 21.4, 21.3] * 6  # Level 1
            + [24.5, 24.6, 24.7, 24.8, 24.7] * 6  # Level 2
            + [28.0, 28.1, 28.1, 28.2, 28.1] * 6  # Level 3
            + [31.3, 31.4, 31.5, 31.6, 31.5] * 6  # Level 4
            + [34.8, 34.9, 35.0, 35.0, 34.9] * 6
        )  # Level 5

        def get_height():
            try:
                return next(heights)
            except StopIteration:
                return None

        thresholds = calibrator.calibrate_height_thresholds(get_height)

        assert len(thresholds) == 6
        # Should be roughly ascending
        for i in range(5):
            assert thresholds[i] < thresholds[i + 1]

    def test_calibrate_height_thresholds_insufficient_data(self):
        """Test height calibration fails with insufficient data."""
        calibrator = Calibrator()

        # Mock that returns None (no hand detected)
        def get_height():
            return None

        with pytest.raises(CalibrationError, match="Not enough"):
            calibrator.calibrate_height_thresholds(get_height)


class TestGetCameraFrame:
    """Tests for _get_camera_frame helper method."""

    def test_get_frame_method(self):
        """Test getting frame via get_frame() method."""
        calibrator = Calibrator()

        camera = Mock()
        expected_frame = np.zeros((480, 640), dtype=np.uint8)
        camera.get_frame.return_value = expected_frame

        frame = calibrator._get_camera_frame(camera)

        np.testing.assert_array_equal(frame, expected_frame)

    def test_read_method_fallback(self):
        """Test getting frame via read() method."""
        calibrator = Calibrator()

        camera = Mock(spec=["read"])
        expected_frame = np.zeros((480, 640), dtype=np.uint8)
        camera.read.return_value = (True, expected_frame)

        frame = calibrator._get_camera_frame(camera)

        np.testing.assert_array_equal(frame, expected_frame)

    def test_read_method_failure(self):
        """Test error when read() returns False."""
        calibrator = Calibrator()

        camera = Mock(spec=["read"])
        camera.read.return_value = (False, None)

        with pytest.raises(CalibrationError, match="Failed to read"):
            calibrator._get_camera_frame(camera)

    def test_no_method_raises(self):
        """Test error when camera has neither method."""
        calibrator = Calibrator()

        camera = Mock(spec=[])  # No get_frame or read

        with pytest.raises(CalibrationError, match="must have"):
            calibrator._get_camera_frame(camera)

    def test_none_frame_returns_none(self):
        """Test that get_frame returning None returns None."""
        calibrator = Calibrator()

        camera = Mock()
        camera.get_frame.return_value = None

        # The new API returns None instead of raising
        result = calibrator._get_camera_frame(camera)
        assert result is None


class TestFullCalibrationResult:
    """Tests for FullCalibrationResult class."""

    def test_to_calibration_data(self):
        """Test conversion to CalibrationData."""
        from chunivision.calibration.chessboard_calibration import (
            ChessboardCalibrationResult,
            StereoCalibrationResult,
        )
        from chunivision.calibration.touchzone_calibration import (
            TouchZoneCalibrationResult,
            ZoneBoundary,
        )
        from chunivision.calibration.lens_distortion import LensDistortion

        # Create mock chessboard results
        left_camera = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[np.zeros(3)],
            tvecs=[np.zeros(3)],
            reprojection_error=0.5,
            num_captures=10,
            image_size=(640, 480),
        )
        right_camera = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[np.zeros(3)],
            tvecs=[np.zeros(3)],
            reprojection_error=0.5,
            num_captures=10,
            image_size=(640, 480),
        )

        stereo = StereoCalibrationResult(
            left_result=left_camera,
            right_result=right_camera,
            rotation_matrix=np.eye(3),
            translation_vector=np.array([20.0, 0.0, 0.0]),
            essential_matrix=np.eye(3),
            fundamental_matrix=np.eye(3),
            stereo_error=0.5,
            rectify_left=np.eye(3),
            rectify_right=np.eye(3),
            projection_left=np.eye(3, 4),
            projection_right=np.eye(3, 4),
            disparity_to_depth=np.eye(4),
            roi_left=(0, 0, 640, 480),
            roi_right=(0, 0, 640, 480),
        )

        # Create mock zone with at least one boundary
        zone_boundary = ZoneBoundary(
            zone_id=1,
            grid_row=0,
            grid_col=0,
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            center=np.array([5, 5], dtype=np.float32),
        )

        left_zone = TouchZoneCalibrationResult(
            perspective_transform=np.eye(3),
            inverse_transform=np.eye(3),
            image_corners=np.array(
                [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32
            ),
            physical_corners=np.array(
                [[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32
            ),
            zone_boundaries=[zone_boundary],
            reprojection_error=0.0,
            quality_score=1.0,
            image_size=(640, 480),
        )

        right_zone = TouchZoneCalibrationResult(
            perspective_transform=np.eye(3),
            inverse_transform=np.eye(3),
            image_corners=np.array(
                [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32
            ),
            physical_corners=np.array(
                [[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32
            ),
            zone_boundaries=[zone_boundary],
            reprojection_error=0.0,
            quality_score=1.0,
            image_size=(640, 480),
        )

        result = FullCalibrationResult(
            stereo_result=stereo,
            left_zone_result=left_zone,
            right_zone_result=right_zone,
            lens_distortion=LensDistortion(),
            timestamp=datetime.now(),
            overall_quality=0.95,
        )

        data = result.to_calibration_data()

        assert isinstance(data, CalibrationData)
        assert data.stereo_baseline == pytest.approx(20.0)
        assert len(data.height_thresholds) == 6

    def test_overall_quality(self):
        """Test overall quality is stored correctly."""
        from chunivision.calibration.chessboard_calibration import (
            ChessboardCalibrationResult,
            StereoCalibrationResult,
        )
        from chunivision.calibration.touchzone_calibration import (
            TouchZoneCalibrationResult,
            ZoneBoundary,
        )
        from chunivision.calibration.lens_distortion import LensDistortion

        # Create minimal mock results
        left_camera = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.3,
            num_captures=10,
            image_size=(640, 480),
        )
        right_camera = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.3,
            num_captures=10,
            image_size=(640, 480),
        )

        stereo = StereoCalibrationResult(
            left_result=left_camera,
            right_result=right_camera,
            rotation_matrix=np.eye(3),
            translation_vector=np.array([20.0, 0.0, 0.0]),
            essential_matrix=np.eye(3),
            fundamental_matrix=np.eye(3),
            stereo_error=0.3,
            rectify_left=np.eye(3),
            rectify_right=np.eye(3),
            projection_left=np.eye(3, 4),
            projection_right=np.eye(3, 4),
            disparity_to_depth=np.eye(4),
            roi_left=(0, 0, 640, 480),
            roi_right=(0, 0, 640, 480),
        )

        zone_boundary = ZoneBoundary(
            zone_id=1,
            grid_row=0,
            grid_col=0,
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            center=np.array([5, 5], dtype=np.float32),
        )

        left_zone = TouchZoneCalibrationResult(
            perspective_transform=np.eye(3),
            inverse_transform=np.eye(3),
            image_corners=np.array(
                [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32
            ),
            physical_corners=np.array(
                [[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32
            ),
            zone_boundaries=[zone_boundary],
            reprojection_error=0.0,
            quality_score=0.8,
            image_size=(640, 480),
        )

        right_zone = TouchZoneCalibrationResult(
            perspective_transform=np.eye(3),
            inverse_transform=np.eye(3),
            image_corners=np.array(
                [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32
            ),
            physical_corners=np.array(
                [[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32
            ),
            zone_boundaries=[zone_boundary],
            reprojection_error=0.0,
            quality_score=0.85,
            image_size=(640, 480),
        )

        result = FullCalibrationResult(
            stereo_result=stereo,
            left_zone_result=left_zone,
            right_zone_result=right_zone,
            lens_distortion=LensDistortion(),
            timestamp=datetime.now(),
            overall_quality=0.82,
        )

        assert 0.7 <= result.overall_quality <= 1.0
