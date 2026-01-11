"""
Tests for touch zone calibration module.
"""

import numpy as np
import pytest

from chunivision.calibration.touchzone_calibration import (
    TouchZoneCalibrator,
    TouchZoneCalibrationResult,
    TouchZoneConfig,
    ZoneBoundary,
)
from chunivision.calibration.lens_distortion import LensDistortion


class TestTouchZoneConfig:
    """Tests for TouchZoneConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TouchZoneConfig()

        assert config.zone_grid == (16, 2)
        assert config.zone_width_cm == 2.75
        assert config.zone_height_cm == 4.5
        assert config.paper_size_cm == (44.0, 9.0)

    def test_custom_config(self):
        """Test custom configuration."""
        config = TouchZoneConfig(
            zone_grid=(8, 2),
            zone_width_cm=5.0,
            paper_size_cm=(40.0, 10.0),
        )

        assert config.zone_grid == (8, 2)
        assert config.zone_width_cm == 5.0
        assert config.paper_size_cm == (40.0, 10.0)

    def test_total_dimensions(self):
        """Test total width/height properties."""
        config = TouchZoneConfig(
            zone_grid=(16, 2),
            zone_width_cm=2.75,
            zone_height_cm=4.5,
        )

        assert config.total_width_cm == 16 * 2.75  # 44.0
        assert config.total_height_cm == 2 * 4.5  # 9.0


class TestZoneBoundary:
    """Tests for ZoneBoundary dataclass."""

    def test_zone_creation(self):
        """Test creating a zone boundary."""
        corners = np.array([[0, 0], [2.75, 0], [2.75, 4.5], [0, 4.5]], dtype=np.float32)
        center = np.array([1.375, 2.25], dtype=np.float32)

        zone = ZoneBoundary(
            zone_id=1,
            grid_row=0,
            grid_col=15,
            corners=corners,
            center=center,
        )

        assert zone.zone_id == 1
        assert zone.grid_row == 0
        assert zone.grid_col == 15

    def test_contains_point_inside(self):
        """Test point containment - inside."""
        corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        center = np.array([5, 5], dtype=np.float32)

        zone = ZoneBoundary(
            zone_id=1, grid_row=0, grid_col=0, corners=corners, center=center
        )

        assert zone.contains_point(np.array([5, 5])) == True
        assert zone.contains_point(np.array([1, 1])) == True
        assert zone.contains_point(np.array([9, 9])) == True

    def test_contains_point_outside(self):
        """Test point containment - outside."""
        corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        center = np.array([5, 5], dtype=np.float32)

        zone = ZoneBoundary(
            zone_id=1, grid_row=0, grid_col=0, corners=corners, center=center
        )

        assert zone.contains_point(np.array([15, 5])) == False
        assert zone.contains_point(np.array([-1, 5])) == False


class TestTouchZoneCalibrationResult:
    """Tests for TouchZoneCalibrationResult."""

    def test_result_creation(self):
        """Test creating a calibration result."""
        transform = np.eye(3)
        corners = np.array(
            [[50, 400], [590, 400], [580, 80], [60, 80]], dtype=np.float32
        )
        physical = np.array([[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32)

        result = TouchZoneCalibrationResult(
            perspective_transform=transform,
            inverse_transform=np.linalg.inv(transform),
            image_corners=corners,
            physical_corners=physical,
            zone_boundaries=[],
            reprojection_error=0.0,
            quality_score=1.0,
            image_size=(640, 480),
        )

        assert result.quality_score == 1.0
        assert result.reprojection_error == 0.0
        assert result.image_size == (640, 480)

    def test_get_zone_at_point_no_zones(self):
        """Test get_zone_at_point with no zones."""
        result = TouchZoneCalibrationResult(
            perspective_transform=np.eye(3),
            inverse_transform=np.eye(3),
            image_corners=np.zeros((4, 2)),
            physical_corners=np.zeros((4, 2)),
            zone_boundaries=[],
            reprojection_error=0.0,
            quality_score=1.0,
            image_size=(640, 480),
        )

        assert result.get_zone_at_point(np.array([5, 5])) is None

    def test_get_zone_at_point_found(self):
        """Test get_zone_at_point finding a zone."""
        zone = ZoneBoundary(
            zone_id=5,
            grid_row=0,
            grid_col=0,
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            center=np.array([5, 5]),
        )

        result = TouchZoneCalibrationResult(
            perspective_transform=np.eye(3),
            inverse_transform=np.eye(3),
            image_corners=np.zeros((4, 2)),
            physical_corners=np.zeros((4, 2)),
            zone_boundaries=[zone],
            reprojection_error=0.0,
            quality_score=1.0,
            image_size=(640, 480),
        )

        assert result.get_zone_at_point(np.array([5, 5])) == 5


class TestTouchZoneCalibrator:
    """Tests for TouchZoneCalibrator class."""

    def test_default_creation(self):
        """Test creating calibrator with defaults."""
        calibrator = TouchZoneCalibrator()

        assert calibrator.config.zone_grid == (16, 2)
        assert calibrator.lens_distortion is None

    def test_creation_with_lens_distortion(self):
        """Test creating with lens distortion handler."""
        distortion = LensDistortion()
        calibrator = TouchZoneCalibrator(lens_distortion=distortion)

        assert calibrator.lens_distortion is distortion

    def test_physical_corners(self):
        """Test physical corners are computed correctly."""
        config = TouchZoneConfig(paper_size_cm=(44.0, 9.0))
        calibrator = TouchZoneCalibrator(config)

        corners = calibrator._physical_corners
        expected = np.array([[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32)

        np.testing.assert_array_equal(corners, expected)

    def test_order_corners(self):
        """Test corner ordering."""
        calibrator = TouchZoneCalibrator()

        # Random order corners
        corners = np.array(
            [
                [100, 0],  # top-right
                [0, 100],  # bottom-left
                [100, 100],  # bottom-right
                [0, 0],  # top-left
            ],
            dtype=np.float32,
        )

        ordered = calibrator._order_corners(corners)

        # Should be: bottom-left, bottom-right, top-right, top-left
        assert ordered[0, 1] > ordered[3, 1]  # BL.y > TL.y
        assert ordered[1, 0] > ordered[0, 0]  # BR.x > BL.x

    def test_set_callbacks(self):
        """Test setting callbacks."""
        calibrator = TouchZoneCalibrator()

        progress_called = []
        calibrator.set_callbacks(
            on_progress=lambda s, p: progress_called.append((s, p)),
        )

        calibrator._notify_progress("test", 0.75)

        assert progress_called == [("test", 0.75)]

    def test_generate_zone_boundaries(self):
        """Test zone boundary generation."""
        config = TouchZoneConfig(zone_grid=(16, 2))
        calibrator = TouchZoneCalibrator(config)

        boundaries = calibrator._generate_zone_boundaries()

        assert len(boundaries) == 32

        # Check zone IDs are 1-32
        ids = [z.zone_id for z in boundaries]
        assert sorted(ids) == list(range(1, 33))

        # Check zone 1 is bottom-right
        zone1 = next(z for z in boundaries if z.zone_id == 1)
        assert zone1.grid_row == 0
        assert zone1.grid_col == 15

        # Check zone 32 is top-left
        zone32 = next(z for z in boundaries if z.zone_id == 32)
        assert zone32.grid_row == 1
        assert zone32.grid_col == 0

    def test_calibrate_valid_corners(self):
        """Test calibration with valid corners."""
        calibrator = TouchZoneCalibrator()

        # Corners forming a quadrilateral
        corners = np.array(
            [
                [50, 400],  # bottom-left
                [590, 400],  # bottom-right
                [580, 80],  # top-right
                [60, 80],  # top-left
            ],
            dtype=np.float32,
        )

        result = calibrator.calibrate(corners, image_size=(640, 480))

        assert isinstance(result, TouchZoneCalibrationResult)
        assert result.perspective_transform.shape == (3, 3)
        assert result.inverse_transform.shape == (3, 3)
        assert len(result.zone_boundaries) == 32
        assert result.quality_score > 0.5

    def test_calibrate_invalid_corners(self):
        """Test calibration with invalid corners raises error."""
        calibrator = TouchZoneCalibrator()

        # Wrong shape
        corners = np.array([[50, 400], [590, 400]], dtype=np.float32)

        with pytest.raises(ValueError):
            calibrator.calibrate(corners, image_size=(640, 480))

    def test_preprocess_frame_grayscale(self):
        """Test preprocessing a grayscale frame."""
        calibrator = TouchZoneCalibrator()

        frame = np.zeros((480, 640), dtype=np.uint8)
        result = calibrator.preprocess_frame(frame, undistort=False)

        assert result.shape == (480, 640)

    def test_preprocess_frame_bgr(self):
        """Test preprocessing a BGR frame."""
        calibrator = TouchZoneCalibrator()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = calibrator.preprocess_frame(frame, undistort=False)

        # Should be converted to grayscale
        assert result.shape == (480, 640)


class TestTouchZoneCalibrationIntegration:
    """Integration tests for touch zone calibration."""

    def test_full_calibration_workflow(self):
        """Test complete calibration workflow."""
        config = TouchZoneConfig(
            zone_grid=(4, 2),
            paper_size_cm=(20.0, 10.0),
        )
        calibrator = TouchZoneCalibrator(config)

        # Simulate corners from a camera view
        corners = np.array(
            [
                [100, 350],  # bottom-left
                [540, 350],  # bottom-right
                [520, 130],  # top-right
                [120, 130],  # top-left
            ],
            dtype=np.float32,
        )

        result = calibrator.calibrate(corners, image_size=(640, 480))

        # Verify result
        assert result.quality_score > 0.5
        assert len(result.zone_boundaries) == 8  # 4x2 = 8 zones

        # Test point transformation
        # Image center should map somewhere in the middle of physical space
        center_physical = result.transform_point_to_physical(np.array([320, 240]))
        assert 0 < center_physical[0] < 20
        assert 0 < center_physical[1] < 10
