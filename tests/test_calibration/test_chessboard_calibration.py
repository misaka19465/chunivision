"""
Tests for chessboard calibration module.
"""

import numpy as np
import pytest

from chunivision.calibration.chessboard_calibration import (
    ChessboardCalibrator,
    ChessboardCalibrationResult,
    ChessboardCapture,
    ChessboardConfig,
    StereoCalibrationResult,
)
from chunivision.calibration.lens_distortion import LensDistortion


class TestChessboardConfig:
    """Tests for ChessboardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChessboardConfig()

        assert config.pattern_size == (9, 6)
        assert config.square_size == 2.5
        assert config.a4_paper_size == (29.7, 21.0)
        assert config.min_captures == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChessboardConfig(
            pattern_size=(7, 5),
            square_size=3.0,
            min_captures=10,
        )

        assert config.pattern_size == (7, 5)
        assert config.square_size == 3.0
        assert config.min_captures == 10

    def test_get_object_points(self):
        """Test object points generation."""
        config = ChessboardConfig(pattern_size=(4, 3), square_size=2.0)
        objp = config.get_object_points()

        # 4x3 pattern = 12 points
        assert objp.shape == (12, 3)
        # All Z should be 0
        assert np.all(objp[:, 2] == 0)
        # First point at origin
        assert objp[0, 0] == 0
        assert objp[0, 1] == 0
        # Points should be spaced by square_size
        assert objp[1, 0] == 2.0  # Second point in first row


class TestChessboardCapture:
    """Tests for ChessboardCapture dataclass."""

    def test_capture_creation(self):
        """Test creating a capture object."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        corners = np.zeros((54, 1, 2), dtype=np.float32)
        objp = np.zeros((54, 3), dtype=np.float32)

        capture = ChessboardCapture(
            frame=frame,
            corners=corners,
            object_points=objp,
            quality=0.95,
        )

        assert capture.frame.shape == (480, 640)
        assert capture.corners.shape == (54, 1, 2)
        assert capture.quality == 0.95


class TestChessboardCalibrationResult:
    """Tests for ChessboardCalibrationResult."""

    def test_quality_score_perfect(self):
        """Test quality score with zero error."""
        result = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.0,
            num_captures=10,
            image_size=(640, 480),
        )

        assert result.get_quality_score() == 1.0

    def test_quality_score_high_error(self):
        """Test quality score with high error."""
        result = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=5.0,  # High error
            num_captures=10,
            image_size=(640, 480),
        )

        # Should be low but not zero
        assert 0.0 < result.get_quality_score() < 0.5


class TestChessboardCalibrator:
    """Tests for ChessboardCalibrator class."""

    def test_default_creation(self):
        """Test creating calibrator with defaults."""
        calibrator = ChessboardCalibrator()

        assert calibrator.config.pattern_size == (9, 6)
        assert calibrator.lens_distortion is None

    def test_creation_with_lens_distortion(self):
        """Test creating calibrator with lens distortion handler."""
        distortion = LensDistortion()
        calibrator = ChessboardCalibrator(lens_distortion=distortion)

        assert calibrator.lens_distortion is distortion

    def test_get_capture_count_empty(self):
        """Test capture count when empty."""
        calibrator = ChessboardCalibrator()

        assert calibrator.get_capture_count("left") == 0
        assert calibrator.get_capture_count("right") == 0

    def test_get_capture_count_invalid_camera(self):
        """Test capture count with invalid camera name."""
        calibrator = ChessboardCalibrator()

        with pytest.raises(ValueError):
            calibrator.get_capture_count("invalid")

    def test_clear_captures(self):
        """Test clearing captures."""
        calibrator = ChessboardCalibrator()

        # Manually add some captures
        frame = np.zeros((480, 640), dtype=np.uint8)
        corners = np.zeros((54, 1, 2), dtype=np.float32)
        objp = np.zeros((54, 3), dtype=np.float32)

        calibrator._left_captures.append(
            ChessboardCapture(
                frame=frame, corners=corners, object_points=objp, quality=0.9
            )
        )
        calibrator._right_captures.append(
            ChessboardCapture(
                frame=frame, corners=corners, object_points=objp, quality=0.9
            )
        )

        assert calibrator.get_capture_count("left") == 1
        assert calibrator.get_capture_count("right") == 1

        calibrator.clear_captures()

        assert calibrator.get_capture_count("left") == 0
        assert calibrator.get_capture_count("right") == 0

    def test_clear_single_camera(self):
        """Test clearing captures for single camera."""
        calibrator = ChessboardCalibrator()

        frame = np.zeros((480, 640), dtype=np.uint8)
        corners = np.zeros((54, 1, 2), dtype=np.float32)
        objp = np.zeros((54, 3), dtype=np.float32)

        calibrator._left_captures.append(
            ChessboardCapture(
                frame=frame, corners=corners, object_points=objp, quality=0.9
            )
        )
        calibrator._right_captures.append(
            ChessboardCapture(
                frame=frame, corners=corners, object_points=objp, quality=0.9
            )
        )

        calibrator.clear_captures("left")

        assert calibrator.get_capture_count("left") == 0
        assert calibrator.get_capture_count("right") == 1

    def test_set_callbacks(self):
        """Test setting callbacks."""
        calibrator = ChessboardCalibrator()

        progress_called = []
        message_called = []

        calibrator.set_callbacks(
            on_progress=lambda s, p: progress_called.append((s, p)),
            on_message=lambda m: message_called.append(m),
        )

        calibrator._notify_progress("test", 0.5)
        calibrator._notify_message("hello")

        assert progress_called == [("test", 0.5)]
        assert message_called == ["hello"]

    def test_calibrate_camera_insufficient_captures(self):
        """Test calibration with insufficient captures raises error."""
        calibrator = ChessboardCalibrator()

        with pytest.raises(ValueError, match="Insufficient"):
            calibrator.calibrate_camera("left")

    def test_compute_detection_quality(self):
        """Test detection quality computation."""
        calibrator = ChessboardCalibrator(config=ChessboardConfig(pattern_size=(4, 3)))

        # Create regular grid of corners
        corners = np.zeros((12, 1, 2), dtype=np.float32)
        for row in range(3):
            for col in range(4):
                idx = row * 4 + col
                corners[idx, 0, 0] = col * 50.0  # Regular horizontal spacing
                corners[idx, 0, 1] = row * 50.0  # Regular vertical spacing

        quality = calibrator._compute_detection_quality(corners)

        # Should be high quality for regular grid
        assert quality > 0.8

    def test_compute_detection_quality_irregular(self):
        """Test detection quality with irregular corners."""
        calibrator = ChessboardCalibrator(config=ChessboardConfig(pattern_size=(4, 3)))

        # Create extremely irregular grid with random distribution
        np.random.seed(42)
        corners = np.random.rand(12, 1, 2).astype(np.float32) * 500

        quality = calibrator._compute_detection_quality(corners)

        # Should be lower quality for irregular grid (but may not be too low
        # since random points can have some regularity)
        assert quality < 0.8  # Just less than a well-structured grid


class TestStereoCalibrationResult:
    """Tests for StereoCalibrationResult."""

    def test_baseline_property(self):
        """Test baseline computation from translation vector."""
        left = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.0,
            num_captures=10,
            image_size=(640, 480),
        )
        right = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.0,
            num_captures=10,
            image_size=(640, 480),
        )

        result = StereoCalibrationResult(
            left_result=left,
            right_result=right,
            rotation_matrix=np.eye(3),
            translation_vector=np.array([20.0, 0.0, 0.0]),  # 20cm baseline
            essential_matrix=np.eye(3),
            fundamental_matrix=np.eye(3),
            rectify_left=np.eye(3),
            rectify_right=np.eye(3),
            projection_left=np.hstack([np.eye(3), np.zeros((3, 1))]),
            projection_right=np.hstack([np.eye(3), np.zeros((3, 1))]),
            disparity_to_depth=np.eye(4),
            roi_left=(0, 0, 640, 480),
            roi_right=(0, 0, 640, 480),
            stereo_error=0.0,
        )

        assert result.baseline == pytest.approx(20.0)

    def test_get_quality_score(self):
        """Test overall quality score computation."""
        left = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.0,
            num_captures=10,
            image_size=(640, 480),
        )
        right = ChessboardCalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rvecs=[],
            tvecs=[],
            reprojection_error=0.0,
            num_captures=10,
            image_size=(640, 480),
        )

        result = StereoCalibrationResult(
            left_result=left,
            right_result=right,
            rotation_matrix=np.eye(3),
            translation_vector=np.array([20.0, 0.0, 0.0]),
            essential_matrix=np.eye(3),
            fundamental_matrix=np.eye(3),
            rectify_left=np.eye(3),
            rectify_right=np.eye(3),
            projection_left=np.hstack([np.eye(3), np.zeros((3, 1))]),
            projection_right=np.hstack([np.eye(3), np.zeros((3, 1))]),
            disparity_to_depth=np.eye(4),
            roi_left=(0, 0, 640, 480),
            roi_right=(0, 0, 640, 480),
            stereo_error=0.0,
        )

        # All errors are 0, so quality should be 1.0
        assert result.get_quality_score() == 1.0
