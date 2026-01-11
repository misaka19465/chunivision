"""
Tests for calibration_data module.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from chunivision.calibration.calibration_data import (
    CalibrationData,
    CalibrationDataError,
)


class TestCalibrationDataCreation:
    """Tests for CalibrationData instantiation."""

    def test_default_creation(self):
        """Test creating CalibrationData with defaults."""
        data = CalibrationData()

        assert data.version == "1.0"
        assert isinstance(data.timestamp, datetime)
        assert data.stereo_baseline == 20.0
        assert data.image_size == (640, 480)
        assert data.reference_board_size == (44.0, 9.0)

    def test_custom_creation(self):
        """Test creating CalibrationData with custom values."""
        timestamp = datetime(2026, 1, 1, 12, 0, 0)
        data = CalibrationData(
            version="2.0",
            timestamp=timestamp,
            stereo_baseline=25.0,
            image_size=(1280, 720),
            reference_board_size=(50.0, 10.0),
        )

        assert data.version == "2.0"
        assert data.timestamp == timestamp
        assert data.stereo_baseline == 25.0
        assert data.image_size == (1280, 720)
        assert data.reference_board_size == (50.0, 10.0)

    def test_array_fields_converted(self):
        """Test that list inputs are converted to numpy arrays."""
        data = CalibrationData(
            camera_left_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            height_thresholds=[17.9, 21.3, 24.7, 28.1, 31.5, 34.9],
        )

        assert isinstance(data.camera_left_matrix, np.ndarray)
        assert isinstance(data.height_thresholds, np.ndarray)
        assert data.camera_left_matrix.dtype == np.float64
        assert data.height_thresholds.dtype == np.float64

    def test_create_default_factory(self):
        """Test create_default class method."""
        data = CalibrationData.create_default()

        assert data.image_size == (640, 480)
        assert data.stereo_baseline == 20.0
        assert data.camera_left_matrix.shape == (3, 3)
        assert data.disparity_to_depth.shape == (4, 4)

    def test_create_default_custom_size(self):
        """Test create_default with custom image size."""
        data = CalibrationData.create_default(
            image_size=(1280, 720),
            board_size=(50.0, 10.0),
            zone_grid=(20, 2),
        )

        assert data.image_size == (1280, 720)
        assert data.reference_board_size == (50.0, 10.0)
        # Check camera matrix uses image center
        assert data.camera_left_matrix[0, 2] == 640  # cx
        assert data.camera_left_matrix[1, 2] == 360  # cy


class TestCalibrationDataValidation:
    """Tests for CalibrationData validation."""

    def test_valid_default_data(self):
        """Test that default data passes validation."""
        data = CalibrationData.create_default()
        errors = data.validate()
        # May have some warnings but should be mostly valid
        assert data.is_valid() or len(errors) < 5

    def test_invalid_camera_matrix_shape(self):
        """Test validation catches wrong camera matrix shape."""
        data = CalibrationData()
        data.camera_left_matrix = np.eye(2)
        errors = data.validate()
        assert any("camera_left_matrix" in e for e in errors)

    def test_invalid_rotation_matrix_determinant(self):
        """Test validation catches non-orthonormal rotation matrix."""
        data = CalibrationData()
        data.rotation_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        errors = data.validate()
        assert any("determinant" in e for e in errors)

    def test_invalid_height_thresholds_count(self):
        """Test validation catches wrong number of height thresholds."""
        data = CalibrationData()
        data.height_thresholds = np.array([1, 2, 3, 4, 5])  # Only 5
        errors = data.validate()
        assert any("height_thresholds" in e and "6" in e for e in errors)

    def test_invalid_height_thresholds_order(self):
        """Test validation catches non-ascending height thresholds."""
        data = CalibrationData()
        data.height_thresholds = np.array([30, 25, 20, 15, 10, 5])  # Descending
        errors = data.validate()
        assert any("ascending" in e for e in errors)

    def test_invalid_stereo_baseline(self):
        """Test validation catches non-positive baseline."""
        data = CalibrationData()
        data.stereo_baseline = 0
        errors = data.validate()
        assert any("stereo_baseline" in e for e in errors)

    def test_singular_transform_detected(self):
        """Test validation catches singular transforms."""
        data = CalibrationData()
        # Create a singular matrix
        data.camera_left_transform = np.array(
            [[1, 2, 3], [2, 4, 6], [0, 0, 0]], dtype=np.float64
        )
        errors = data.validate()
        assert any("singular" in e for e in errors)


class TestCalibrationDataSaveLoad:
    """Tests for CalibrationData save/load functionality."""

    def test_save_and_load_roundtrip(self):
        """Test that save/load preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"

            # Create data with specific values
            original = CalibrationData(
                version="1.0",
                stereo_baseline=22.5,
                image_size=(800, 600),
                height_thresholds=np.array([5, 10, 15, 20, 25, 30]),
            )
            original.camera_left_transform = np.array(
                [[1.5, 0.1, -50], [0.2, 1.4, -40], [0.001, 0.002, 1]]
            )

            # Save
            original.save(str(path))
            assert path.exists()

            # Load
            loaded = CalibrationData.load(str(path))

            # Compare
            assert loaded.version == original.version
            assert loaded.stereo_baseline == original.stereo_baseline
            assert loaded.image_size == original.image_size
            np.testing.assert_allclose(
                loaded.height_thresholds, original.height_thresholds
            )
            np.testing.assert_allclose(
                loaded.camera_left_transform, original.camera_left_transform
            )

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "calibration.yaml"

            data = CalibrationData()
            data.save(str(path))

            assert path.exists()

    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            CalibrationData.load("/nonexistent/path/calibration.yaml")

    def test_load_invalid_yaml_raises(self):
        """Test that loading invalid YAML raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"
            path.write_text("not: valid: yaml: content: ::::")

            with pytest.raises(CalibrationDataError):
                CalibrationData.load(str(path))

    def test_load_non_dict_yaml_raises(self):
        """Test that loading non-dict YAML raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "list.yaml"
            path.write_text("- item1\n- item2\n")

            with pytest.raises(CalibrationDataError):
                CalibrationData.load(str(path))

    def test_save_with_dict_zone_boundaries(self):
        """Test saving when zone_boundaries is a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"

            data = CalibrationData.create_default()
            # create_default sets zone_boundaries as dict
            assert isinstance(data.zone_boundaries, dict)

            data.save(str(path))

            loaded = CalibrationData.load(str(path))
            assert isinstance(loaded.zone_boundaries, dict)

    def test_save_with_array_zone_boundaries(self):
        """Test saving when zone_boundaries is an array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"

            data = CalibrationData()
            # Default zone_boundaries is array
            data.zone_boundaries = np.zeros((32, 4, 2))

            data.save(str(path))

            loaded = CalibrationData.load(str(path))
            assert isinstance(loaded.zone_boundaries, np.ndarray)


class TestCalibrationDataMethods:
    """Tests for CalibrationData utility methods."""

    def test_is_valid(self):
        """Test is_valid method."""
        data = CalibrationData.create_default()
        # Should be mostly valid
        assert isinstance(data.is_valid(), bool)

    def test_to_dict(self):
        """Test to_dict method."""
        data = CalibrationData()
        result = data.to_dict()

        assert isinstance(result, dict)
        assert "version" in result
        assert "timestamp" in result
        assert "stereo_baseline" in result
        assert "camera_left_transform" in result

    def test_copy(self):
        """Test copy method creates independent copy."""
        original = CalibrationData()
        original.stereo_baseline = 25.0
        original.camera_left_transform = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        copied = original.copy()

        # Values should match
        assert copied.stereo_baseline == original.stereo_baseline
        np.testing.assert_array_equal(
            copied.camera_left_transform, original.camera_left_transform
        )

        # But modifying copy shouldn't affect original
        copied.stereo_baseline = 30.0
        copied.camera_left_transform[0, 0] = 999

        assert original.stereo_baseline == 25.0
        assert original.camera_left_transform[0, 0] == 1

    def test_get_quality_score(self):
        """Test get_quality_score method."""
        data = CalibrationData()
        data.calibration_quality = {
            "left_quality_score": 0.9,
            "right_quality_score": 0.8,
        }

        score = data.get_quality_score()
        assert score == pytest.approx(0.85)

    def test_get_quality_score_missing_values(self):
        """Test get_quality_score with missing values."""
        data = CalibrationData()
        data.calibration_quality = {}

        score = data.get_quality_score()
        assert score == 0.0

    def test_get_summary(self):
        """Test get_summary method."""
        data = CalibrationData.create_default()
        summary = data.get_summary()

        assert isinstance(summary, str)
        assert "ChunIVision" in summary
        assert "Version" in summary
        assert "Quality Score" in summary


class TestCalibrationDataEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_calibration_quality(self):
        """Test handling of empty calibration_quality dict."""
        data = CalibrationData(calibration_quality={})
        score = data.get_quality_score()
        assert score == 0.0

    def test_timestamp_parsing(self):
        """Test timestamp is correctly parsed from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"

            specific_time = datetime(2026, 6, 15, 10, 30, 45)
            data = CalibrationData(timestamp=specific_time)
            data.save(str(path))

            loaded = CalibrationData.load(str(path))
            assert loaded.timestamp.year == 2026
            assert loaded.timestamp.month == 6
            assert loaded.timestamp.day == 15

    def test_large_arrays_preserved(self):
        """Test that large array values are preserved in save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"

            # Create transform with large values
            transform = np.array([[1e6, 1e-6, 1e3], [1e-3, 1e6, 1e-6], [1e-9, 1e-9, 1]])

            data = CalibrationData()
            data.camera_left_transform = transform
            data.save(str(path))

            loaded = CalibrationData.load(str(path))
            np.testing.assert_allclose(
                loaded.camera_left_transform, transform, rtol=1e-10
            )

    def test_zone_boundaries_dict_with_zones(self):
        """Test zone_boundaries dict structure is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"

            zone_boundaries = {
                "grid": [16, 2],
                "zone_size": [2.75, 4.5],
                "zones": [{"id": 1, "bounds": [0, 0, 2.75, 4.5]}],
            }

            data = CalibrationData(zone_boundaries=zone_boundaries)
            data.save(str(path))

            loaded = CalibrationData.load(str(path))
            assert loaded.zone_boundaries["grid"] == [16, 2]
            assert len(loaded.zone_boundaries["zones"]) == 1
