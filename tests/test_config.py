"""
Unit tests for configuration modules.
"""

import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path

from chunivision.config import Settings, ZoneConfig, CameraConfig


class TestSettings:
    """Tests for Settings class."""

    def test_settings_creation(self):
        """Test Settings creation with defaults."""
        settings = Settings()
        assert settings is not None
        assert settings.app.name == "ChunIVision"
        assert settings.app.mode == "run"
        assert settings.logging.level == "INFO"

    def test_load_settings(self):
        """Test loading settings from YAML file."""
        config_data = {
            "app": {"name": "TestApp", "mode": "debug"},
            "logging": {"level": "DEBUG"},
            "camera": {"fps": 120},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            settings = Settings.load(str(config_file))
            assert settings.app.name == "TestApp"
            assert settings.app.mode == "debug"
            assert settings.logging.level == "DEBUG"
            assert settings.camera.fps == 120

    def test_save_settings(self):
        """Test saving settings to YAML file."""
        settings = Settings()
        settings.app.mode = "calibration"
        settings.camera.fps = 90

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "saved_config.yaml"
            settings.save(str(config_file))

            assert config_file.exists()

            # Load and verify
            loaded = Settings.load(str(config_file))
            assert loaded.app.mode == "calibration"
            assert loaded.camera.fps == 90

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        config_data = {"logging": {"level": "INFO"}, "camera": {"fps": 60}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Set environment variable
            monkeypatch.setenv("CHUNIVISION_LOGGING_LEVEL", "DEBUG")
            monkeypatch.setenv("CHUNIVISION_CAMERA_FPS", "120")

            settings = Settings.load(str(config_file))
            assert settings.logging.level == "DEBUG"
            assert settings.camera.fps == 120

    def test_validate_settings(self):
        """Test settings validation."""
        settings = Settings()
        errors = settings.validate()
        assert len(errors) == 0

        # Invalid mode
        settings.app.mode = "invalid"
        errors = settings.validate()
        assert len(errors) > 0
        assert any("mode" in err.lower() for err in errors)

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            Settings.load("nonexistent_config.yaml")


class TestZoneConfig:
    """Tests for ZoneConfig class."""

    def test_zone_config_creation(self):
        """Test ZoneConfig creation with defaults."""
        config = ZoneConfig()
        assert config.num_rows == 2
        assert config.num_cols == 16
        assert config.zone_width == 2.75
        assert config.zone_height == 4.5

    def test_get_zone_boundary(self):
        """Test getting zone boundary."""
        config = ZoneConfig()

        # Zone 1 (bottom-right)
        boundary = config.get_zone_boundary(1)
        assert boundary.shape == (4, 2)
        assert isinstance(boundary, np.ndarray)

        # Zone 32 (top-left)
        boundary32 = config.get_zone_boundary(32)
        assert boundary32.shape == (4, 2)

    def test_get_zone_center(self):
        """Test getting zone center."""
        config = ZoneConfig()

        center1 = config.get_zone_center(1)
        assert center1.shape == (2,)

        center32 = config.get_zone_center(32)
        assert center32.shape == (2,)

        # Zone 1 should be at bottom-right
        assert center1[0] > 0  # Positive X
        assert center1[1] < center32[1]  # Lower Y than zone 32

    def test_invalid_zone_id(self):
        """Test invalid zone IDs raise errors."""
        config = ZoneConfig()

        with pytest.raises(ValueError):
            config.get_zone_boundary(0)

        with pytest.raises(ValueError):
            config.get_zone_boundary(33)

        with pytest.raises(ValueError):
            config.get_zone_center(-1)

    def test_zone_id_to_grid_conversion(self):
        """Test zone ID to grid position conversion."""
        config = ZoneConfig()

        # Zone 1 (bottom-right) should be row 0, col 15
        row, col = config._zone_id_to_grid(1)
        assert row == 0
        assert col == 15

        # Zone 2 (top-right) should be row 1, col 15
        row, col = config._zone_id_to_grid(2)
        assert row == 1
        assert col == 15

        # Zone 31 (bottom-left) should be row 0, col 0
        row, col = config._zone_id_to_grid(31)
        assert row == 0
        assert col == 0

        # Zone 32 (top-left) should be row 1, col 0
        row, col = config._zone_id_to_grid(32)
        assert row == 1
        assert col == 0

    def test_grid_to_zone_id_conversion(self):
        """Test grid position to zone ID conversion."""
        config = ZoneConfig()

        # Bottom-right (row 0, col 15) -> zone 1
        assert config._grid_to_zone_id(0, 15) == 1

        # Top-right (row 1, col 15) -> zone 2
        assert config._grid_to_zone_id(1, 15) == 2

        # Bottom-left (row 0, col 0) -> zone 31
        assert config._grid_to_zone_id(0, 0) == 31

        # Top-left (row 1, col 0) -> zone 32
        assert config._grid_to_zone_id(1, 0) == 32

    def test_round_trip_conversion(self):
        """Test zone ID <-> grid conversion round trip."""
        config = ZoneConfig()

        for zone_id in range(1, 33):
            row, col = config._zone_id_to_grid(zone_id)
            recovered_id = config._grid_to_zone_id(row, col)
            assert recovered_id == zone_id

    def test_get_all_zone_centers(self):
        """Test getting all zone centers."""
        config = ZoneConfig()
        centers = config.get_all_zone_centers()

        assert centers.shape == (32, 2)
        # Verify zone 1 center matches individual query
        assert np.allclose(centers[0], config.get_zone_center(1))

    def test_get_all_zone_boundaries(self):
        """Test getting all zone boundaries."""
        config = ZoneConfig()
        boundaries = config.get_all_zone_boundaries()

        assert len(boundaries) == 32
        assert all(b.shape == (4, 2) for b in boundaries)

    def test_point_to_zone_id(self):
        """Test finding zone from point coordinates."""
        config = ZoneConfig()

        # Get center of zone 1 and verify it maps back to zone 1
        center1 = config.get_zone_center(1)
        zone_id = config.point_to_zone_id(center1[0], center1[1])
        assert zone_id == 1

        # Point outside grid should return 0
        zone_id = config.point_to_zone_id(1000, 1000)
        assert zone_id == 0

    def test_total_dimensions(self):
        """Test total grid dimensions."""
        config = ZoneConfig()

        total_width = config.get_total_width()
        assert total_width == config.num_cols * config.zone_width

        total_height = config.get_total_height()
        assert total_height == config.num_rows * config.zone_height


class TestCameraConfig:
    """Tests for CameraConfig class."""

    def test_camera_config_creation(self):
        """Test CameraConfig creation with defaults."""
        config = CameraConfig()
        assert config.left_camera_index == 0
        assert config.right_camera_index == 1
        assert config.resolution == (640, 480)
        assert config.fps == 60

    def test_camera_config_with_arrays(self):
        """Test CameraConfig with numpy arrays."""
        config = CameraConfig(
            left_camera_position=np.array([-15.0, 20.0, 35.0]),
            right_camera_position=np.array([15.0, 20.0, 35.0]),
        )

        assert isinstance(config.left_camera_position, np.ndarray)
        assert isinstance(config.right_camera_position, np.ndarray)

    def test_camera_config_with_lists(self):
        """Test CameraConfig converts lists to arrays."""
        config = CameraConfig(
            left_camera_position=[-15.0, 20.0, 35.0],
            right_camera_position=[15.0, 20.0, 35.0],
            resolution=[1920, 1080],
        )

        assert isinstance(config.left_camera_position, np.ndarray)
        assert isinstance(config.right_camera_position, np.ndarray)
        assert config.resolution == (1920, 1080)

    def test_validate_camera_config(self):
        """Test camera configuration validation."""
        config = CameraConfig()
        errors = config.validate()
        assert len(errors) == 0

        # Test that invalid config raises ValueError
        # We can't create invalid config directly because __post_init__ validates
        # So just verify the validate() method works correctly
        config_test = CameraConfig()
        # Temporarily modify to invalid state
        object.__setattr__(config_test, "fps", -1)
        errors = config_test.validate()
        assert len(errors) > 0
        assert any("FPS" in err for err in errors)

    def test_get_dimensions(self):
        """Test getting camera dimensions."""
        config = CameraConfig(resolution=(1920, 1080))

        assert config.get_width() == 1920
        assert config.get_height() == 1080
        assert abs(config.get_aspect_ratio() - 16 / 9) < 0.01

    def test_baseline_vector(self):
        """Test baseline vector calculation."""
        config = CameraConfig(
            left_camera_position=np.array([-10.0, 0.0, 0.0]),
            right_camera_position=np.array([10.0, 0.0, 0.0]),
        )

        baseline_vec = config.get_baseline_vector()
        expected = np.array([20.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(baseline_vec, expected)

    def test_calculated_baseline(self):
        """Test calculated baseline distance."""
        config = CameraConfig(
            left_camera_position=np.array([-10.0, 0.0, 0.0]),
            right_camera_position=np.array([10.0, 0.0, 0.0]),
            baseline_distance=20.0,
        )

        calculated = config.get_calculated_baseline()
        assert abs(calculated - 20.0) < 0.01

        assert config.check_baseline_consistency()

    def test_camera_center(self):
        """Test camera center calculation."""
        config = CameraConfig(
            left_camera_position=np.array([-10.0, 15.0, 30.0]),
            right_camera_position=np.array([10.0, 15.0, 30.0]),
        )

        center = config.get_camera_center()
        expected = np.array([0.0, 15.0, 30.0])
        np.testing.assert_array_almost_equal(center, expected)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CameraConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert "left_camera_index" in data
        assert "resolution" in data
        assert isinstance(data["resolution"], list)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "left_camera_index": 0,
            "right_camera_index": 1,
            "resolution": [1280, 720],
            "fps": 90,
            "left_camera_position": [-12.0, 18.0, 32.0],
            "right_camera_position": [12.0, 18.0, 32.0],
        }

        config = CameraConfig.from_dict(data)
        assert config.fps == 90
        assert config.resolution == (1280, 720)
        assert isinstance(config.left_camera_position, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
