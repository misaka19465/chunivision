"""
Tests for StereoProcessor module.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chunivision.calibration.calibration_data import CalibrationData
from chunivision.vision.point_cloud import PointCloud3D
from chunivision.vision.stereo_processor import (
    StereoAlgorithm,
    StereoConfig,
    StereoProcessor,
)


class TestStereoConfig:
    """Tests for StereoConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StereoConfig()

        assert config.algorithm == StereoAlgorithm.SGBM
        assert config.num_disparities == 128
        assert config.block_size == 5
        assert config.min_depth == 0.0
        assert config.max_depth == 35.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StereoConfig(
            algorithm=StereoAlgorithm.BM,
            num_disparities=64,
            block_size=7,
            min_depth=5.0,
            max_depth=50.0,
        )

        assert config.algorithm == StereoAlgorithm.BM
        assert config.num_disparities == 64
        assert config.block_size == 7
        assert config.min_depth == 5.0
        assert config.max_depth == 50.0

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = StereoConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_num_disparities(self):
        """Test validation catches invalid num_disparities."""
        config = StereoConfig(num_disparities=100)  # Not divisible by 16
        errors = config.validate()
        assert len(errors) == 1
        assert "num_disparities" in errors[0]

    def test_validate_invalid_block_size_even(self):
        """Test validation catches even block size."""
        config = StereoConfig(block_size=6)
        errors = config.validate()
        assert len(errors) == 1
        assert "block_size" in errors[0]

    def test_validate_invalid_block_size_too_small(self):
        """Test validation catches too small block size."""
        config = StereoConfig(block_size=1)
        errors = config.validate()
        assert len(errors) == 1
        assert "block_size" in errors[0]

    def test_validate_invalid_depth_range(self):
        """Test validation catches invalid depth range."""
        config = StereoConfig(min_depth=30.0, max_depth=10.0)
        errors = config.validate()
        assert len(errors) == 1
        assert "max_depth" in errors[0]

    def test_validate_negative_min_depth(self):
        """Test validation catches negative min_depth."""
        config = StereoConfig(min_depth=-5.0)
        errors = config.validate()
        assert len(errors) == 1
        assert "min_depth" in errors[0]


class TestCalibrationData:
    """Tests for CalibrationData class."""

    def test_default_calibration(self):
        """Test default calibration data creation."""
        calib = CalibrationData()

        assert calib.version == "1.0"
        assert calib.stereo_baseline == 20.0
        assert calib.image_size == (640, 480)
        assert calib.camera_left_matrix.shape == (3, 3)
        assert calib.camera_right_matrix.shape == (3, 3)

    def test_create_default(self):
        """Test create_default factory method."""
        calib = CalibrationData.create_default(image_size=(1280, 960))

        assert calib.image_size == (1280, 960)
        assert calib.camera_left_matrix[0, 0] == 1280  # fx
        assert calib.camera_left_matrix[1, 1] == 1280  # fy
        assert calib.camera_left_matrix[0, 2] == 640  # cx
        assert calib.camera_left_matrix[1, 2] == 480  # cy

    def test_validate_valid(self):
        """Test validation of valid calibration data."""
        calib = CalibrationData.create_default()
        errors = calib.validate()
        assert len(errors) == 0
        assert calib.is_valid()

    def test_validate_invalid_camera_matrix(self):
        """Test validation catches invalid camera matrix shape."""
        calib = CalibrationData()
        calib.camera_left_matrix = np.eye(4)  # Wrong shape
        errors = calib.validate()
        assert any("camera_left_matrix" in e for e in errors)

    def test_validate_invalid_rotation_matrix(self):
        """Test validation catches non-orthonormal rotation matrix."""
        calib = CalibrationData()
        calib.rotation_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])  # det != 1
        errors = calib.validate()
        assert any("rotation_matrix" in e for e in errors)

    def test_validate_invalid_height_thresholds(self):
        """Test validation catches wrong number of height thresholds."""
        calib = CalibrationData()
        calib.height_thresholds = np.array([1, 2, 3])  # Should be 6
        errors = calib.validate()
        assert any("height_thresholds" in e for e in errors)

    def test_validate_unordered_height_thresholds(self):
        """Test validation catches unordered height thresholds."""
        calib = CalibrationData()
        calib.height_thresholds = np.array([30, 25, 20, 15, 10, 5])  # Descending
        errors = calib.validate()
        assert any("ascending order" in e for e in errors)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading calibration data."""
        calib = CalibrationData.create_default()
        calib.stereo_baseline = 25.0

        path = tmp_path / "test_calibration.yaml"
        calib.save(str(path))

        loaded = CalibrationData.load(str(path))

        assert loaded.version == calib.version
        assert loaded.stereo_baseline == 25.0
        np.testing.assert_array_almost_equal(
            loaded.camera_left_matrix, calib.camera_left_matrix
        )

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            CalibrationData.load("/nonexistent/path.yaml")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        calib = CalibrationData.create_default()
        data = calib.to_dict()

        assert "version" in data
        assert "camera_left_matrix" in data
        assert "stereo_baseline" in data


class TestPointCloud3D:
    """Tests for PointCloud3D class."""

    def test_empty_point_cloud(self):
        """Test empty point cloud creation."""
        cloud = PointCloud3D(points=np.empty((0, 3)))

        assert cloud.num_points == 0
        assert cloud.is_empty
        assert cloud.centroid.tolist() == [0, 0, 0]

    def test_point_cloud_from_array(self):
        """Test creating point cloud from array."""
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        cloud = PointCloud3D(points=points)

        assert cloud.num_points == 3
        assert not cloud.is_empty
        np.testing.assert_array_almost_equal(cloud.centroid, [4, 5, 6])

    def test_bounds(self):
        """Test bounding box calculation."""
        points = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float64)
        cloud = PointCloud3D(points=points)

        min_b, max_b = cloud.bounds
        np.testing.assert_array_equal(min_b, [0, 0, 0])
        np.testing.assert_array_equal(max_b, [10, 20, 30])

    def test_filter_by_depth(self):
        """Test filtering by Z coordinate."""
        points = np.array(
            [[0, 0, 5], [0, 0, 15], [0, 0, 25], [0, 0, 35]], dtype=np.float64
        )
        cloud = PointCloud3D(points=points)

        filtered = cloud.filter_by_depth(10, 30)

        assert filtered.num_points == 2
        assert 15 in filtered.points[:, 2]
        assert 25 in filtered.points[:, 2]

    def test_filter_by_region(self):
        """Test filtering by spatial region."""
        points = np.array(
            [[0, 0, 0], [5, 5, 5], [10, 10, 10], [15, 15, 15]], dtype=np.float64
        )
        cloud = PointCloud3D(points=points)

        filtered = cloud.filter_by_region(
            x_range=(3, 12), y_range=(3, 12), z_range=(3, 12)
        )

        assert filtered.num_points == 2
        assert 5 in filtered.points[:, 0]
        assert 10 in filtered.points[:, 0]

    def test_downsample(self):
        """Test voxel downsampling."""
        # Create a dense point cloud
        points = np.random.rand(1000, 3) * 10  # Points in 10x10x10 cube
        cloud = PointCloud3D(points=points)

        downsampled = cloud.downsample(voxel_size=1.0)

        # Should have fewer points after downsampling
        assert downsampled.num_points < cloud.num_points
        # Max possible voxels in 10x10x10 cube with 1cm voxels is 1000
        assert downsampled.num_points <= 1000

    def test_transform(self):
        """Test rigid transformation."""
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        cloud = PointCloud3D(points=points)

        # 90 degree rotation around Z axis
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        translation = np.array([1, 2, 3], dtype=np.float64)

        transformed = cloud.transform(rotation, translation)

        # Check first point: [1, 0, 0] -> R @ [1, 0, 0] + t = [0, 1, 0] + [1, 2, 3] = [1, 3, 3]
        np.testing.assert_array_almost_equal(transformed.points[0], [1, 3, 3])

    def test_merge(self):
        """Test merging two point clouds."""
        cloud1 = PointCloud3D(
            points=np.array([[1, 2, 3]], dtype=np.float64), frame_id=1
        )
        cloud2 = PointCloud3D(
            points=np.array([[4, 5, 6]], dtype=np.float64), frame_id=2
        )

        merged = cloud1.merge(cloud2)

        assert merged.num_points == 2
        assert merged.frame_id == 2

    def test_get_statistics(self):
        """Test statistics calculation."""
        points = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64)
        cloud = PointCloud3D(points=points)

        stats = cloud.get_statistics()

        assert stats["num_points"] == 2
        assert stats["centroid"] == [5, 5, 5]
        assert stats["bounds_min"] == [0, 0, 0]
        assert stats["bounds_max"] == [10, 10, 10]


class TestStereoProcessor:
    """Tests for StereoProcessor class."""

    @pytest.fixture
    def default_calibration(self):
        """Create default calibration data for testing."""
        return CalibrationData.create_default(image_size=(640, 480))

    @pytest.fixture
    def stereo_processor(self, default_calibration):
        """Create stereo processor for testing."""
        return StereoProcessor(default_calibration)

    def test_initialization(self, default_calibration):
        """Test processor initialization."""
        processor = StereoProcessor(default_calibration)

        assert processor.calibration_data is default_calibration
        assert processor.config.algorithm == StereoAlgorithm.SGBM
        assert processor._frame_count == 0

    def test_initialization_with_config(self, default_calibration):
        """Test processor initialization with custom config."""
        config = StereoConfig(
            algorithm=StereoAlgorithm.BM, num_disparities=64, block_size=7
        )
        processor = StereoProcessor(default_calibration, config=config)

        assert processor.config.algorithm == StereoAlgorithm.BM
        assert processor.config.num_disparities == 64

    def test_initialization_invalid_config(self, default_calibration):
        """Test processor initialization with invalid config raises error."""
        config = StereoConfig(num_disparities=100)  # Invalid

        with pytest.raises(ValueError):
            StereoProcessor(default_calibration, config=config)

    def test_rectify_images(self, stereo_processor):
        """Test image rectification."""
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        left_rect, right_rect = stereo_processor.rectify_images(left, right)

        assert left_rect.shape == (480, 640)
        assert right_rect.shape == (480, 640)

    def test_compute_disparity(self, stereo_processor):
        """Test disparity computation."""
        # Create simple test images
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        disparity = stereo_processor.compute_disparity(left, right)

        assert disparity.shape == (480, 640)
        assert disparity.dtype == np.float32

    def test_compute_disparity_color_images(self, stereo_processor):
        """Test disparity computation with color images."""
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        disparity = stereo_processor.compute_disparity(left, right)

        assert disparity.shape == (480, 640)

    def test_get_depth_map(self, stereo_processor):
        """Test depth map generation."""
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        depth = stereo_processor.get_depth_map(left, right)

        assert depth.shape == (480, 640)
        assert stereo_processor.get_processing_time_ms() > 0

    def test_process(self, stereo_processor):
        """Test full processing pipeline."""
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        cloud = stereo_processor.process(left, right)

        assert isinstance(cloud, PointCloud3D)
        assert cloud.frame_id == 1
        assert cloud.timestamp > 0

    def test_process_increments_frame_count(self, stereo_processor):
        """Test that processing increments frame counter."""
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        stereo_processor.process(left, right)
        cloud = stereo_processor.process(left, right)

        assert cloud.frame_id == 2
        assert stereo_processor._frame_count == 2

    def test_update_calibration(self, stereo_processor):
        """Test calibration update."""
        new_calib = CalibrationData.create_default(image_size=(1280, 960))
        stereo_processor.update_calibration(new_calib)

        assert stereo_processor.calibration_data is new_calib

    def test_set_depth_range(self, stereo_processor):
        """Test setting depth range."""
        stereo_processor.set_depth_range(5.0, 50.0)

        assert stereo_processor.config.min_depth == 5.0
        assert stereo_processor.config.max_depth == 50.0

    def test_set_depth_range_invalid(self, stereo_processor):
        """Test setting invalid depth range raises error."""
        with pytest.raises(ValueError):
            stereo_processor.set_depth_range(50.0, 5.0)

    def test_set_algorithm(self, stereo_processor):
        """Test changing stereo algorithm."""
        stereo_processor.set_algorithm(StereoAlgorithm.BM)

        assert stereo_processor.config.algorithm == StereoAlgorithm.BM

    def test_set_num_disparities(self, stereo_processor):
        """Test changing number of disparities."""
        stereo_processor.set_num_disparities(64)

        assert stereo_processor.config.num_disparities == 64

    def test_set_num_disparities_invalid(self, stereo_processor):
        """Test setting invalid num_disparities raises error."""
        with pytest.raises(ValueError):
            stereo_processor.set_num_disparities(100)

    def test_set_block_size(self, stereo_processor):
        """Test changing block size."""
        stereo_processor.set_block_size(7)

        assert stereo_processor.config.block_size == 7

    def test_set_block_size_invalid(self, stereo_processor):
        """Test setting invalid block size raises error."""
        with pytest.raises(ValueError):
            stereo_processor.set_block_size(6)

    def test_get_stats(self, stereo_processor):
        """Test getting processing statistics."""
        stats = stereo_processor.get_stats()

        assert "algorithm" in stats
        assert "num_disparities" in stats
        assert "depth_range" in stats
        assert stats["algorithm"] == "SGBM"

    def test_visualize_disparity(self, stereo_processor):
        """Test disparity visualization."""
        disparity = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
        disparity[0:100, 0:100] = np.nan  # Some invalid regions

        vis = stereo_processor.visualize_disparity(disparity)

        assert vis.shape == (480, 640, 3)
        assert vis.dtype == np.uint8

    def test_visualize_depth(self, stereo_processor):
        """Test depth visualization."""
        depth = np.random.uniform(0, 35, (480, 640)).astype(np.float32)
        depth[0:100, 0:100] = np.nan  # Some invalid regions

        vis = stereo_processor.visualize_depth(depth)

        assert vis.shape == (480, 640, 3)
        assert vis.dtype == np.uint8

    def test_visualize_empty_disparity(self, stereo_processor):
        """Test visualization of all-NaN disparity map."""
        disparity = np.full((480, 640), np.nan, dtype=np.float32)

        vis = stereo_processor.visualize_disparity(disparity)

        assert vis.shape == (480, 640, 3)
        assert np.all(vis == 0)


class TestStereoProcessorWithSyntheticData:
    """Tests using synthetic stereo images."""

    @pytest.fixture
    def stereo_pair(self):
        """Create synthetic stereo pair with known disparity."""
        # Create a simple pattern that would produce known disparities
        h, w = 480, 640

        # Left image: vertical stripes
        left = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, w, 20):
            left[:, i : i + 10] = 255

        # Right image: same pattern shifted (simulating disparity)
        shift = 50  # 50 pixel disparity
        right = np.zeros((h, w), dtype=np.uint8)
        for i in range(shift, w, 20):
            right[:, i : i + 10] = 255

        return left, right

    def test_disparity_computation_runs(self, stereo_pair):
        """Test disparity computation runs without error on synthetic stereo pair."""
        left, right = stereo_pair
        calib = CalibrationData.create_default(image_size=(640, 480))
        processor = StereoProcessor(calib)

        # The synthetic images may not produce valid disparities, but the
        # computation should run without errors
        disparity = processor.compute_disparity(left, right, rectify=False)

        # Should return a disparity map of correct shape
        assert disparity.shape == (480, 640)
        assert disparity.dtype == np.float32


class TestStereoProcessorBMAlgorithm:
    """Tests specifically for BM algorithm."""

    @pytest.fixture
    def bm_processor(self):
        """Create processor with BM algorithm."""
        calib = CalibrationData.create_default()
        config = StereoConfig(algorithm=StereoAlgorithm.BM, block_size=15)
        return StereoProcessor(calib, config=config)

    def test_bm_initialization(self, bm_processor):
        """Test BM processor initialization."""
        assert bm_processor.config.algorithm == StereoAlgorithm.BM

    def test_bm_compute_disparity(self, bm_processor):
        """Test BM disparity computation."""
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        disparity = bm_processor.compute_disparity(left, right)

        assert disparity.shape == (480, 640)

    def test_bm_block_size_minimum(self):
        """Test BM requires block_size >= 5."""
        calib = CalibrationData.create_default()
        config = StereoConfig(algorithm=StereoAlgorithm.BM, block_size=3)

        errors = config.validate()
        assert any("block_size" in e for e in errors)
