"""
Tests for lens distortion correction module.
"""

import numpy as np
import pytest

from chunivision.calibration.lens_distortion import (
    LensDistortion,
    LensDistortionParams,
)


class TestLensDistortionParams:
    """Tests for LensDistortionParams."""

    def test_default_params(self):
        """Test default Oculus CV1 parameters."""
        params = LensDistortionParams.get_default()

        assert params.frame_size == (1280, 960)
        assert params.center == (655.052, 475.083)
        assert len(params.kappas) == 3
        assert len(params.rhos) == 2

    def test_from_camera_params(self):
        """Test creating params from camera calibration dict."""
        cal_params = {
            "frame_size": (640, 480),
            "cx": 320.0,
            "cy": 240.0,
            "k": [1e-6, 2e-12, 3e-18],
            "max_r2": 100000.0,
        }

        params = LensDistortionParams.from_camera_params(cal_params)

        assert params.frame_size == (640, 480)
        assert params.center == (320.0, 240.0)
        assert params.kappas[0] == 1e-6
        assert params.max_r2 == 100000.0

    def test_to_opencv_params(self):
        """Test converting to OpenCV format."""
        params = LensDistortionParams()
        camera_matrix, dist_coeffs = params.to_opencv_params()

        assert camera_matrix.shape == (3, 3)
        assert dist_coeffs.shape == (5,)

        # Check camera matrix structure
        assert camera_matrix[0, 0] > 0  # fx
        assert camera_matrix[1, 1] > 0  # fy
        assert camera_matrix[2, 2] == 1.0


class TestLensDistortion:
    """Tests for LensDistortion class."""

    def test_default_creation(self):
        """Test creating with default parameters."""
        distortion = LensDistortion()

        assert distortion.frame_size == (1280, 960)
        assert distortion.center == (655.052, 475.083)

    def test_custom_params_creation(self):
        """Test creating with custom parameters."""
        params = LensDistortionParams(
            frame_size=(640, 480),
            center=(320.0, 240.0),
        )
        distortion = LensDistortion(params)

        assert distortion.frame_size == (640, 480)
        assert distortion.center == (320.0, 240.0)

    def test_undistort_center_point(self):
        """Test that center point is unchanged by undistortion."""
        distortion = LensDistortion()
        center = distortion.center

        undistorted = distortion.undistort_point(center)

        # Center should be nearly unchanged
        np.testing.assert_array_almost_equal(undistorted, center, decimal=5)

    def test_undistort_point(self):
        """Test undistorting a point away from center."""
        distortion = LensDistortion()

        # Point away from center
        point = (400.0, 300.0)
        undistorted = distortion.undistort_point(point)

        # Should be slightly different due to distortion
        assert undistorted != point
        # But not too far
        assert abs(undistorted[0] - point[0]) < 50
        assert abs(undistorted[1] - point[1]) < 50

    def test_distort_undistort_roundtrip(self):
        """Test that distort(undistort(p)) ≈ p."""
        distortion = LensDistortion()

        original = (400.0, 300.0)
        undistorted = distortion.undistort_point(original)
        redistorted = distortion.distort_point(undistorted)

        np.testing.assert_array_almost_equal(redistorted, original, decimal=3)

    def test_undistort_points_batch(self):
        """Test undistorting multiple points."""
        distortion = LensDistortion()

        points = np.array(
            [
                [400, 300],
                [500, 400],
                [600, 500],
            ],
            dtype=np.float64,
        )

        result = distortion.undistort_points(points)

        assert result.shape == (3, 2)
        # Each point should be close to original
        for i in range(3):
            assert abs(result[i, 0] - points[i, 0]) < 50
            assert abs(result[i, 1] - points[i, 1]) < 50

    def test_distort_points_batch(self):
        """Test distorting multiple points."""
        distortion = LensDistortion()

        points = np.array(
            [
                [400, 300],
                [500, 400],
            ],
            dtype=np.float64,
        )

        result = distortion.distort_points(points)

        assert result.shape == (2, 2)

    def test_undistort_single_point_array(self):
        """Test undistorting single point as 1D array."""
        distortion = LensDistortion()

        point = np.array([400.0, 300.0])
        result = distortion.undistort_points(point)

        assert result.ndim == 1
        assert len(result) == 2

    def test_can_undistort(self):
        """Test can_undistort check."""
        distortion = LensDistortion()

        # Valid point
        assert distortion.can_undistort((400, 300)) == True

        # Invalid inputs
        assert distortion.can_undistort(None) == False
        assert distortion.can_undistort((400,)) == False

    def test_undistort_invalid_pixel(self):
        """Test undistort with invalid input raises error."""
        distortion = LensDistortion()

        with pytest.raises(ValueError):
            distortion.undistort_point(None)

        with pytest.raises(ValueError):
            distortion.undistort_point((100,))

    def test_distort_invalid_params(self):
        """Test distort with invalid parameters raises error."""
        distortion = LensDistortion()

        with pytest.raises(ValueError):
            distortion.distort_point((400, 300), max_iterations=0)

        with pytest.raises(ValueError):
            distortion.distort_point((400, 300), tolerance=0)


class TestLensDistortionFrame:
    """Tests for frame-level distortion correction."""

    def test_compute_remap_tables(self):
        """Test computing remap tables."""
        params = LensDistortionParams(frame_size=(320, 240))
        distortion = LensDistortion(params)

        distortion.compute_remap_tables()

        assert distortion._maps_computed
        assert distortion._map_x is not None
        assert distortion._map_y is not None
        assert distortion._map_x.shape == (240, 320)
        assert distortion._map_y.shape == (240, 320)

    def test_undistort_frame(self):
        """Test undistorting an entire frame."""
        params = LensDistortionParams(frame_size=(320, 240))
        distortion = LensDistortion(params)

        # Create test frame
        frame = np.random.randint(0, 256, (240, 320), dtype=np.uint8)

        result = distortion.undistort_frame(frame)

        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_undistort_frame_lazy_computation(self):
        """Test that remap tables are computed lazily."""
        params = LensDistortionParams(frame_size=(160, 120))
        distortion = LensDistortion(params)

        assert not distortion._maps_computed

        frame = np.zeros((120, 160), dtype=np.uint8)
        distortion.undistort_frame(frame)

        assert distortion._maps_computed

    def test_get_opencv_undistort_maps(self):
        """Test getting OpenCV-compatible undistortion maps."""
        distortion = LensDistortion()

        map_x, map_y = distortion.get_opencv_undistort_maps()

        assert map_x.shape == (960, 1280)
        assert map_y.shape == (960, 1280)
        assert map_x.dtype == np.float32
        assert map_y.dtype == np.float32
