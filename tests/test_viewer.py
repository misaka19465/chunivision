"""Tests for viewer distortion calibration."""

import pytest

from chunivision.oculus.viewer import DistortionCalibration


class TestDistortionCalibration:
    """Test suite for DistortionCalibration class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        calibration = DistortionCalibration()
        assert calibration.frame_size == (1280, 960)
        assert calibration.center == (655.052, 475.083)
        assert len(calibration.kappas) == 3
        assert len(calibration.rhos) == 2
        assert calibration.max_r2 == 0.0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        frame_size = (1920, 1080)
        center = (960.0, 540.0)
        kappas = (1e-7, 2e-13, 3e-19)
        rhos = (-1e-7, 1e-7)
        max_r2 = 100000.0

        calibration = DistortionCalibration(
            frame_size=frame_size,
            center=center,
            kappas=kappas,
            rhos=rhos,
            max_r2=max_r2,
        )

        assert calibration.frame_size == frame_size
        assert calibration.center == center
        assert calibration.kappas == kappas
        assert calibration.rhos == rhos
        assert calibration.max_r2 == max_r2

    def test_can_undistort(self):
        """Test undistortion validity check."""
        calibration = DistortionCalibration()
        # Set max_r2 to a known value for testing
        calibration.max_r2 = 655 * 655 + 475 * 475

        # Center pixel should be valid
        assert calibration.can_undistort((655.0, 475.0))
        # Far corner should be invalid (beyond max_r2)
        assert not calibration.can_undistort((0.0, 0.0))

    def test_can_undistort_invalid_pixel(self):
        """Test can_undistort with invalid pixel."""
        calibration = DistortionCalibration()

        # None pixel should return False
        assert not calibration.can_undistort(None)

        # Invalid tuple length should return False
        assert not calibration.can_undistort((1.0,))

    def test_undistort_basic(self):
        """Test basic undistortion."""
        calibration = DistortionCalibration()

        # Test center pixel (should be unchanged)
        center = (655.052, 475.083)
        result = calibration.undistort(center)
        assert abs(result[0] - center[0]) < 0.01
        assert abs(result[1] - center[1]) < 0.01

        # Test non-center pixel (should be different)
        pixel = (700.0, 500.0)
        result = calibration.undistort(pixel)
        # With the default radial distortion, result should differ
        assert result != pixel

    def test_undistort_invalid_pixel(self):
        """Test undistort with invalid pixel."""
        calibration = DistortionCalibration()

        with pytest.raises(ValueError, match="Pixel must be"):
            calibration.undistort(None)

        with pytest.raises(ValueError, match="Pixel must be"):
            calibration.undistort((1.0,))

    def test_distort_basic(self):
        """Test basic distortion."""
        calibration = DistortionCalibration()

        # Test center pixel (should be unchanged)
        center = (655.052, 475.083)
        result = calibration.distort(center)
        assert abs(result[0] - center[0]) < 0.01
        assert abs(result[1] - center[1]) < 0.01

        # Test non-center pixel (should be different)
        pixel = (700.0, 500.0)
        result = calibration.distort(pixel)
        # Result should be different and clamped within frame bounds
        assert result != pixel
        assert 0 <= result[0] < calibration.frame_size[0]
        assert 0 <= result[1] < calibration.frame_size[1]

    def test_distort_invalid_parameters(self):
        """Test distort with invalid parameters."""
        calibration = DistortionCalibration()

        with pytest.raises(ValueError, match="Pixel must be"):
            calibration.distort(None)

        with pytest.raises(ValueError, match="Pixel must be"):
            calibration.distort((1.0,))  # Only 1 element

        with pytest.raises(ValueError, match="max_iterations"):
            calibration.distort((100.0, 100.0), max_iterations=0)

        with pytest.raises(ValueError, match="Tolerance"):
            calibration.distort((100.0, 100.0), tolerance=0)

    def test_undistort_distort_roundtrip(self):
        """Test undistort/distort form approximate inverse."""
        calibration = DistortionCalibration()
        calibration.max_r2 = 655 * 655 + 475 * 475  # Allow all pixels

        # Test roundtrip for several points
        test_points = [
            (655.0, 475.0),  # Center
            (700.0, 500.0),
            (600.0, 450.0),
            (800.0, 600.0),
        ]

        for original in test_points:
            undistorted = calibration.undistort(original)
            back_to_distorted = calibration.distort(undistorted)

            # Should be close (within 1 pixel)
            assert abs(back_to_distorted[0] - original[0]) < 1.0
            assert abs(back_to_distorted[1] - original[1]) < 1.0

    def test_distort_undistort_roundtrip(self):
        """Test distort/undistort roundtrip."""
        calibration = DistortionCalibration()

        # Test points in undistorted space
        test_points = [
            (655.052, 475.083),  # Center
            (700.0, 500.0),
            (600.0, 450.0),
            (800.0, 600.0),
        ]

        for original in test_points:
            distorted = calibration.distort(original)
            back_to_undistorted = calibration.undistort(distorted)

            # Should be close (within 1 pixel)
            assert abs(back_to_undistorted[0] - original[0]) < 1.0
            assert abs(back_to_undistorted[1] - original[1]) < 1.0

    def test_distort_clamping(self):
        """Test that distort clamps to valid image bounds."""
        calibration = DistortionCalibration(frame_size=(1280, 960))

        # Test extreme points that might distort outside bounds
        extreme_points = [
            (0.0, 0.0),
            (1279.0, 959.0),
            (0.0, 959.0),
            (1279.0, 0.0),
        ]

        for point in extreme_points:
            result = calibration.distort(point)
            # Result should be clamped within valid bounds
            assert 0 <= result[0] < 1280
            assert 0 <= result[1] < 960

    def test_distort_convergence(self):
        """Test distort convergence with different iteration counts."""
        calibration = DistortionCalibration()

        pixel = (700.0, 500.0)

        # With more iterations, should get better accuracy
        result_5 = calibration.distort(pixel, max_iterations=5)
        result_20 = calibration.distort(pixel, max_iterations=20)

        # Both should converge to similar results
        assert abs(result_5[0] - result_20[0]) < 0.1
        assert abs(result_5[1] - result_20[1]) < 0.1

    def test_distort_tolerance(self):
        """Test distort with different tolerance values."""
        calibration = DistortionCalibration()

        pixel = (700.0, 500.0)

        # With tighter tolerance, should iterate more
        result_loose = calibration.distort(pixel, tolerance=1e-3)
        result_tight = calibration.distort(pixel, tolerance=1e-9)

        # Both should converge to similar results
        assert abs(result_loose[0] - result_tight[0]) < 0.01
        assert abs(result_loose[1] - result_tight[1]) < 0.01
