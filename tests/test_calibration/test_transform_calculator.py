"""
Tests for transform_calculator module.
"""

import numpy as np
import pytest
from chunivision.calibration.transform_calculator import TransformCalculator


class TestCalculatePerspectiveTransform:
    """Tests for calculate_perspective_transform method."""

    def test_identity_transform(self):
        """Test that identical points produce identity-like transform."""
        points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

        transform = TransformCalculator.calculate_perspective_transform(points, points)

        # Transform should be close to identity
        assert transform.shape == (3, 3)
        # When points are the same, transform maps to itself
        transformed = TransformCalculator.apply_transform(points, transform)
        np.testing.assert_allclose(transformed, points, atol=1e-5)

    def test_simple_translation(self):
        """Test transform with pure translation."""
        image_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        world_pts = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float32)

        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        # Verify transform produces correct output
        transformed = TransformCalculator.apply_transform(image_pts, transform)
        np.testing.assert_allclose(transformed, world_pts, atol=1e-4)

    def test_scaling_transform(self):
        """Test transform with scaling."""
        image_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        world_pts = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32)

        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        transformed = TransformCalculator.apply_transform(image_pts, transform)
        np.testing.assert_allclose(transformed, world_pts, atol=1e-4)

    def test_rotation_transform(self):
        """Test transform with rotation."""
        # Square
        image_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        # Rotated 90 degrees counter-clockwise
        world_pts = np.array([[0, 0], [0, 10], [-10, 10], [-10, 0]], dtype=np.float32)

        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        transformed = TransformCalculator.apply_transform(image_pts, transform)
        np.testing.assert_allclose(transformed, world_pts, atol=1e-4)

    def test_complex_perspective(self):
        """Test with trapezoidal perspective."""
        # Rectangle in image
        image_pts = np.array(
            [[10, 10], [100, 10], [100, 50], [10, 50]], dtype=np.float32
        )
        # Trapezoid in world (perspective effect)
        world_pts = np.array([[0, 0], [20, 0], [18, 10], [2, 10]], dtype=np.float32)

        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        transformed = TransformCalculator.apply_transform(image_pts, transform)
        np.testing.assert_allclose(transformed, world_pts, atol=1e-3)

    def test_more_than_4_points(self):
        """Test with more than 4 points (uses RANSAC)."""
        # 6 points
        image_pts = np.array(
            [[0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10]], dtype=np.float32
        )
        world_pts = np.array(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32
        )

        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        # Should still produce reasonable transform
        assert transform.shape == (3, 3)
        transformed = TransformCalculator.apply_transform(image_pts, transform)
        # RANSAC may have some error, but should be close
        np.testing.assert_allclose(transformed, world_pts, atol=0.5)

    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched point arrays raise error."""
        image_pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        world_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

        with pytest.raises(ValueError, match="must have same shape"):
            TransformCalculator.calculate_perspective_transform(image_pts, world_pts)

    def test_too_few_points_raises_error(self):
        """Test that fewer than 4 points raises error."""
        image_pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        world_pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)

        with pytest.raises(ValueError, match="at least 4 points"):
            TransformCalculator.calculate_perspective_transform(image_pts, world_pts)

    def test_invalid_dimensions_raises_error(self):
        """Test that points with wrong dimensions raise error."""
        # 3D points instead of 2D
        image_pts = np.array(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float32
        )
        world_pts = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )

        with pytest.raises(ValueError, match="must be shape"):
            TransformCalculator.calculate_perspective_transform(image_pts, world_pts)

    def test_float64_output(self):
        """Test that output is float64."""
        image_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        world_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        assert transform.dtype == np.float64


class TestCalculateInverseTransform:
    """Tests for calculate_inverse_transform method."""

    def test_identity_inverse(self):
        """Test inverse of identity is identity."""
        identity = np.eye(3, dtype=np.float64)

        inverse = TransformCalculator.calculate_inverse_transform(identity)

        np.testing.assert_allclose(inverse, identity, atol=1e-10)

    def test_inverse_of_inverse(self):
        """Test that inverse of inverse is original."""
        # Create a random invertible transform
        transform = np.array([[2, 1, 3], [1, 3, 2], [0.1, 0.2, 1]], dtype=np.float64)

        inverse = TransformCalculator.calculate_inverse_transform(transform)
        inverse_inverse = TransformCalculator.calculate_inverse_transform(inverse)

        np.testing.assert_allclose(inverse_inverse, transform, atol=1e-10)

    def test_inverse_composition(self):
        """Test that transform * inverse = identity."""
        transform = np.array(
            [[2, 0.5, 10], [0.5, 2, 20], [0.01, 0.02, 1]], dtype=np.float64
        )

        inverse = TransformCalculator.calculate_inverse_transform(transform)
        product = transform @ inverse

        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_singular_matrix_raises_error(self):
        """Test that singular matrix raises error."""
        # Singular matrix (determinant = 0)
        singular = np.array([[1, 2, 3], [2, 4, 6], [0, 0, 1]], dtype=np.float64)

        with pytest.raises(ValueError, match="singular"):
            TransformCalculator.calculate_inverse_transform(singular)

    def test_wrong_shape_raises_error(self):
        """Test that wrong shape raises error."""
        wrong_shape = np.array([[1, 2], [3, 4]], dtype=np.float64)

        with pytest.raises(ValueError, match="must be 3x3"):
            TransformCalculator.calculate_inverse_transform(wrong_shape)

    def test_float64_output(self):
        """Test that output is float64."""
        transform = np.eye(3, dtype=np.float32)

        inverse = TransformCalculator.calculate_inverse_transform(transform)

        assert inverse.dtype == np.float64


class TestApplyTransform:
    """Tests for apply_transform method."""

    def test_identity_transform_no_change(self):
        """Test identity transform doesn't change points."""
        points = np.array([[0, 0], [10, 5], [20, 30]], dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)

        transformed = TransformCalculator.apply_transform(points, identity)

        np.testing.assert_allclose(transformed, points, atol=1e-10)

    def test_translation_transform(self):
        """Test simple translation."""
        points = np.array([[0, 0], [10, 10]], dtype=np.float64)
        # Translation by (5, 3)
        transform = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], dtype=np.float64)

        transformed = TransformCalculator.apply_transform(points, transform)
        expected = np.array([[5, 3], [15, 13]], dtype=np.float64)

        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_scaling_transform(self):
        """Test scaling transform."""
        points = np.array([[1, 2], [3, 4]], dtype=np.float64)
        # Scale by 2x in both directions
        transform = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)

        transformed = TransformCalculator.apply_transform(points, transform)
        expected = np.array([[2, 4], [6, 8]], dtype=np.float64)

        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_single_point(self):
        """Test transform on single point."""
        point = np.array([5, 10], dtype=np.float64)
        transform = np.array([[2, 0, 1], [0, 2, 2], [0, 0, 1]], dtype=np.float64)

        transformed = TransformCalculator.apply_transform(point, transform)
        expected = np.array([11, 22], dtype=np.float64)

        assert transformed.shape == (2,)
        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_perspective_division(self):
        """Test proper perspective division."""
        points = np.array([[10, 20]], dtype=np.float64)
        # Transform with perspective component
        transform = np.array(
            [[1, 0, 0], [0, 1, 0], [0.1, 0, 1]], dtype=np.float64  # perspective in x
        )

        transformed = TransformCalculator.apply_transform(points, transform)

        # Expected: w = 0.1*10 + 1 = 2
        # x' = 10/2 = 5, y' = 20/2 = 10
        expected = np.array([[5, 10]], dtype=np.float64)
        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_multiple_points(self):
        """Test transform on multiple points."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        transform = np.array([[10, 0, 5], [0, 10, 3], [0, 0, 1]], dtype=np.float64)

        transformed = TransformCalculator.apply_transform(points, transform)
        expected = np.array([[5, 3], [15, 3], [15, 13], [5, 13]], dtype=np.float64)

        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_wrong_transform_shape_raises_error(self):
        """Test that wrong transform shape raises error."""
        points = np.array([[0, 0]], dtype=np.float64)
        wrong_transform = np.array([[1, 0], [0, 1]], dtype=np.float64)

        with pytest.raises(ValueError, match="must be 3x3"):
            TransformCalculator.apply_transform(points, wrong_transform)

    def test_wrong_point_dimensions_raises_error(self):
        """Test that wrong point dimensions raise error."""
        # 3D points
        points = np.array([[0, 0, 0]], dtype=np.float64)
        transform = np.eye(3, dtype=np.float64)

        with pytest.raises(ValueError, match="must be shape"):
            TransformCalculator.apply_transform(points, transform)

    def test_points_at_infinity_raises_error(self):
        """Test that points at infinity raise error."""
        points = np.array([[1, 1]], dtype=np.float64)
        # Transform that produces w=0
        transform = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],  # w = -x + 0 = -1*1 = -1... wait, need w=0
            ],
            dtype=np.float64,
        )
        transform[2, 0] = -1
        transform[2, 2] = 1
        # This won't produce w=0 for point [1,1]

        # Better: construct a transform that maps [1,1] to infinity
        # w = p1*x + p2*y + 1 = 0 for point [1,1]
        # So p1 + p2 + 1 = 0, so p1 = -1, p2 = 0 works
        transform = np.array(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0]],  # w = -x for x=1 gives w=-1 (not zero)
            dtype=np.float64,
        )

        # Actually create a degenerate case
        points = np.array([[0, 0]], dtype=np.float64)
        transform = np.array(
            [[1, 0, 0], [0, 1, 0], [1, 1, 0]],  # w = x + y = 0 for [0,0]
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="infinity"):
            TransformCalculator.apply_transform(points, transform)


class TestValidateTransformQuality:
    """Tests for validate_transform_quality method."""

    def test_perfect_transform(self):
        """Test quality metrics for perfect transform."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        expected_pts = test_pts.copy()

        quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_pts
        )

        assert quality == 1.0
        assert mean_err == 0.0
        assert max_err == 0.0

    def test_small_error_high_quality(self):
        """Test that small errors give high quality score."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [10, 10]], dtype=np.float64)
        # Add small error
        expected_pts = test_pts + 0.1

        quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_pts, max_error_threshold=5.0
        )

        assert 0.95 < quality <= 1.0
        assert 0.1 <= mean_err < 0.2
        assert 0.1 <= max_err < 0.2

    def test_large_error_low_quality(self):
        """Test that large errors give low quality score."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [10, 10]], dtype=np.float64)
        # Add large error
        expected_pts = test_pts + 10

        quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_pts, max_error_threshold=5.0
        )

        assert quality < 0.2
        assert mean_err > 10
        assert max_err > 10

    def test_quality_score_range(self):
        """Test that quality score is in [0, 1]."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
        expected_pts = test_pts + 100  # Very large error

        quality, _, _ = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_pts
        )

        assert 0.0 <= quality <= 1.0

    def test_mean_and_max_error_calculation(self):
        """Test correct calculation of mean and max error."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float64)
        # Errors: 1.0, 2.0, 5.0 (Euclidean distance)
        expected_pts = np.array([[1, 0], [12, 0], [10, 15]], dtype=np.float64)

        quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_pts
        )

        # Mean error should be (1.0 + 2.0 + 5.0) / 3 ≈ 2.67
        assert 2.6 < mean_err < 2.7
        # Max error should be 5.0
        assert 4.9 < max_err < 5.1

    def test_penalty_for_large_outlier(self):
        """Test that large outliers reduce quality score."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [1, 0]], dtype=np.float64)

        # Case 1: Small errors
        expected_small = test_pts + 0.1
        quality_small, _, _ = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_small, max_error_threshold=5.0
        )

        # Case 2: One large outlier (> 2x threshold)
        expected_outlier = np.array([[0.1, 0.1], [12, 0]], dtype=np.float64)
        quality_outlier, _, _ = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_outlier, max_error_threshold=5.0
        )

        # Outlier case should have much lower quality
        assert quality_outlier < quality_small

    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched shapes raise error."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
        expected_pts = np.array([[0, 0]], dtype=np.float64)

        with pytest.raises(ValueError, match="must have same shape"):
            TransformCalculator.validate_transform_quality(
                transform, test_pts, expected_pts
            )

    def test_empty_points_raises_error(self):
        """Test that empty point array raises error."""
        transform = np.eye(3, dtype=np.float64)
        test_pts = np.array([], dtype=np.float64).reshape(0, 2)
        expected_pts = np.array([], dtype=np.float64).reshape(0, 2)

        with pytest.raises(ValueError, match="at least one test point"):
            TransformCalculator.validate_transform_quality(
                transform, test_pts, expected_pts
            )

    def test_failed_transform_returns_zero_quality(self):
        """Test that failing transform returns zero quality."""
        # Degenerate transform
        transform = np.zeros((3, 3), dtype=np.float64)
        test_pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
        expected_pts = test_pts.copy()

        quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
            transform, test_pts, expected_pts
        )

        assert quality == 0.0
        assert mean_err == float("inf")
        assert max_err == float("inf")


class TestEstimateReprojectionError:
    """Tests for estimate_reprojection_error method."""

    def test_perfect_reprojection(self):
        """Test zero error for perfect reprojection."""
        transform = np.eye(3, dtype=np.float64)
        image_pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
        world_pts = image_pts.copy()

        error = TransformCalculator.estimate_reprojection_error(
            transform, image_pts, world_pts
        )

        assert error == 0.0

    def test_nonzero_reprojection_error(self):
        """Test non-zero error for imperfect reprojection."""
        transform = np.eye(3, dtype=np.float64)
        image_pts = np.array([[0, 0], [10, 0]], dtype=np.float64)
        world_pts = np.array([[1, 0], [11, 0]], dtype=np.float64)

        error = TransformCalculator.estimate_reprojection_error(
            transform, image_pts, world_pts
        )

        assert error == 1.0  # All points have error of 1.0

    def test_error_is_mean_distance(self):
        """Test that error is mean Euclidean distance."""
        transform = np.eye(3, dtype=np.float64)
        image_pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float64)
        # Errors: 1.0, 2.0, 5.0
        world_pts = np.array([[1, 0], [12, 0], [10, 15]], dtype=np.float64)

        error = TransformCalculator.estimate_reprojection_error(
            transform, image_pts, world_pts
        )

        expected_error = (1.0 + 2.0 + 5.0) / 3
        assert abs(error - expected_error) < 0.01


class TestDecomposeTransform:
    """Tests for decompose_transform method."""

    def test_identity_decomposition(self):
        """Test decomposition of identity transform."""
        identity = np.eye(3, dtype=np.float64)

        components = TransformCalculator.decompose_transform(identity)

        assert components["translation"] == (0.0, 0.0)
        assert components["scale"] == pytest.approx((1.0, 1.0), abs=1e-6)
        assert components["rotation"] == pytest.approx(0.0, abs=1e-6)
        assert components["shear"] == pytest.approx(0.0, abs=1e-6)
        assert components["perspective"] == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_translation_decomposition(self):
        """Test decomposition of pure translation."""
        transform = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float64)

        components = TransformCalculator.decompose_transform(transform)

        assert components["translation"] == pytest.approx((10.0, 20.0), abs=1e-6)
        assert components["scale"] == pytest.approx((1.0, 1.0), abs=1e-6)
        assert components["rotation"] == pytest.approx(0.0, abs=1e-6)

    def test_scaling_decomposition(self):
        """Test decomposition of scaling transform."""
        transform = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float64)

        components = TransformCalculator.decompose_transform(transform)

        assert components["scale"] == pytest.approx((2.0, 3.0), abs=1e-6)
        assert components["translation"] == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_rotation_decomposition(self):
        """Test decomposition of rotation transform."""
        # 90 degree rotation
        angle = np.pi / 2
        transform = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        components = TransformCalculator.decompose_transform(transform)

        assert components["rotation"] == pytest.approx(angle, abs=1e-6)
        assert components["scale"] == pytest.approx((1.0, 1.0), abs=1e-6)

    def test_perspective_decomposition(self):
        """Test decomposition with perspective components."""
        transform = np.array([[1, 0, 0], [0, 1, 0], [0.1, 0.2, 1]], dtype=np.float64)

        components = TransformCalculator.decompose_transform(transform)

        assert components["perspective"] == pytest.approx((0.1, 0.2), abs=1e-6)

    def test_combined_transform_decomposition(self):
        """Test decomposition of combined transform."""
        # Scale by 2, rotate 45 degrees, translate by (5, 10)
        angle = np.pi / 4
        scale = 2.0
        transform = np.array(
            [
                [scale * np.cos(angle), -scale * np.sin(angle), 5],
                [scale * np.sin(angle), scale * np.cos(angle), 10],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        components = TransformCalculator.decompose_transform(transform)

        assert components["translation"] == pytest.approx((5.0, 10.0), abs=1e-5)
        assert components["scale"][0] == pytest.approx(scale, abs=1e-5)
        assert components["rotation"] == pytest.approx(angle, abs=1e-5)

    def test_wrong_shape_raises_error(self):
        """Test that wrong shape raises error."""
        wrong_shape = np.array([[1, 2], [3, 4]], dtype=np.float64)

        with pytest.raises(ValueError, match="must be 3x3"):
            TransformCalculator.decompose_transform(wrong_shape)

    def test_zero_bottom_right_raises_error(self):
        """Test that zero in h33 raises error."""
        transform = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64  # Invalid
        )

        with pytest.raises(ValueError, match="zero in bottom-right"):
            TransformCalculator.decompose_transform(transform)


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_round_trip_transform(self):
        """Test that transform -> inverse -> transform gives original points."""
        image_pts = np.array(
            [[10, 10], [100, 10], [100, 50], [10, 50]], dtype=np.float32
        )
        world_pts = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float32)

        # Calculate forward transform
        forward = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        # Calculate inverse
        inverse = TransformCalculator.calculate_inverse_transform(forward)

        # Apply forward then inverse
        transformed = TransformCalculator.apply_transform(image_pts, forward)
        back = TransformCalculator.apply_transform(transformed, inverse)

        np.testing.assert_allclose(back, image_pts, atol=1e-4)

    def test_quality_validation_workflow(self):
        """Test complete quality validation workflow."""
        # Create calibration points
        image_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        world_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

        # Calculate transform
        transform = TransformCalculator.calculate_perspective_transform(
            image_pts, world_pts
        )

        # Validate quality using same points
        quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
            transform, image_pts, world_pts
        )

        # Should have perfect or near-perfect quality
        assert quality > 0.99
        assert mean_err < 0.1
        assert max_err < 0.2

    def test_real_world_calibration_scenario(self):
        """Test realistic calibration scenario."""
        # Simulate camera view of calibration board
        # Board corners in image coordinates (640x480 image)
        image_corners = np.array(
            [
                [120, 100],  # Top-left
                [520, 110],  # Top-right
                [510, 380],  # Bottom-right
                [130, 370],  # Bottom-left
            ],
            dtype=np.float32,
        )

        # Known world coordinates (44cm x 9cm board)
        world_corners = np.array(
            [
                [0, 0],  # Top-left
                [44, 0],  # Top-right
                [44, 9],  # Bottom-right
                [0, 9],  # Bottom-left
            ],
            dtype=np.float32,
        )

        # Calculate transform
        transform = TransformCalculator.calculate_perspective_transform(
            image_corners, world_corners
        )

        # Test transform on new points
        test_image_pts = np.array(
            [[320, 240], [200, 150]], dtype=np.float32  # Image center  # Another point
        )

        world_pts = TransformCalculator.apply_transform(test_image_pts, transform)

        # Results should be reasonable (within board bounds)
        assert np.all(world_pts[:, 0] >= -5)  # X in reasonable range
        assert np.all(world_pts[:, 0] <= 50)
        assert np.all(world_pts[:, 1] >= -5)  # Y in reasonable range
        assert np.all(world_pts[:, 1] <= 15)

        # Verify quality
        quality, _, _ = TransformCalculator.validate_transform_quality(
            transform, image_corners, world_corners
        )
        assert quality > 0.99
