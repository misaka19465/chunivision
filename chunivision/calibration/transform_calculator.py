"""
Transform calculator for calibration.

Computes geometric transformations (perspective transforms) from
calibration points and provides utilities for applying and validating
these transforms.
"""

from typing import Tuple
import numpy as np
import cv2


class TransformCalculator:
    """
    Computes geometric transformations for calibration.

    This class provides static methods for calculating perspective transforms,
    applying them to points, and validating their quality. It's primarily used
    during the calibration process to map image coordinates to world coordinates.
    """

    @staticmethod
    def calculate_perspective_transform(
        image_points: np.ndarray, world_points: np.ndarray
    ) -> np.ndarray:
        """
        Calculate perspective transform matrix from corresponding points.

        Uses OpenCV's getPerspectiveTransform to compute a 3x3 homography matrix
        that maps image coordinates to world coordinates.

        Args:
            image_points: Points in image coordinates, shape (4, 2) or (N, 2) where N >= 4
                         For 4 points: typically corners of a quadrilateral
            world_points: Corresponding points in world coordinates, same shape as image_points

        Returns:
            3x3 perspective transform matrix (np.float64)

        Raises:
            ValueError: If point arrays are invalid or don't match in shape

        Example:
            >>> image_pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
            >>> world_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
            >>> transform = TransformCalculator.calculate_perspective_transform(image_pts, world_pts)
            >>> transform.shape
            (3, 3)
        """
        # Validate inputs
        if image_points.shape != world_points.shape:
            raise ValueError(
                f"image_points and world_points must have same shape, "
                f"got {image_points.shape} and {world_points.shape}"
            )

        if len(image_points.shape) != 2 or image_points.shape[1] != 2:
            raise ValueError(f"Points must be shape (N, 2), got {image_points.shape}")

        if image_points.shape[0] < 4:
            raise ValueError(
                f"Need at least 4 points for perspective transform, got {image_points.shape[0]}"
            )

        # Convert to float32 for OpenCV
        src_pts = np.asarray(image_points, dtype=np.float32)
        dst_pts = np.asarray(world_points, dtype=np.float32)

        # For exactly 4 points, use getPerspectiveTransform (more accurate)
        if src_pts.shape[0] == 4:
            transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        else:
            # For more than 4 points, use findHomography with RANSAC
            transform, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if transform is None:
                raise ValueError(
                    "Failed to compute homography - points may be degenerate"
                )

        return transform.astype(np.float64)

    @staticmethod
    def calculate_inverse_transform(transform: np.ndarray) -> np.ndarray:
        """
        Calculate inverse of transform for reverse mapping.

        The inverse transform maps world coordinates back to image coordinates,
        which is useful for verification and certain calibration workflows.

        Args:
            transform: Forward transform matrix (3, 3)

        Returns:
            Inverse transform matrix (3, 3)

        Raises:
            ValueError: If transform is singular (not invertible)

        Example:
            >>> forward = np.eye(3)
            >>> inverse = TransformCalculator.calculate_inverse_transform(forward)
            >>> np.allclose(forward, inverse)
            True
        """
        if transform.shape != (3, 3):
            raise ValueError(f"Transform must be 3x3, got {transform.shape}")

        # Check if matrix is invertible
        det = np.linalg.det(transform)
        if np.abs(det) < 1e-10:
            raise ValueError(
                f"Transform is singular (det={det:.2e}) and cannot be inverted"
            )

        inverse = np.linalg.inv(transform)
        return inverse.astype(np.float64)

    @staticmethod
    def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to points.

        Transforms points from source coordinate system to destination coordinate
        system using homogeneous coordinates. Handles both single points and arrays.

        Args:
            points: Input points, shape (2,) or (N, 2)
            transform: Transform matrix (3, 3)

        Returns:
            Transformed points, same shape as input

        Example:
            >>> pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            >>> transform = np.eye(3)
            >>> transformed = TransformCalculator.apply_transform(pts, transform)
            >>> np.allclose(pts, transformed)
            True
        """
        if transform.shape != (3, 3):
            raise ValueError(f"Transform must be 3x3, got {transform.shape}")

        # Handle single point
        single_point = False
        if points.ndim == 1:
            if len(points) != 2:
                raise ValueError(
                    f"Single point must have 2 coordinates, got {len(points)}"
                )
            points = points.reshape(1, 2)
            single_point = True

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Points must be shape (N, 2) or (2,), got {points.shape}")

        # Convert to homogeneous coordinates (add column of ones)
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        homogeneous = np.hstack([points, ones])

        # Apply transform: transform @ point^T = transformed_point^T
        transformed_homogeneous = (transform @ homogeneous.T).T

        # Convert back from homogeneous coordinates (divide by w)
        w = transformed_homogeneous[:, 2:3]

        # Avoid division by zero
        if np.any(np.abs(w) < 1e-10):
            raise ValueError("Transform produced points at infinity (w ≈ 0)")

        transformed = transformed_homogeneous[:, :2] / w

        # Return single point if input was single point
        if single_point:
            return transformed[0]

        return transformed

    @staticmethod
    def validate_transform_quality(
        transform: np.ndarray,
        test_points: np.ndarray,
        expected_points: np.ndarray,
        max_error_threshold: float = 5.0,
    ) -> Tuple[float, float, float]:
        """
        Compute quality metrics for a transform.

        Applies the transform to test points and compares against expected
        output. Returns multiple metrics to assess transform quality.

        Args:
            transform: Transform to validate (3, 3)
            test_points: Input test points (N, 2)
            expected_points: Expected output points (N, 2)
            max_error_threshold: Maximum acceptable error in world units (default: 5.0)

        Returns:
            Tuple of (quality_score, mean_error, max_error)
            - quality_score: Overall quality [0.0, 1.0], 1.0 is perfect
            - mean_error: Mean Euclidean distance between transformed and expected points
            - max_error: Maximum error for any single point

        Example:
            >>> transform = np.eye(3)
            >>> test_pts = np.array([[0, 0], [1, 1]], dtype=np.float32)
            >>> expected = test_pts.copy()
            >>> score, mean_err, max_err = TransformCalculator.validate_transform_quality(
            ...     transform, test_pts, expected
            ... )
            >>> score
            1.0
            >>> mean_err
            0.0
        """
        if test_points.shape != expected_points.shape:
            raise ValueError(
                f"test_points and expected_points must have same shape, "
                f"got {test_points.shape} and {expected_points.shape}"
            )

        if test_points.shape[0] == 0:
            raise ValueError("Need at least one test point")

        # Apply transform to test points
        try:
            transformed = TransformCalculator.apply_transform(test_points, transform)
        except Exception as e:
            # If transform fails, quality is 0
            return 0.0, float("inf"), float("inf")

        # Calculate per-point errors (Euclidean distance)
        errors = np.linalg.norm(transformed - expected_points, axis=1)

        # Compute metrics
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))

        # Calculate quality score
        # Score decreases exponentially with mean error relative to threshold
        # Perfect transform (0 error) = 1.0
        # Error at threshold = ~0.37
        # Error at 2x threshold = ~0.14
        quality_score = float(np.exp(-mean_error / max_error_threshold))

        # Additional penalty if any point has error > 2x threshold
        if max_error > 2 * max_error_threshold:
            quality_score *= 0.5

        return quality_score, mean_error, max_error

    @staticmethod
    def estimate_reprojection_error(
        transform: np.ndarray, image_points: np.ndarray, world_points: np.ndarray
    ) -> float:
        """
        Estimate the reprojection error for the transform.

        This is the mean distance between the transformed image points
        and the expected world points. Lower is better.

        Args:
            transform: Transform matrix (3, 3)
            image_points: Source points in image coordinates (N, 2)
            world_points: Expected points in world coordinates (N, 2)

        Returns:
            Mean reprojection error in world coordinate units

        Example:
            >>> transform = np.eye(3)
            >>> img_pts = np.array([[0, 0], [1, 1]], dtype=np.float32)
            >>> world_pts = img_pts.copy()
            >>> error = TransformCalculator.estimate_reprojection_error(
            ...     transform, img_pts, world_pts
            ... )
            >>> error
            0.0
        """
        _, mean_error, _ = TransformCalculator.validate_transform_quality(
            transform, image_points, world_points
        )
        return mean_error

    @staticmethod
    def decompose_transform(transform: np.ndarray) -> dict:
        """
        Decompose a perspective transform into interpretable components.

        Extracts rotation, scale, shear, and translation components for
        analysis and debugging. Note that perspective transforms can't be
        perfectly decomposed into these components, so this is approximate.

        Args:
            transform: Transform matrix (3, 3)

        Returns:
            Dictionary containing:
            - 'translation': (tx, ty) translation vector
            - 'scale': (sx, sy) scale factors
            - 'rotation': rotation angle in radians
            - 'shear': shear factor
            - 'perspective': (p1, p2) perspective components

        Note:
            This decomposition is approximate for perspective transforms.
            It's most accurate for affine transforms.
        """
        if transform.shape != (3, 3):
            raise ValueError(f"Transform must be 3x3, got {transform.shape}")

        # Extract components
        # Translation is in the third column (normalized by scale)
        h33 = transform[2, 2]
        if np.abs(h33) < 1e-10:
            raise ValueError("Transform has zero in bottom-right corner")

        # Normalize by h33
        H = transform / h33

        # Translation
        tx = H[0, 2]
        ty = H[1, 2]

        # Perspective components
        p1 = H[2, 0]
        p2 = H[2, 1]

        # Upper-left 2x2 submatrix contains rotation, scale, shear
        M = H[:2, :2]

        # Decompose using SVD
        # M = U @ S @ Vt where U and V are rotations and S is scaling
        a = M[0, 0]
        b = M[0, 1]
        c = M[1, 0]
        d = M[1, 1]

        # Calculate scale and rotation
        sx = np.sqrt(a * a + c * c)
        sy = np.sqrt(b * b + d * d)

        # Sign adjustment
        det = a * d - b * c
        if det < 0:
            sy = -sy

        # Rotation angle
        rotation = np.arctan2(c, a)

        # Shear (simplified)
        shear = (a * b + c * d) / (sx * sx) if sx > 1e-10 else 0.0

        return {
            "translation": (float(tx), float(ty)),
            "scale": (float(sx), float(sy)),
            "rotation": float(rotation),
            "shear": float(shear),
            "perspective": (float(p1), float(p2)),
        }
