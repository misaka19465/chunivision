"""
Geometric utility functions for ChunIVision.

Provides:
- Point2D and Point3D classes for coordinate representation
- Coordinate transformation functions
- Point-in-polygon test
- Distance calculations
- Zone boundary and center calculations
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np


@dataclass
class Point2D:
    """2D point representation."""

    x: float
    y: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y])

    def distance_to(self, other: "Point2D") -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: "Point2D") -> "Point2D":
        """Add two points."""
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2D") -> "Point2D":
        """Subtract two points."""
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point2D":
        """Multiply point by scalar."""
        return Point2D(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.2f}, y={self.y:.2f})"


@dataclass
class Point3D:
    """3D point representation."""

    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Point3D") -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def to_2d(self) -> Point2D:
        """Project to 2D by dropping z coordinate."""
        return Point2D(self.x, self.y)

    def __add__(self, other: "Point3D") -> "Point3D":
        """Add two points."""
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        """Subtract two points."""
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Point3D":
        """Multiply point by scalar."""
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


def point_in_polygon(
    point: Union[Point2D, np.ndarray], polygon: Union[List[Point2D], np.ndarray]
) -> bool:
    """
    Test if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: Point to test (Point2D or array [x, y])
        polygon: List of vertices defining the polygon (Point2D list or array [[x1,y1], [x2,y2], ...])

    Returns:
        True if point is inside polygon, False otherwise
    """
    # Convert to numpy arrays for efficiency
    if isinstance(point, Point2D):
        px, py = point.x, point.y
    else:
        px, py = point[0], point[1]

    if isinstance(polygon[0], Point2D):
        poly_array = np.array([[p.x, p.y] for p in polygon])
    else:
        poly_array = np.array(polygon)

    n = len(poly_array)
    inside = False

    # Ray casting algorithm
    p1x, p1y = poly_array[0]
    for i in range(1, n + 1):
        p2x, p2y = poly_array[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def distance_2d(
    p1: Union[Point2D, np.ndarray], p2: Union[Point2D, np.ndarray]
) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        p1: First point (Point2D or array [x, y])
        p2: Second point (Point2D or array [x, y])

    Returns:
        Euclidean distance
    """
    if isinstance(p1, Point2D) and isinstance(p2, Point2D):
        return p1.distance_to(p2)

    # Convert to arrays
    a1 = p1 if isinstance(p1, np.ndarray) else np.array([p1.x, p1.y])
    a2 = p2 if isinstance(p2, np.ndarray) else np.array([p2.x, p2.y])

    return float(np.linalg.norm(a1 - a2))


def distance_3d(
    p1: Union[Point3D, np.ndarray], p2: Union[Point3D, np.ndarray]
) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        p1: First point (Point3D or array [x, y, z])
        p2: Second point (Point3D or array [x, y, z])

    Returns:
        Euclidean distance
    """
    if isinstance(p1, Point3D) and isinstance(p2, Point3D):
        return p1.distance_to(p2)

    # Convert to arrays
    a1 = p1 if isinstance(p1, np.ndarray) else np.array([p1.x, p1.y, p1.z])
    a2 = p2 if isinstance(p2, np.ndarray) else np.array([p2.x, p2.y, p2.z])

    return float(np.linalg.norm(a1 - a2))


def transform_camera_to_world(
    point_camera: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    """
    Transform point from camera coordinates to world coordinates.

    Args:
        point_camera: Point in camera coordinates (3,) array
        rotation_matrix: 3x3 rotation matrix
        translation_vector: Translation vector (3,) array

    Returns:
        Point in world coordinates (3,) array
    """
    return rotation_matrix @ point_camera + translation_vector


def transform_world_to_camera(
    point_world: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray
) -> np.ndarray:
    """
    Transform point from world coordinates to camera coordinates.

    Args:
        point_world: Point in world coordinates (3,) array
        rotation_matrix: 3x3 rotation matrix
        translation_vector: Translation vector (3,) array

    Returns:
        Point in camera coordinates (3,) array
    """
    return rotation_matrix.T @ (point_world - translation_vector)


def project_3d_to_2d(point_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D point to 2D image plane using camera intrinsics.

    Args:
        point_3d: 3D point in camera coordinates (3,) array [x, y, z]
        camera_matrix: 3x3 camera intrinsic matrix

    Returns:
        2D point in pixel coordinates (2,) array [u, v]
    """
    # Convert to homogeneous coordinates
    point_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2]])

    # Project to image plane
    pixel_homogeneous = camera_matrix @ point_homogeneous

    # Normalize by z coordinate
    if pixel_homogeneous[2] != 0:
        u = pixel_homogeneous[0] / pixel_homogeneous[2]
        v = pixel_homogeneous[1] / pixel_homogeneous[2]
    else:
        u, v = 0.0, 0.0

    return np.array([u, v])


def backproject_2d_to_3d(
    point_2d: np.ndarray, depth: float, camera_matrix: np.ndarray
) -> np.ndarray:
    """
    Back-project 2D pixel to 3D point given depth.

    Args:
        point_2d: 2D point in pixel coordinates (2,) array [u, v]
        depth: Depth value (z coordinate in camera frame)
        camera_matrix: 3x3 camera intrinsic matrix

    Returns:
        3D point in camera coordinates (3,) array [x, y, z]
    """
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Back-project
    u, v = point_2d
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.array([x, y, z])


def calculate_rectangle_corners(
    center: Point2D, width: float, height: float
) -> List[Point2D]:
    """
    Calculate corners of a rectangle given center and dimensions.

    Args:
        center: Center point of rectangle
        width: Rectangle width
        height: Rectangle height

    Returns:
        List of 4 corner points [bottom-left, bottom-right, top-right, top-left]
    """
    half_w = width / 2.0
    half_h = height / 2.0

    return [
        Point2D(center.x - half_w, center.y - half_h),  # Bottom-left
        Point2D(center.x + half_w, center.y - half_h),  # Bottom-right
        Point2D(center.x + half_w, center.y + half_h),  # Top-right
        Point2D(center.x - half_w, center.y + half_h),  # Top-left
    ]


def interpolate_points(
    p1: Union[Point2D, Point3D], p2: Union[Point2D, Point3D], t: float
) -> Union[Point2D, Point3D]:
    """
    Linear interpolation between two points.

    Args:
        p1: Start point
        p2: End point
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated point
    """
    if isinstance(p1, Point2D) and isinstance(p2, Point2D):
        return Point2D(p1.x + (p2.x - p1.x) * t, p1.y + (p2.y - p1.y) * t)
    elif isinstance(p1, Point3D) and isinstance(p2, Point3D):
        return Point3D(
            p1.x + (p2.x - p1.x) * t, p1.y + (p2.y - p1.y) * t, p1.z + (p2.z - p1.z) * t
        )
    else:
        raise TypeError("Both points must be of the same type (Point2D or Point3D)")
