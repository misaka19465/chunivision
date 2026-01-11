"""
Hand Detection module for ChunIVision.

Detects and tracks hands in 3D point clouds using clustering algorithms
and temporal tracking with Kalman filtering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

from ..utils.logger import Logger
from .point_cloud import PointCloud3D

logger = Logger.get_logger(__name__)


@dataclass
class Hand:
    """
    Represents a detected hand.

    Attributes:
        position: 3D position (x, y, z) in cm from origin
        velocity: 3D velocity vector in cm/s
        confidence: Detection confidence [0.0, 1.0]
        track_id: Unique identifier for tracking across frames
        timestamp: Detection timestamp
        num_points: Number of points in the hand cluster
        bounding_box: Optional (min_xyz, max_xyz) bounding box
    """

    position: np.ndarray  # shape (3,)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 1.0
    track_id: int = -1
    timestamp: float = 0.0
    num_points: int = 0
    bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __post_init__(self) -> None:
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)

    @property
    def x(self) -> float:
        """X coordinate in cm."""
        return float(self.position[0])

    @property
    def y(self) -> float:
        """Y coordinate in cm."""
        return float(self.position[1])

    @property
    def z(self) -> float:
        """Z coordinate (height) in cm."""
        return float(self.position[2])

    @property
    def speed(self) -> float:
        """Speed magnitude in cm/s."""
        return float(np.linalg.norm(self.velocity))

    def distance_to(self, other: "Hand") -> float:
        """Calculate Euclidean distance to another hand."""
        return float(np.linalg.norm(self.position - other.position))

    def __repr__(self) -> str:
        return (
            f"Hand(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}], "
            f"id={self.track_id}, conf={self.confidence:.2f})"
        )


@dataclass
class HandDetectorConfig:
    """
    Configuration for hand detection.

    Attributes:
        min_hand_points: Minimum points to consider a cluster as a hand
        max_hand_points: Maximum points for a hand cluster (filter large objects)
        min_hand_size_cm: Minimum hand dimension in cm
        max_hand_size_cm: Maximum hand dimension in cm
        clustering_eps: DBSCAN epsilon (max distance between points in cluster) in cm
        clustering_min_samples: DBSCAN minimum samples per cluster
        max_hands: Maximum number of hands to detect
        tracking_max_distance: Maximum distance for tracking association in cm
        tracking_max_frames_lost: Frames before dropping a lost track
        velocity_smoothing: Exponential smoothing factor for velocity (0-1)
        min_confidence: Minimum confidence threshold for detection
        intensity_threshold: Minimum IR intensity for valid points (0-255)
        height_range: Valid height range (min_z, max_z) in cm
    """

    min_hand_points: int = 50
    max_hand_points: int = 5000
    min_hand_size_cm: float = 3.0
    max_hand_size_cm: float = 25.0
    clustering_eps: float = 2.0  # cm
    clustering_min_samples: int = 10
    max_hands: int = 10
    tracking_max_distance: float = 15.0  # cm
    tracking_max_frames_lost: int = 5
    velocity_smoothing: float = 0.3
    min_confidence: float = 0.3
    intensity_threshold: int = 30
    height_range: Tuple[float, float] = (0.0, 40.0)  # cm

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if self.min_hand_points <= 0:
            errors.append(
                f"min_hand_points must be positive, got {self.min_hand_points}"
            )
        if self.max_hand_points <= self.min_hand_points:
            errors.append("max_hand_points must be greater than min_hand_points")
        if self.clustering_eps <= 0:
            errors.append(f"clustering_eps must be positive, got {self.clustering_eps}")
        if self.max_hands <= 0:
            errors.append(f"max_hands must be positive, got {self.max_hands}")
        if not 0 <= self.velocity_smoothing <= 1:
            errors.append(
                f"velocity_smoothing must be in [0, 1], got {self.velocity_smoothing}"
            )
        if self.height_range[0] >= self.height_range[1]:
            errors.append("height_range[0] must be less than height_range[1]")

        return errors


class KalmanTracker:
    """
    Simple Kalman filter for 3D position tracking.

    Tracks position and velocity with constant velocity model.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        process_noise: float = 1.0,
        measurement_noise: float = 0.5,
    ):
        """
        Initialize Kalman tracker.

        Args:
            initial_position: Initial 3D position
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.state[:3] = initial_position

        # State covariance
        self.P = np.eye(6) * 10

        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        # Will be updated with dt

        # Measurement matrix (we only observe position)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Process noise
        self.Q = np.eye(6) * process_noise

        # Measurement noise
        self.R = np.eye(3) * measurement_noise

        self.last_update_time = time.time()

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Predict next state.

        Args:
            dt: Time delta in seconds (auto-computed if None)

        Returns:
            Predicted position
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

        # Update state transition matrix with dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.

        Args:
            measurement: Observed 3D position

        Returns:
            Updated position
        """
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        self.last_update_time = time.time()

        return self.state[:3].copy()

    @property
    def position(self) -> np.ndarray:
        """Current position estimate."""
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate."""
        return self.state[3:].copy()


class TrackedHand:
    """Internal class for tracking hand state across frames."""

    def __init__(self, hand: Hand, track_id: int):
        """Initialize tracked hand."""
        self.track_id = track_id
        self.kalman = KalmanTracker(hand.position)
        self.frames_since_update = 0
        self.total_frames = 1
        self.last_hand = hand
        self.confidence_history: List[float] = [hand.confidence]

    def predict(self) -> np.ndarray:
        """Predict next position."""
        self.frames_since_update += 1
        return self.kalman.predict()

    def update(self, hand: Hand) -> Hand:
        """Update with new detection."""
        self.frames_since_update = 0
        self.total_frames += 1

        # Update Kalman filter
        self.kalman.update(hand.position)

        # Update confidence history
        self.confidence_history.append(hand.confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)

        # Create updated hand
        updated_hand = Hand(
            position=self.kalman.position,
            velocity=self.kalman.velocity,
            confidence=hand.confidence,
            track_id=self.track_id,
            timestamp=hand.timestamp,
            num_points=hand.num_points,
            bounding_box=hand.bounding_box,
        )

        self.last_hand = updated_hand
        return updated_hand

    @property
    def is_lost(self) -> bool:
        """Check if track is lost."""
        return self.frames_since_update > 0

    @property
    def average_confidence(self) -> float:
        """Average confidence over recent frames."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)


class HandDetector:
    """
    Detects hands in point cloud and tracks them across frames.

    Uses density-based clustering to find hand-sized regions in the
    3D point cloud, then tracks them across frames using Kalman filtering.

    Usage:
        detector = HandDetector()
        for point_cloud in point_clouds:
            hands = detector.detect(point_cloud)
            for hand in hands:
                print(f"Hand at {hand.position}, id={hand.track_id}")
    """

    def __init__(self, config: Optional[HandDetectorConfig] = None):
        """
        Initialize hand detector.

        Args:
            config: Detection parameters (uses defaults if None)
        """
        self.config = config or HandDetectorConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid hand detector config: {', '.join(errors)}")

        # Tracking state
        self._tracks: Dict[int, TrackedHand] = {}
        self._next_track_id = 0
        self._previous_hands: List[Hand] = []

        # Performance tracking
        self._frame_count = 0
        self._last_processing_time_ms = 0.0

        logger.info(
            f"HandDetector initialized: min_points={self.config.min_hand_points}, "
            f"max_hands={self.config.max_hands}"
        )

    def detect(self, point_cloud: PointCloud3D) -> List[Hand]:
        """
        Detect all hands in current point cloud.

        Args:
            point_cloud: 3D point cloud from stereo processing

        Returns:
            List of detected Hand objects with tracking IDs
        """
        start_time = time.perf_counter()
        timestamp = point_cloud.timestamp or time.time()

        # Filter point cloud by height range
        filtered_cloud = point_cloud.filter_by_depth(
            self.config.height_range[0], self.config.height_range[1]
        )

        # Filter by intensity if available
        if filtered_cloud.intensities is not None:
            intensity_mask = (
                filtered_cloud.intensities >= self.config.intensity_threshold
            )
            filtered_cloud = PointCloud3D(
                points=filtered_cloud.points[intensity_mask],
                intensities=filtered_cloud.intensities[intensity_mask],
                timestamp=filtered_cloud.timestamp,
                frame_id=filtered_cloud.frame_id,
            )

        if filtered_cloud.is_empty:
            self._handle_no_detections()
            self._frame_count += 1
            self._last_processing_time_ms = (time.perf_counter() - start_time) * 1000
            return []

        # Cluster points
        clusters = self._cluster_points(filtered_cloud.points)

        # Convert clusters to hand detections
        raw_hands = self._clusters_to_hands(clusters, filtered_cloud, timestamp)

        # Track hands across frames
        tracked_hands = self._track_hands(raw_hands)

        # Sort by confidence and limit to max_hands
        tracked_hands.sort(key=lambda h: h.confidence, reverse=True)
        tracked_hands = tracked_hands[: self.config.max_hands]

        self._previous_hands = tracked_hands
        self._frame_count += 1
        self._last_processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Frame {self._frame_count}: detected {len(tracked_hands)} hands, "
            f"{self._last_processing_time_ms:.1f}ms"
        )

        return tracked_hands

    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster 3D points using grid-based density clustering.

        Uses a voxel grid approach for efficiency, similar to DBSCAN
        but optimized for real-time processing.
        """
        if len(points) == 0:
            return []

        # Voxelize the point cloud
        voxel_size = self.config.clustering_eps
        min_bounds = points.min(axis=0)
        voxel_indices = ((points - min_bounds) / voxel_size).astype(np.int32)

        # Create a 3D grid
        max_indices = voxel_indices.max(axis=0) + 1
        grid_shape = tuple(max_indices)

        # Handle very large grids
        if np.prod(grid_shape) > 10_000_000:
            logger.warning("Point cloud too large for clustering, downsampling")
            # Downsample by taking every nth point
            step = max(1, len(points) // 100_000)
            points = points[::step]
            voxel_indices = ((points - min_bounds) / voxel_size).astype(np.int32)
            max_indices = voxel_indices.max(axis=0) + 1
            grid_shape = tuple(max_indices)

        # Create occupancy grid
        grid = np.zeros(grid_shape, dtype=np.int32)

        # Count points per voxel
        for idx in voxel_indices:
            grid[tuple(idx)] += 1

        # Threshold grid (minimum density)
        min_density = max(1, self.config.clustering_min_samples // 10)
        binary_grid = grid >= min_density

        # Label connected components
        structure = ndimage.generate_binary_structure(3, 2)  # 26-connectivity
        labeled, num_clusters = ndimage.label(binary_grid, structure=structure)

        # Extract clusters
        clusters = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = np.zeros(len(points), dtype=bool)
            for i, idx in enumerate(voxel_indices):
                if labeled[tuple(idx)] == cluster_id:
                    cluster_mask[i] = True

            cluster_points = points[cluster_mask]
            if len(cluster_points) >= self.config.min_hand_points:
                clusters.append(cluster_points)

        return clusters

    def _clusters_to_hands(
        self,
        clusters: List[np.ndarray],
        point_cloud: PointCloud3D,
        timestamp: float,
    ) -> List[Hand]:
        """Convert point clusters to Hand objects."""
        hands = []

        for cluster in clusters:
            num_points = len(cluster)

            # Filter by point count
            if num_points < self.config.min_hand_points:
                continue
            if num_points > self.config.max_hand_points:
                continue

            # Compute bounding box
            min_bounds = cluster.min(axis=0)
            max_bounds = cluster.max(axis=0)
            size = max_bounds - min_bounds

            # Filter by size
            max_dim = size.max()
            if max_dim < self.config.min_hand_size_cm:
                continue
            if max_dim > self.config.max_hand_size_cm:
                continue

            # Compute centroid as position
            centroid = cluster.mean(axis=0)

            # Compute confidence based on cluster properties
            confidence = self._compute_confidence(cluster, num_points, size)

            if confidence < self.config.min_confidence:
                continue

            hand = Hand(
                position=centroid,
                velocity=np.zeros(3),
                confidence=confidence,
                track_id=-1,
                timestamp=timestamp,
                num_points=num_points,
                bounding_box=(min_bounds, max_bounds),
            )
            hands.append(hand)

        return hands

    def _compute_confidence(
        self, cluster: np.ndarray, num_points: int, size: np.ndarray
    ) -> float:
        """
        Compute detection confidence based on cluster properties.

        Higher confidence for:
        - More points (up to expected hand size)
        - Compact shape (low aspect ratio)
        - Hand-like dimensions
        """
        # Point count factor (peaks around expected hand points)
        expected_points = 500
        point_factor = 1.0 - abs(num_points - expected_points) / (expected_points * 3)
        point_factor = max(0.2, min(1.0, point_factor))

        # Size factor (peaks around expected hand size ~10-15cm)
        expected_size = 12.0  # cm
        max_dim = size.max()
        size_factor = 1.0 - abs(max_dim - expected_size) / expected_size
        size_factor = max(0.2, min(1.0, size_factor))

        # Compactness factor (prefer more spherical shapes)
        if size.min() > 0:
            aspect_ratio = size.max() / size.min()
            compactness = 1.0 / (1.0 + (aspect_ratio - 1) * 0.2)
        else:
            compactness = 0.5

        # Combined confidence
        confidence = point_factor * 0.3 + size_factor * 0.4 + compactness * 0.3

        return float(confidence)

    def _track_hands(self, current_hands: List[Hand]) -> List[Hand]:
        """
        Associate current detections with existing tracks.

        Uses Hungarian algorithm for optimal assignment.
        """
        # Predict all existing tracks
        for track in self._tracks.values():
            track.predict()

        if not current_hands:
            self._handle_no_detections()
            return []

        if not self._tracks:
            # No existing tracks, create new ones for all detections
            tracked = []
            for hand in current_hands:
                tracked_hand = self._create_new_track(hand)
                tracked.append(tracked_hand)
            return tracked

        # Compute cost matrix (distance between detections and tracks)
        track_ids = list(self._tracks.keys())
        cost_matrix = np.zeros((len(current_hands), len(track_ids)))

        for i, hand in enumerate(current_hands):
            for j, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                predicted_pos = track.kalman.position
                distance = np.linalg.norm(hand.position - predicted_pos)
                cost_matrix[i, j] = distance

        # Greedy assignment (could use scipy.optimize.linear_sum_assignment for optimal)
        assignments = self._greedy_assignment(cost_matrix)

        tracked_hands = []
        assigned_tracks = set()
        assigned_detections = set()

        for det_idx, track_idx in assignments:
            if cost_matrix[det_idx, track_idx] <= self.config.tracking_max_distance:
                track_id = track_ids[track_idx]
                track = self._tracks[track_id]
                updated_hand = track.update(current_hands[det_idx])
                tracked_hands.append(updated_hand)
                assigned_tracks.add(track_id)
                assigned_detections.add(det_idx)

        # Create new tracks for unassigned detections
        for i, hand in enumerate(current_hands):
            if i not in assigned_detections:
                tracked_hand = self._create_new_track(hand)
                tracked_hands.append(tracked_hand)

        # Handle unassigned tracks (lost)
        lost_tracks = []
        for track_id in self._tracks:
            if track_id not in assigned_tracks:
                track = self._tracks[track_id]
                if track.frames_since_update > self.config.tracking_max_frames_lost:
                    lost_tracks.append(track_id)

        # Remove lost tracks
        for track_id in lost_tracks:
            del self._tracks[track_id]
            logger.debug(f"Removed lost track {track_id}")

        return tracked_hands

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Greedy assignment for detection-track association."""
        assignments = []
        num_detections, num_tracks = cost_matrix.shape

        if num_detections == 0 or num_tracks == 0:
            return assignments

        # Flatten and sort by cost
        flat_indices = np.argsort(cost_matrix.flatten())

        assigned_dets = set()
        assigned_tracks = set()

        for flat_idx in flat_indices:
            det_idx = flat_idx // num_tracks
            track_idx = flat_idx % num_tracks

            if det_idx not in assigned_dets and track_idx not in assigned_tracks:
                assignments.append((det_idx, track_idx))
                assigned_dets.add(det_idx)
                assigned_tracks.add(track_idx)

            if len(assignments) == min(num_detections, num_tracks):
                break

        return assignments

    def _create_new_track(self, hand: Hand) -> Hand:
        """Create a new track for a detection."""
        track_id = self._next_track_id
        self._next_track_id += 1

        tracked_hand = TrackedHand(hand, track_id)
        self._tracks[track_id] = tracked_hand

        # Return hand with assigned track ID
        return Hand(
            position=hand.position,
            velocity=np.zeros(3),
            confidence=hand.confidence,
            track_id=track_id,
            timestamp=hand.timestamp,
            num_points=hand.num_points,
            bounding_box=hand.bounding_box,
        )

    def _handle_no_detections(self) -> None:
        """Handle frame with no detections."""
        # Update all tracks as lost
        lost_tracks = []
        for track_id, track in self._tracks.items():
            track.frames_since_update += 1
            if track.frames_since_update > self.config.tracking_max_frames_lost:
                lost_tracks.append(track_id)

        for track_id in lost_tracks:
            del self._tracks[track_id]

    def track(
        self, current_hands: List[Hand], previous_hands: List[Hand]
    ) -> List[Hand]:
        """
        Associate current detections with previous frame for tracking.

        This is an alternative to the internal tracking that allows
        external control of the tracking process.

        Args:
            current_hands: Hands detected in current frame
            previous_hands: Hands from previous frame

        Returns:
            List of hands with updated track_ids and velocities
        """
        if not previous_hands:
            # No previous hands, assign new IDs
            for i, hand in enumerate(current_hands):
                hand.track_id = i
            return current_hands

        if not current_hands:
            return []

        # Compute distance matrix
        current_positions = np.array([h.position for h in current_hands])
        previous_positions = np.array([h.position for h in previous_hands])

        distances = cdist(current_positions, previous_positions)

        # Greedy assignment
        assignments = self._greedy_assignment(distances)

        tracked_hands = []
        assigned_current = set()

        for curr_idx, prev_idx in assignments:
            if distances[curr_idx, prev_idx] <= self.config.tracking_max_distance:
                hand = current_hands[curr_idx]
                prev_hand = previous_hands[prev_idx]

                # Compute velocity
                dt = hand.timestamp - prev_hand.timestamp
                if dt > 0:
                    velocity = (hand.position - prev_hand.position) / dt
                else:
                    velocity = prev_hand.velocity

                # Apply smoothing
                alpha = self.config.velocity_smoothing
                smoothed_velocity = alpha * velocity + (1 - alpha) * prev_hand.velocity

                tracked_hand = Hand(
                    position=hand.position,
                    velocity=smoothed_velocity,
                    confidence=hand.confidence,
                    track_id=prev_hand.track_id,
                    timestamp=hand.timestamp,
                    num_points=hand.num_points,
                    bounding_box=hand.bounding_box,
                )
                tracked_hands.append(tracked_hand)
                assigned_current.add(curr_idx)

        # Assign new IDs to unmatched hands
        next_id = max((h.track_id for h in previous_hands), default=-1) + 1
        for i, hand in enumerate(current_hands):
            if i not in assigned_current:
                hand.track_id = next_id
                next_id += 1
                tracked_hands.append(hand)

        return tracked_hands

    def reset_tracking(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._next_track_id = 0
        self._previous_hands = []
        logger.info("Hand tracking reset")

    def get_active_tracks(self) -> List[int]:
        """Get list of active track IDs."""
        return list(self._tracks.keys())

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific track."""
        if track_id not in self._tracks:
            return None

        track = self._tracks[track_id]
        return {
            "track_id": track_id,
            "position": track.kalman.position.tolist(),
            "velocity": track.kalman.velocity.tolist(),
            "frames_tracked": track.total_frames,
            "frames_since_update": track.frames_since_update,
            "average_confidence": track.average_confidence,
        }

    def get_processing_time_ms(self) -> float:
        """Get processing time of last frame in milliseconds."""
        return self._last_processing_time_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get detection and tracking statistics."""
        return {
            "frame_count": self._frame_count,
            "active_tracks": len(self._tracks),
            "next_track_id": self._next_track_id,
            "last_processing_time_ms": self._last_processing_time_ms,
            "config": {
                "min_hand_points": self.config.min_hand_points,
                "max_hands": self.config.max_hands,
                "tracking_max_distance": self.config.tracking_max_distance,
            },
        }
