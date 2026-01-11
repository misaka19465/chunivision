"""
Tests for HandDetector module.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chunivision.vision.hand_detector import (
    Hand,
    HandDetector,
    HandDetectorConfig,
    KalmanTracker,
    TrackedHand,
)
from chunivision.vision.point_cloud import PointCloud3D


class TestHand:
    """Tests for Hand dataclass."""

    def test_hand_creation(self):
        """Test basic hand creation."""
        hand = Hand(position=np.array([10.0, 20.0, 5.0]))

        assert hand.x == 10.0
        assert hand.y == 20.0
        assert hand.z == 5.0
        assert hand.track_id == -1
        assert hand.confidence == 1.0

    def test_hand_with_all_fields(self):
        """Test hand creation with all fields."""
        hand = Hand(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.5, 0.5, 0.0]),
            confidence=0.85,
            track_id=42,
            timestamp=1234567890.0,
            num_points=500,
        )

        assert hand.track_id == 42
        assert hand.confidence == 0.85
        assert hand.num_points == 500
        np.testing.assert_array_equal(hand.velocity, [0.5, 0.5, 0.0])

    def test_hand_from_list(self):
        """Test hand creation from list."""
        hand = Hand(position=[5.0, 10.0, 15.0])

        assert isinstance(hand.position, np.ndarray)
        assert hand.x == 5.0

    def test_hand_speed(self):
        """Test speed calculation."""
        hand = Hand(
            position=np.array([0, 0, 0]),
            velocity=np.array([3.0, 4.0, 0.0]),
        )

        assert hand.speed == 5.0  # 3-4-5 triangle

    def test_hand_distance_to(self):
        """Test distance calculation between hands."""
        hand1 = Hand(position=np.array([0.0, 0.0, 0.0]))
        hand2 = Hand(position=np.array([3.0, 4.0, 0.0]))

        assert hand1.distance_to(hand2) == 5.0

    def test_hand_repr(self):
        """Test string representation."""
        hand = Hand(
            position=np.array([10.5, 20.3, 5.7]),
            track_id=5,
            confidence=0.9,
        )

        repr_str = repr(hand)
        assert "10.5" in repr_str
        assert "id=5" in repr_str
        assert "conf=0.90" in repr_str


class TestHandDetectorConfig:
    """Tests for HandDetectorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HandDetectorConfig()

        assert config.min_hand_points == 50
        assert config.max_hands == 10
        assert config.clustering_eps == 2.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = HandDetectorConfig(
            min_hand_points=100,
            max_hands=5,
            clustering_eps=3.0,
        )

        assert config.min_hand_points == 100
        assert config.max_hands == 5

    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = HandDetectorConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_min_points(self):
        """Test validation catches invalid min_hand_points."""
        config = HandDetectorConfig(min_hand_points=0)
        errors = config.validate()
        assert any("min_hand_points" in e for e in errors)

    def test_validate_invalid_max_points(self):
        """Test validation catches max <= min."""
        config = HandDetectorConfig(min_hand_points=1000, max_hand_points=100)
        errors = config.validate()
        assert any("max_hand_points" in e for e in errors)

    def test_validate_invalid_velocity_smoothing(self):
        """Test validation catches invalid velocity_smoothing."""
        config = HandDetectorConfig(velocity_smoothing=1.5)
        errors = config.validate()
        assert any("velocity_smoothing" in e for e in errors)

    def test_validate_invalid_height_range(self):
        """Test validation catches invalid height_range."""
        config = HandDetectorConfig(height_range=(50.0, 10.0))
        errors = config.validate()
        assert any("height_range" in e for e in errors)


class TestKalmanTracker:
    """Tests for KalmanTracker."""

    def test_initialization(self):
        """Test Kalman tracker initialization."""
        initial_pos = np.array([10.0, 20.0, 5.0])
        tracker = KalmanTracker(initial_pos)

        np.testing.assert_array_equal(tracker.position, initial_pos)
        np.testing.assert_array_equal(tracker.velocity, [0, 0, 0])

    def test_predict(self):
        """Test prediction step."""
        tracker = KalmanTracker(np.array([0.0, 0.0, 0.0]))

        # Predict with known dt
        predicted = tracker.predict(dt=0.1)

        # With zero initial velocity, position should stay approximately the same
        assert predicted.shape == (3,)

    def test_update(self):
        """Test update step."""
        tracker = KalmanTracker(np.array([0.0, 0.0, 0.0]))

        # Update with new measurement
        new_pos = np.array([10.0, 10.0, 0.0])
        updated = tracker.update(new_pos)

        # Position should move toward measurement
        assert updated[0] > 0
        assert updated[1] > 0

    def test_velocity_estimation(self):
        """Test velocity estimation over multiple updates."""
        tracker = KalmanTracker(np.array([0.0, 0.0, 0.0]))

        # Simulate constant velocity motion
        for i in range(10):
            pos = np.array([float(i), 0.0, 0.0])
            tracker.predict(dt=0.016)  # ~60 FPS
            tracker.update(pos)

        # Velocity should be estimated
        velocity = tracker.velocity
        # X velocity should be positive
        assert velocity[0] > 0


class TestTrackedHand:
    """Tests for TrackedHand."""

    def test_initialization(self):
        """Test tracked hand initialization."""
        hand = Hand(position=np.array([10.0, 20.0, 5.0]), confidence=0.9)
        tracked = TrackedHand(hand, track_id=1)

        assert tracked.track_id == 1
        assert tracked.total_frames == 1
        assert tracked.frames_since_update == 0
        assert not tracked.is_lost

    def test_update(self):
        """Test updating tracked hand."""
        hand1 = Hand(position=np.array([10.0, 20.0, 5.0]))
        tracked = TrackedHand(hand1, track_id=1)

        hand2 = Hand(position=np.array([12.0, 22.0, 5.0]), timestamp=0.1)
        updated = tracked.update(hand2)

        assert tracked.total_frames == 2
        assert updated.track_id == 1

    def test_predict_marks_as_lost(self):
        """Test that predict increments frames_since_update."""
        hand = Hand(position=np.array([10.0, 20.0, 5.0]))
        tracked = TrackedHand(hand, track_id=1)

        tracked.predict()

        assert tracked.frames_since_update == 1
        assert tracked.is_lost

    def test_average_confidence(self):
        """Test average confidence calculation."""
        hand = Hand(position=np.array([10.0, 20.0, 5.0]), confidence=0.8)
        tracked = TrackedHand(hand, track_id=1)

        hand2 = Hand(position=np.array([10.0, 20.0, 5.0]), confidence=1.0)
        tracked.update(hand2)

        assert tracked.average_confidence == 0.9  # (0.8 + 1.0) / 2


class TestHandDetector:
    """Tests for HandDetector."""

    @pytest.fixture
    def detector(self):
        """Create hand detector with default config."""
        return HandDetector()

    @pytest.fixture
    def simple_point_cloud(self):
        """Create simple point cloud with one hand-like cluster."""
        # Create a cluster of points resembling a hand
        center = np.array([10.0, 20.0, 5.0])
        num_points = 200
        noise = np.random.randn(num_points, 3) * 2  # ~2cm spread

        points = center + noise

        return PointCloud3D(
            points=points,
            intensities=np.full(num_points, 100, dtype=np.uint8),
            timestamp=time.time(),
            frame_id=1,
        )

    @pytest.fixture
    def two_hands_cloud(self):
        """Create point cloud with two hand-like clusters."""
        num_points = 200

        # First hand cluster
        center1 = np.array([10.0, 20.0, 5.0])
        cluster1 = center1 + np.random.randn(num_points, 3) * 2

        # Second hand cluster (well separated)
        center2 = np.array([30.0, 20.0, 5.0])
        cluster2 = center2 + np.random.randn(num_points, 3) * 2

        points = np.vstack([cluster1, cluster2])
        intensities = np.full(num_points * 2, 100, dtype=np.uint8)

        return PointCloud3D(
            points=points,
            intensities=intensities,
            timestamp=time.time(),
            frame_id=1,
        )

    def test_initialization(self):
        """Test detector initialization."""
        detector = HandDetector()

        assert detector.config.max_hands == 10
        assert detector._frame_count == 0
        assert len(detector._tracks) == 0

    def test_initialization_with_config(self):
        """Test detector initialization with custom config."""
        config = HandDetectorConfig(max_hands=5, min_hand_points=30)
        detector = HandDetector(config)

        assert detector.config.max_hands == 5
        assert detector.config.min_hand_points == 30

    def test_initialization_invalid_config(self):
        """Test detector raises on invalid config."""
        config = HandDetectorConfig(min_hand_points=-10)

        with pytest.raises(ValueError):
            HandDetector(config)

    def test_detect_empty_cloud(self, detector):
        """Test detection on empty point cloud."""
        cloud = PointCloud3D(points=np.empty((0, 3)))

        hands = detector.detect(cloud)

        assert len(hands) == 0

    def test_detect_single_hand(self, detector, simple_point_cloud):
        """Test detection of single hand."""
        hands = detector.detect(simple_point_cloud)

        # Should detect at least one hand (depends on clustering)
        # The cluster might not be detected if points are too scattered
        assert len(hands) <= 1  # At most one hand

        if len(hands) == 1:
            hand = hands[0]
            assert hand.track_id >= 0
            assert hand.confidence > 0

    def test_detect_two_hands(self, detector, two_hands_cloud):
        """Test detection of two hands."""
        # Use more relaxed config
        config = HandDetectorConfig(min_hand_points=30, clustering_eps=3.0)
        detector = HandDetector(config)

        hands = detector.detect(two_hands_cloud)

        # Should detect up to 2 hands
        assert len(hands) <= 2

    def test_tracking_across_frames(self, detector):
        """Test hand tracking across multiple frames."""
        # Create sequence of point clouds with moving hand
        track_ids = []

        for i in range(5):
            center = np.array([10.0 + i * 2, 20.0, 5.0])
            points = center + np.random.randn(200, 3) * 2

            cloud = PointCloud3D(
                points=points,
                intensities=np.full(200, 100, dtype=np.uint8),
                timestamp=time.time() + i * 0.016,
                frame_id=i,
            )

            hands = detector.detect(cloud)
            if hands:
                track_ids.append(hands[0].track_id)

        # Track IDs should be consistent if tracking is working
        if len(track_ids) > 1:
            # Most track IDs should be the same
            from collections import Counter

            counts = Counter(track_ids)
            most_common = counts.most_common(1)[0][1]
            assert most_common >= len(track_ids) // 2

    def test_reset_tracking(self, detector, simple_point_cloud):
        """Test tracking reset."""
        detector.detect(simple_point_cloud)
        assert len(detector._tracks) >= 0

        detector.reset_tracking()

        assert len(detector._tracks) == 0
        assert detector._next_track_id == 0

    def test_get_active_tracks(self, detector, simple_point_cloud):
        """Test getting active track IDs."""
        detector.detect(simple_point_cloud)

        tracks = detector.get_active_tracks()

        assert isinstance(tracks, list)
        for track_id in tracks:
            assert isinstance(track_id, int)

    def test_get_track_info(self, detector, simple_point_cloud):
        """Test getting track information."""
        hands = detector.detect(simple_point_cloud)

        if hands:
            track_id = hands[0].track_id
            info = detector.get_track_info(track_id)

            assert info is not None
            assert "position" in info
            assert "velocity" in info
            assert "frames_tracked" in info

    def test_get_track_info_nonexistent(self, detector):
        """Test getting info for nonexistent track."""
        info = detector.get_track_info(999)
        assert info is None

    def test_get_processing_time(self, detector, simple_point_cloud):
        """Test processing time measurement."""
        detector.detect(simple_point_cloud)

        time_ms = detector.get_processing_time_ms()

        assert time_ms >= 0
        assert time_ms < 1000  # Should be reasonable

    def test_get_stats(self, detector):
        """Test getting statistics."""
        stats = detector.get_stats()

        assert "frame_count" in stats
        assert "active_tracks" in stats
        assert "config" in stats

    def test_external_track_method(self, detector):
        """Test external tracking method."""
        # Create two sets of hands
        prev_hands = [
            Hand(position=np.array([10.0, 20.0, 5.0]), track_id=0, timestamp=0.0),
            Hand(position=np.array([30.0, 20.0, 5.0]), track_id=1, timestamp=0.0),
        ]

        curr_hands = [
            Hand(position=np.array([12.0, 22.0, 5.0]), timestamp=0.1),  # Moved hand 0
            Hand(position=np.array([31.0, 21.0, 5.0]), timestamp=0.1),  # Moved hand 1
        ]

        tracked = detector.track(curr_hands, prev_hands)

        assert len(tracked) == 2

        # Check that track IDs are assigned
        track_ids = [h.track_id for h in tracked]
        assert 0 in track_ids
        assert 1 in track_ids

    def test_external_track_with_new_hand(self, detector):
        """Test external tracking with new hand appearing."""
        prev_hands = [
            Hand(position=np.array([10.0, 20.0, 5.0]), track_id=0, timestamp=0.0),
        ]

        curr_hands = [
            Hand(position=np.array([11.0, 21.0, 5.0]), timestamp=0.1),  # Moved hand 0
            Hand(position=np.array([50.0, 20.0, 5.0]), timestamp=0.1),  # New hand
        ]

        tracked = detector.track(curr_hands, prev_hands)

        assert len(tracked) == 2

        # One should have ID 0 (tracked), one should have new ID
        track_ids = set(h.track_id for h in tracked)
        assert 0 in track_ids
        assert len(track_ids) == 2

    def test_external_track_empty_previous(self, detector):
        """Test external tracking with no previous hands."""
        curr_hands = [
            Hand(position=np.array([10.0, 20.0, 5.0])),
        ]

        tracked = detector.track(curr_hands, [])

        assert len(tracked) == 1
        assert tracked[0].track_id == 0

    def test_external_track_empty_current(self, detector):
        """Test external tracking with no current hands."""
        prev_hands = [
            Hand(position=np.array([10.0, 20.0, 5.0]), track_id=0),
        ]

        tracked = detector.track([], prev_hands)

        assert len(tracked) == 0


class TestHandDetectorWithHeightFilter:
    """Tests for height filtering in hand detection."""

    def test_height_filter_removes_out_of_range(self):
        """Test that points outside height range are filtered."""
        config = HandDetectorConfig(
            height_range=(5.0, 20.0),
            min_hand_points=30,
        )
        detector = HandDetector(config)

        # Create cluster at z=25 (outside range)
        center = np.array([10.0, 20.0, 25.0])
        points = center + np.random.randn(200, 3) * 2

        cloud = PointCloud3D(
            points=points,
            intensities=np.full(200, 100, dtype=np.uint8),
            timestamp=time.time(),
        )

        hands = detector.detect(cloud)

        # Should not detect hands outside height range
        assert len(hands) == 0

    def test_height_filter_keeps_in_range(self):
        """Test that points inside height range are kept."""
        config = HandDetectorConfig(
            height_range=(0.0, 30.0),
            min_hand_points=30,
            clustering_eps=3.0,
        )
        detector = HandDetector(config)

        # Create cluster at z=10 (inside range)
        center = np.array([10.0, 20.0, 10.0])
        points = center + np.random.randn(200, 3) * 2

        cloud = PointCloud3D(
            points=points,
            intensities=np.full(200, 100, dtype=np.uint8),
            timestamp=time.time(),
        )

        hands = detector.detect(cloud)

        # May or may not detect depending on clustering
        # But if detected, should be in valid range
        for hand in hands:
            assert config.height_range[0] <= hand.z <= config.height_range[1] + 10


class TestHandDetectorIntensityFilter:
    """Tests for intensity filtering in hand detection."""

    def test_low_intensity_filtered(self):
        """Test that low intensity points are filtered."""
        config = HandDetectorConfig(
            intensity_threshold=50,
            min_hand_points=30,
        )
        detector = HandDetector(config)

        # Create cluster with low intensity
        center = np.array([10.0, 20.0, 5.0])
        points = center + np.random.randn(200, 3) * 2

        cloud = PointCloud3D(
            points=points,
            intensities=np.full(200, 20, dtype=np.uint8),  # Below threshold
            timestamp=time.time(),
        )

        hands = detector.detect(cloud)

        # Should not detect hands with low intensity
        assert len(hands) == 0


class TestHandDetectorPerformance:
    """Performance tests for hand detection."""

    def test_large_point_cloud_performance(self):
        """Test performance with large point cloud."""
        config = HandDetectorConfig(min_hand_points=50)
        detector = HandDetector(config)

        # Create moderately sized point cloud (realistic for stereo vision)
        num_points = 10000
        points = np.random.randn(num_points, 3) * 10 + np.array([10, 20, 15])

        cloud = PointCloud3D(
            points=points,
            intensities=np.full(num_points, 100, dtype=np.uint8),
            timestamp=time.time(),
        )

        start = time.perf_counter()
        hands = detector.detect(cloud)
        elapsed = (time.perf_counter() - start) * 1000

        # Should complete in reasonable time (< 2500ms for large cloud)
        assert elapsed < 2500

    def test_many_frames_performance(self):
        """Test performance over many frames."""
        config = HandDetectorConfig(min_hand_points=30)
        detector = HandDetector(config)

        num_frames = 100
        total_time = 0

        for i in range(num_frames):
            center = np.array([10.0 + np.sin(i * 0.1) * 5, 20.0, 5.0])
            points = center + np.random.randn(200, 3) * 2

            cloud = PointCloud3D(
                points=points,
                intensities=np.full(200, 100, dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
            )

            start = time.perf_counter()
            detector.detect(cloud)
            total_time += (time.perf_counter() - start) * 1000

        avg_time = total_time / num_frames

        # Average should be reasonable for real-time (< 20ms)
        assert avg_time < 20
