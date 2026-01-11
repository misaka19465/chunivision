"""
Tests for TouchDetector module.
"""

import time

import numpy as np
import pytest

from chunivision.config.zone_config import ZoneConfig
from chunivision.vision.hand_detector import Hand
from chunivision.vision.touch_detector import (
    TouchDetector,
    TouchDetectorConfig,
    TouchState,
)


class TestTouchState:
    """Tests for TouchState dataclass."""

    def test_touch_state_creation_default(self):
        """Test default touch state creation."""
        state = TouchState()

        assert state.zones.shape == (32,)
        assert not any(state.zones)
        assert state.get_touched_count() == 0
        assert state.active_zone_ids == []

    def test_touch_state_with_touches(self):
        """Test touch state with active touches."""
        zones = np.zeros(32, dtype=bool)
        zones[0] = True  # Zone 1
        zones[4] = True  # Zone 5

        state = TouchState(zones=zones, timestamp=1234567890.0)

        assert state.get_touched_count() == 2
        assert state.is_touched(1)
        assert not state.is_touched(2)
        assert state.is_touched(5)
        assert 1 in state.active_zone_ids
        assert 5 in state.active_zone_ids

    def test_touch_state_from_list(self):
        """Test touch state creation from list."""
        zones_list = [True] + [False] * 31

        state = TouchState(zones=zones_list)

        assert isinstance(state.zones, np.ndarray)
        assert state.is_touched(1)

    def test_touch_state_invalid_zone_id(self):
        """Test is_touched with invalid zone ID."""
        state = TouchState()

        with pytest.raises(ValueError, match="Zone ID must be 1-32"):
            state.is_touched(0)

        with pytest.raises(ValueError, match="Zone ID must be 1-32"):
            state.is_touched(33)

    def test_touch_state_invalid_shape(self):
        """Test touch state with invalid zones shape."""
        with pytest.raises(ValueError, match="zones must have shape"):
            TouchState(zones=np.zeros(16, dtype=bool))

    def test_touch_state_to_bytes(self):
        """Test conversion to packed bytes."""
        zones = np.zeros(32, dtype=bool)
        zones[0] = True  # Zone 1 -> bit 0 of byte 0
        zones[7] = True  # Zone 8 -> bit 7 of byte 0
        zones[8] = True  # Zone 9 -> bit 0 of byte 1

        state = TouchState(zones=zones)
        data = state.to_bytes()

        assert len(data) == 4
        assert data[0] == 0b10000001  # bits 0 and 7
        assert data[1] == 0b00000001  # bit 0

    def test_touch_state_from_bytes(self):
        """Test creation from packed bytes."""
        data = bytes([0b10000001, 0b00000001, 0, 0])

        state = TouchState.from_bytes(data, timestamp=123.0)

        assert state.is_touched(1)
        assert state.is_touched(8)
        assert state.is_touched(9)
        assert state.timestamp == 123.0

    def test_touch_state_from_bytes_roundtrip(self):
        """Test bytes conversion roundtrip."""
        zones = np.zeros(32, dtype=bool)
        zones[0] = True
        zones[15] = True
        zones[16] = True
        zones[31] = True

        original = TouchState(zones=zones)
        data = original.to_bytes()
        restored = TouchState.from_bytes(data)

        np.testing.assert_array_equal(original.zones, restored.zones)

    def test_touch_state_repr(self):
        """Test string representation."""
        zones = np.zeros(32, dtype=bool)
        zones[0] = True
        zones[4] = True

        state = TouchState(zones=zones)
        repr_str = repr(state)

        assert "touched=2" in repr_str
        assert "1" in repr_str
        assert "5" in repr_str


class TestTouchDetectorConfig:
    """Tests for TouchDetectorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TouchDetectorConfig()

        assert config.touch_threshold_z == 2.0
        assert config.release_threshold_z == 3.0
        assert config.min_confidence == 0.3
        assert config.debounce_frames == 2
        assert config.enable_multi_touch

    def test_custom_config(self):
        """Test custom configuration."""
        config = TouchDetectorConfig(
            touch_threshold_z=1.5,
            release_threshold_z=2.5,
            min_confidence=0.5,
            debounce_frames=3,
        )

        assert config.touch_threshold_z == 1.5
        assert config.release_threshold_z == 2.5
        assert config.min_confidence == 0.5
        assert config.debounce_frames == 3

    def test_config_validation_success(self):
        """Test valid configuration passes validation."""
        config = TouchDetectorConfig()
        errors = config.validate()

        assert errors == []

    def test_config_validation_negative_threshold(self):
        """Test validation with negative touch threshold."""
        config = TouchDetectorConfig(touch_threshold_z=-1.0)
        errors = config.validate()

        assert any("touch_threshold_z" in e for e in errors)

    def test_config_validation_invalid_hysteresis(self):
        """Test validation with invalid hysteresis (release < touch)."""
        config = TouchDetectorConfig(touch_threshold_z=3.0, release_threshold_z=2.0)
        errors = config.validate()

        assert any("release_threshold_z" in e for e in errors)

    def test_config_validation_invalid_confidence(self):
        """Test validation with invalid confidence value."""
        config = TouchDetectorConfig(min_confidence=1.5)
        errors = config.validate()

        assert any("min_confidence" in e for e in errors)

    def test_config_validation_invalid_overlap_priority(self):
        """Test validation with invalid overlap priority."""
        config = TouchDetectorConfig(overlap_priority="invalid")
        errors = config.validate()

        assert any("overlap_priority" in e for e in errors)


class TestTouchDetector:
    """Tests for TouchDetector class."""

    @pytest.fixture
    def zone_config(self):
        """Create default zone configuration."""
        return ZoneConfig()

    @pytest.fixture
    def detector(self, zone_config):
        """Create default touch detector."""
        config = TouchDetectorConfig(debounce_frames=0)  # Disable debouncing for tests
        return TouchDetector(zone_config=zone_config, config=config)

    @pytest.fixture
    def detector_with_debounce(self, zone_config):
        """Create touch detector with debouncing enabled."""
        config = TouchDetectorConfig(debounce_frames=2)
        return TouchDetector(zone_config=zone_config, config=config)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.zone_config is not None
        assert detector.config is not None
        assert len(detector._zone_boundaries) == 32
        assert len(detector._zone_centers) == 32

    def test_detector_initialization_defaults(self):
        """Test detector with default configurations."""
        detector = TouchDetector()

        assert detector.zone_config is not None
        assert detector.config is not None

    def test_detector_invalid_config(self):
        """Test detector rejects invalid configuration."""
        invalid_config = TouchDetectorConfig(touch_threshold_z=-1.0)

        with pytest.raises(ValueError, match="Invalid configuration"):
            TouchDetector(config=invalid_config)

    def test_detect_no_hands(self, detector):
        """Test detection with no hands."""
        state = detector.detect_touches([])

        assert state.get_touched_count() == 0
        assert isinstance(state, TouchState)

    def test_detect_single_touch_zone1(self, detector, zone_config):
        """Test single touch in zone 1 (bottom-right)."""
        # Zone 1 is at row=0, col=15 (bottom-right)
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),  # Below threshold
            confidence=0.9,
        )

        state = detector.detect_touches([hand])

        assert state.is_touched(1)
        assert state.get_touched_count() == 1

    def test_detect_single_touch_zone32(self, detector, zone_config):
        """Test single touch in zone 32 (top-left)."""
        center = zone_config.get_zone_center(32)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.9,
        )

        state = detector.detect_touches([hand])

        assert state.is_touched(32)
        assert state.get_touched_count() == 1

    def test_detect_hand_above_threshold(self, detector, zone_config):
        """Test hand above touch threshold is not registered."""
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 5.0]),  # Above threshold (2.0)
            confidence=0.9,
        )

        state = detector.detect_touches([hand])

        assert not state.is_touched(1)
        assert state.get_touched_count() == 0

    def test_detect_low_confidence_filtered(self, detector, zone_config):
        """Test low confidence hands are filtered."""
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.1,  # Below min_confidence (0.3)
        )

        state = detector.detect_touches([hand])

        assert state.get_touched_count() == 0

    def test_detect_hand_outside_zones(self, detector):
        """Test hand outside zone grid is not registered."""
        # Position far outside grid
        hand = Hand(
            position=np.array([100.0, 100.0, 1.0]),
            confidence=0.9,
        )

        state = detector.detect_touches([hand])

        assert state.get_touched_count() == 0

    def test_detect_multi_touch(self, detector, zone_config):
        """Test multiple simultaneous touches."""
        center1 = zone_config.get_zone_center(1)
        center16 = zone_config.get_zone_center(16)

        hands = [
            Hand(position=np.array([center1[0], center1[1], 1.0]), confidence=0.9),
            Hand(position=np.array([center16[0], center16[1], 1.0]), confidence=0.9),
        ]

        state = detector.detect_touches(hands)

        assert state.is_touched(1)
        assert state.is_touched(16)
        assert state.get_touched_count() == 2

    def test_detect_touch_positions_recorded(self, detector, zone_config):
        """Test touch positions are recorded."""
        center = zone_config.get_zone_center(1)
        pos = np.array([center[0], center[1], 0.5])
        hand = Hand(position=pos, confidence=0.9)

        state = detector.detect_touches([hand])

        assert state.touch_positions is not None
        assert len(state.touch_positions) == 1
        np.testing.assert_array_almost_equal(state.touch_positions[0], pos)

    def test_debouncing_prevents_immediate_touch(
        self, detector_with_debounce, zone_config
    ):
        """Test debouncing prevents immediate touch registration."""
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.9,
        )

        # First frame - should not register yet
        state1 = detector_with_debounce.detect_touches([hand])
        assert not state1.is_touched(1)

        # Second frame - should register now (debounce_frames=2)
        state2 = detector_with_debounce.detect_touches([hand])
        assert state2.is_touched(1)

    def test_debouncing_prevents_immediate_release(
        self, detector_with_debounce, zone_config
    ):
        """Test debouncing prevents immediate release."""
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.9,
        )

        # Establish touch
        detector_with_debounce.detect_touches([hand])
        detector_with_debounce.detect_touches([hand])

        # Remove hand - should not release immediately
        state1 = detector_with_debounce.detect_touches([])
        assert state1.is_touched(1)

        # Second frame without hand - should release
        state2 = detector_with_debounce.detect_touches([])
        assert not state2.is_touched(1)

    def test_set_touch_threshold(self, detector):
        """Test setting touch threshold."""
        detector.set_touch_threshold(3.0)

        assert detector.config.touch_threshold_z == 3.0
        assert detector.config.release_threshold_z >= 3.0

    def test_set_touch_threshold_negative(self, detector):
        """Test setting negative touch threshold raises error."""
        with pytest.raises(ValueError, match="z_threshold must be >= 0"):
            detector.set_touch_threshold(-1.0)

    def test_set_release_threshold(self, detector):
        """Test setting release threshold."""
        detector.set_release_threshold(4.0)

        assert detector.config.release_threshold_z == 4.0

    def test_set_release_threshold_invalid(self, detector):
        """Test setting release threshold below touch threshold raises error."""
        detector.config.touch_threshold_z = 3.0

        with pytest.raises(
            ValueError, match="release_threshold must be >= touch_threshold"
        ):
            detector.set_release_threshold(2.0)

    def test_get_zone_boundary(self, detector):
        """Test getting zone boundary."""
        boundary = detector.get_zone_boundary(1)

        assert boundary.shape == (4, 2)
        # Returned boundary should be a copy
        boundary[0, 0] = 999
        assert detector.get_zone_boundary(1)[0, 0] != 999

    def test_get_zone_boundary_invalid(self, detector):
        """Test getting boundary for invalid zone."""
        with pytest.raises(ValueError, match="Zone ID must be 1-32"):
            detector.get_zone_boundary(0)

        with pytest.raises(ValueError, match="Zone ID must be 1-32"):
            detector.get_zone_boundary(33)

    def test_get_zone_center(self, detector):
        """Test getting zone center."""
        center = detector.get_zone_center(1)

        assert center.shape == (2,)

    def test_get_zone_center_invalid(self, detector):
        """Test getting center for invalid zone."""
        with pytest.raises(ValueError, match="Zone ID must be 1-32"):
            detector.get_zone_center(0)

    def test_get_current_state(self, detector, zone_config):
        """Test getting current state."""
        center = zone_config.get_zone_center(5)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.9,
        )

        detector.detect_touches([hand])
        state = detector.get_current_state()

        assert state.is_touched(5)

    def test_get_state_history(self, detector, zone_config):
        """Test state history is maintained."""
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.9,
        )

        detector.detect_touches([])
        detector.detect_touches([hand])
        detector.detect_touches([hand])

        history = detector.get_state_history()

        assert len(history) == 3

    def test_reset(self, detector, zone_config):
        """Test reset clears all state."""
        center = zone_config.get_zone_center(1)
        hand = Hand(
            position=np.array([center[0], center[1], 1.0]),
            confidence=0.9,
        )

        detector.detect_touches([hand])
        detector.reset()

        state = detector.get_current_state()
        history = detector.get_state_history()

        assert state.get_touched_count() == 0
        assert len(history) == 0

    def test_get_zone_for_position(self, detector, zone_config):
        """Test getting zone ID for a position."""
        # Get center of zone 1 and verify
        center = zone_config.get_zone_center(1)
        zone_id = detector.get_zone_for_position(center[0], center[1])

        assert zone_id == 1

    def test_get_zone_for_position_outside(self, detector):
        """Test position outside zones returns 0."""
        zone_id = detector.get_zone_for_position(100.0, 100.0)

        assert zone_id == 0


class TestTouchDetectorZoneMapping:
    """Tests for zone mapping accuracy."""

    @pytest.fixture
    def zone_config(self):
        """Create default zone configuration."""
        return ZoneConfig()

    @pytest.fixture
    def detector(self, zone_config):
        """Create touch detector."""
        config = TouchDetectorConfig(debounce_frames=0)
        return TouchDetector(zone_config=zone_config, config=config)

    def test_all_zones_reachable(self, detector, zone_config):
        """Test all 32 zones can be touched."""
        for zone_id in range(1, 33):
            center = zone_config.get_zone_center(zone_id)
            hand = Hand(
                position=np.array([center[0], center[1], 1.0]),
                confidence=0.9,
            )

            state = detector.detect_touches([hand])

            assert state.is_touched(zone_id), f"Zone {zone_id} should be touched"
            detector.reset()

    def test_zone_boundary_touches(self, detector, zone_config):
        """Test touches at zone boundaries."""
        # Test at boundary between zone 1 and zone 3
        boundary1 = zone_config.get_zone_boundary(1)
        boundary3 = zone_config.get_zone_boundary(3)

        # Point should map to one zone or the other
        mid_x = (boundary1[0, 0] + boundary3[1, 0]) / 2
        mid_y = (boundary1[0, 1] + boundary1[2, 1]) / 2

        hand = Hand(
            position=np.array([mid_x, mid_y, 1.0]),
            confidence=0.9,
        )

        state = detector.detect_touches([hand])

        # Should touch exactly one zone at the boundary
        assert state.get_touched_count() >= 1


class TestTouchDetectorPerformance:
    """Performance-related tests for TouchDetector."""

    @pytest.fixture
    def detector(self):
        """Create touch detector."""
        config = TouchDetectorConfig(debounce_frames=0)
        return TouchDetector(config=config)

    def test_detect_many_hands_performance(self, detector):
        """Test detection with many hands is reasonably fast."""
        # Create 10 hands at random positions
        hands = []
        np.random.seed(42)
        for _ in range(10):
            pos = np.array(
                [
                    np.random.uniform(-20, 20),
                    np.random.uniform(0, 9),
                    np.random.uniform(0, 5),
                ]
            )
            hands.append(Hand(position=pos, confidence=0.9))

        start = time.time()
        for _ in range(100):
            detector.detect_touches(hands)
        elapsed = time.time() - start

        # Should process 100 frames with 10 hands in under 100ms
        assert elapsed < 0.1, f"Detection too slow: {elapsed:.3f}s for 100 frames"

    def test_state_history_limit(self, detector):
        """Test state history doesn't grow unbounded."""
        for _ in range(200):
            detector.detect_touches([])

        history = detector.get_state_history()

        assert len(history) <= 100  # _max_history
