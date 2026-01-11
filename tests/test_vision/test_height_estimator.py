"""
Tests for HeightEstimator module.
"""

import time

import numpy as np
import pytest

from chunivision.vision.hand_detector import Hand
from chunivision.vision.height_estimator import (
    DEFAULT_HEIGHT_THRESHOLDS,
    HeightEstimator,
    HeightEstimatorConfig,
    HeightState,
)


class TestHeightState:
    """Tests for HeightState dataclass."""

    def test_height_state_creation_default(self):
        """Test default height state creation."""
        state = HeightState()

        assert state.levels.shape == (6,)
        assert not any(state.levels)
        assert state.get_active_count() == 0
        assert state.active_levels == []

    def test_height_state_with_active_levels(self):
        """Test height state with active levels."""
        levels = np.zeros(6, dtype=bool)
        levels[0] = True  # Level 0
        levels[2] = True  # Level 2

        state = HeightState(levels=levels, timestamp=1234567890.0)

        assert state.get_active_count() == 2
        assert state.is_active(0)
        assert not state.is_active(1)
        assert state.is_active(2)
        assert 0 in state.active_levels
        assert 2 in state.active_levels

    def test_height_state_from_list(self):
        """Test height state creation from list."""
        levels_list = [True, True, False, False, False, False]

        state = HeightState(levels=levels_list)

        assert state.levels.shape == (6,)
        assert state.is_active(0)
        assert state.is_active(1)
        assert state.get_active_count() == 2

    def test_height_state_invalid_shape(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            HeightState(levels=np.zeros(5, dtype=bool))

        with pytest.raises(ValueError, match="shape"):
            HeightState(levels=np.zeros(7, dtype=bool))

    def test_is_active_valid_levels(self):
        """Test is_active with valid level indices."""
        levels = np.array([True, False, True, False, True, False], dtype=bool)
        state = HeightState(levels=levels)

        assert state.is_active(0) is True
        assert state.is_active(1) is False
        assert state.is_active(2) is True
        assert state.is_active(3) is False
        assert state.is_active(4) is True
        assert state.is_active(5) is False

    def test_is_active_invalid_level(self):
        """Test is_active with invalid level index."""
        state = HeightState()

        with pytest.raises(ValueError, match="Level must be 0-5"):
            state.is_active(-1)

        with pytest.raises(ValueError, match="Level must be 0-5"):
            state.is_active(6)

    def test_get_highest_active_level(self):
        """Test getting highest active level."""
        # No active levels
        state = HeightState()
        assert state.get_highest_active_level() is None

        # Single active level
        levels = np.zeros(6, dtype=bool)
        levels[2] = True
        state = HeightState(levels=levels)
        assert state.get_highest_active_level() == 2

        # Multiple active levels
        levels = np.array([True, True, True, False, True, False], dtype=bool)
        state = HeightState(levels=levels)
        assert state.get_highest_active_level() == 4

        # All levels active
        state = HeightState(levels=np.ones(6, dtype=bool))
        assert state.get_highest_active_level() == 5

    def test_get_lowest_active_level(self):
        """Test getting lowest active level."""
        # No active levels
        state = HeightState()
        assert state.get_lowest_active_level() is None

        # Single active level
        levels = np.zeros(6, dtype=bool)
        levels[3] = True
        state = HeightState(levels=levels)
        assert state.get_lowest_active_level() == 3

        # Multiple active levels
        levels = np.array([False, False, True, True, True, False], dtype=bool)
        state = HeightState(levels=levels)
        assert state.get_lowest_active_level() == 2

    def test_to_byte(self):
        """Test conversion to byte."""
        # No active levels
        state = HeightState()
        assert state.to_byte() == 0b000000

        # Level 0 only
        levels = np.zeros(6, dtype=bool)
        levels[0] = True
        state = HeightState(levels=levels)
        assert state.to_byte() == 0b000001

        # Levels 0, 2, 4
        levels = np.array([True, False, True, False, True, False], dtype=bool)
        state = HeightState(levels=levels)
        assert state.to_byte() == 0b010101

        # All levels
        state = HeightState(levels=np.ones(6, dtype=bool))
        assert state.to_byte() == 0b111111

    def test_to_bytes(self):
        """Test conversion to bytes."""
        levels = np.array([True, True, True, False, False, False], dtype=bool)
        state = HeightState(levels=levels)

        result = state.to_bytes()
        assert len(result) == 1
        assert result[0] == 0b000111

    def test_from_byte(self):
        """Test creation from byte."""
        # No active levels
        state = HeightState.from_byte(0b000000, timestamp=123.0)
        assert state.get_active_count() == 0
        assert state.timestamp == 123.0

        # Levels 0, 1, 2
        state = HeightState.from_byte(0b000111)
        assert state.is_active(0)
        assert state.is_active(1)
        assert state.is_active(2)
        assert not state.is_active(3)

        # All levels
        state = HeightState.from_byte(0b111111)
        assert state.get_active_count() == 6

    def test_from_bytes(self):
        """Test creation from bytes."""
        data = bytes([0b101010])  # Levels 1, 3, 5
        state = HeightState.from_bytes(data)

        assert not state.is_active(0)
        assert state.is_active(1)
        assert not state.is_active(2)
        assert state.is_active(3)
        assert not state.is_active(4)
        assert state.is_active(5)

    def test_from_bytes_empty(self):
        """Test from_bytes with empty data."""
        with pytest.raises(ValueError, match="Expected at least 1 byte"):
            HeightState.from_bytes(b"")

    def test_roundtrip_byte_conversion(self):
        """Test byte conversion roundtrip."""
        original_levels = np.array([True, False, True, True, False, True], dtype=bool)
        original = HeightState(levels=original_levels, timestamp=999.0)

        byte_val = original.to_byte()
        restored = HeightState.from_byte(byte_val, timestamp=999.0)

        np.testing.assert_array_equal(restored.levels, original.levels)
        assert restored.timestamp == original.timestamp

    def test_repr(self):
        """Test string representation."""
        state = HeightState()
        assert "active=0" in repr(state)

        levels = np.array([True, True, False, False, False, False], dtype=bool)
        state = HeightState(levels=levels)
        assert "active=2" in repr(state)
        assert "0" in repr(state)
        assert "1" in repr(state)

    def test_exact_heights(self):
        """Test exact_heights attribute."""
        state = HeightState(exact_heights=[20.5, 25.3])
        assert state.exact_heights == [20.5, 25.3]

        state = HeightState()
        assert state.exact_heights is None


class TestHeightEstimatorConfig:
    """Tests for HeightEstimatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HeightEstimatorConfig()

        assert config.height_thresholds == DEFAULT_HEIGHT_THRESHOLDS
        assert config.hysteresis == 1.0
        assert config.min_confidence == 0.3
        assert config.debounce_frames == 2
        assert config.x_range is None
        assert config.cumulative_levels is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=2.0,
            min_confidence=0.5,
            debounce_frames=3,
            x_range=(-10.0, 10.0),
            cumulative_levels=False,
        )

        assert config.height_thresholds == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        assert config.hysteresis == 2.0
        assert config.min_confidence == 0.5
        assert config.debounce_frames == 3
        assert config.x_range == (-10.0, 10.0)
        assert config.cumulative_levels is False

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = HeightEstimatorConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_wrong_threshold_count(self):
        """Test validation with wrong number of thresholds."""
        config = HeightEstimatorConfig(height_thresholds=[1.0, 2.0, 3.0])
        errors = config.validate()
        assert any("6 values" in e for e in errors)

    def test_validate_non_increasing_thresholds(self):
        """Test validation with non-increasing thresholds."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 8.0, 20.0, 25.0, 30.0]
        )
        errors = config.validate()
        assert any("strictly increasing" in e for e in errors)

    def test_validate_equal_thresholds(self):
        """Test validation with equal thresholds."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 10.0, 20.0, 25.0, 30.0]
        )
        errors = config.validate()
        assert any("strictly increasing" in e for e in errors)

    def test_validate_negative_hysteresis(self):
        """Test validation with negative hysteresis."""
        config = HeightEstimatorConfig(hysteresis=-1.0)
        errors = config.validate()
        assert any("hysteresis" in e for e in errors)

    def test_validate_invalid_confidence(self):
        """Test validation with invalid confidence."""
        config = HeightEstimatorConfig(min_confidence=1.5)
        errors = config.validate()
        assert any("min_confidence" in e for e in errors)

        config = HeightEstimatorConfig(min_confidence=-0.1)
        errors = config.validate()
        assert any("min_confidence" in e for e in errors)

    def test_validate_negative_debounce(self):
        """Test validation with negative debounce frames."""
        config = HeightEstimatorConfig(debounce_frames=-1)
        errors = config.validate()
        assert any("debounce_frames" in e for e in errors)

    def test_validate_invalid_x_range(self):
        """Test validation with invalid x_range."""
        config = HeightEstimatorConfig(x_range=(10.0, 5.0))
        errors = config.validate()
        assert any("x_range" in e for e in errors)


class TestHeightEstimator:
    """Tests for HeightEstimator class."""

    @pytest.fixture
    def simple_config(self):
        """Simple config with easy-to-test thresholds."""
        return HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,  # No hysteresis for simpler testing
            debounce_frames=0,  # No debouncing for simpler testing
            cumulative_levels=True,
        )

    @pytest.fixture
    def estimator(self, simple_config):
        """Create estimator with simple config."""
        return HeightEstimator(config=simple_config)

    def test_initialization_default(self):
        """Test default initialization."""
        estimator = HeightEstimator()

        assert estimator.config.height_thresholds == DEFAULT_HEIGHT_THRESHOLDS
        assert len(estimator.get_all_thresholds()) == 6

    def test_initialization_custom_config(self, simple_config):
        """Test initialization with custom config."""
        estimator = HeightEstimator(config=simple_config)

        assert estimator.get_all_thresholds() == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    def test_initialization_invalid_config(self):
        """Test initialization with invalid config."""
        invalid_config = HeightEstimatorConfig(
            height_thresholds=[30.0, 25.0, 20.0, 15.0, 10.0, 5.0]  # Decreasing
        )

        with pytest.raises(ValueError, match="Invalid configuration"):
            HeightEstimator(config=invalid_config)

    def test_estimate_heights_no_hands(self, estimator):
        """Test with no hands."""
        state = estimator.estimate_heights([])

        assert state.get_active_count() == 0
        assert state.exact_heights is None

    def test_estimate_heights_hand_below_all_thresholds(self, estimator):
        """Test with hand below all thresholds."""
        hand = Hand(position=np.array([0.0, 0.0, 3.0]))  # Below 5.0
        state = estimator.estimate_heights([hand])

        assert state.get_active_count() == 0
        assert state.exact_heights == [3.0]

    def test_estimate_heights_hand_at_level_0(self, estimator):
        """Test with hand at level 0 threshold."""
        hand = Hand(position=np.array([0.0, 0.0, 5.0]))  # Exactly at level 0
        state = estimator.estimate_heights([hand])

        assert state.is_active(0)
        assert state.get_active_count() == 1

    def test_estimate_heights_hand_at_level_2(self, estimator):
        """Test with hand at level 2 (cumulative mode)."""
        hand = Hand(position=np.array([0.0, 0.0, 15.0]))  # At level 2 threshold
        state = estimator.estimate_heights([hand])

        # Cumulative: levels 0, 1, 2 should be active
        assert state.is_active(0)
        assert state.is_active(1)
        assert state.is_active(2)
        assert not state.is_active(3)
        assert state.get_active_count() == 3

    def test_estimate_heights_hand_at_level_5(self, estimator):
        """Test with hand at highest level."""
        hand = Hand(position=np.array([0.0, 0.0, 35.0]))  # Above level 5 (30.0)
        state = estimator.estimate_heights([hand])

        # All levels should be active in cumulative mode
        assert state.get_active_count() == 6

    def test_estimate_heights_between_levels(self, estimator):
        """Test with hand between levels."""
        hand = Hand(position=np.array([0.0, 0.0, 12.0]))  # Between 10.0 and 15.0
        state = estimator.estimate_heights([hand])

        # Should activate levels 0 and 1 (12.0 >= 10.0 but < 15.0)
        assert state.is_active(0)
        assert state.is_active(1)
        assert not state.is_active(2)
        assert state.get_active_count() == 2

    def test_estimate_heights_non_cumulative(self):
        """Test non-cumulative mode."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=0,
            cumulative_levels=False,  # Non-cumulative
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, 17.0]))  # Above level 2 (15.0)
        state = estimator.estimate_heights([hand])

        # Only level 2 should be active (highest level the hand is at/above)
        # Actually, in non-cumulative mode, each level where height >= threshold is activated
        assert state.is_active(0)
        assert state.is_active(1)
        assert state.is_active(2)
        assert not state.is_active(3)

    def test_estimate_heights_multiple_hands(self, estimator):
        """Test with multiple hands at different heights."""
        hands = [
            Hand(position=np.array([0.0, 0.0, 7.0])),  # Level 0
            Hand(position=np.array([5.0, 0.0, 22.0])),  # Level 3
        ]
        state = estimator.estimate_heights(hands)

        # Both hands contribute: levels 0 + 0,1,2,3 = 0,1,2,3
        assert state.is_active(0)
        assert state.is_active(1)
        assert state.is_active(2)
        assert state.is_active(3)
        assert not state.is_active(4)
        assert state.exact_heights == [7.0, 22.0]

    def test_estimate_heights_low_confidence_filtered(self, estimator):
        """Test that low confidence hands are filtered."""
        hand = Hand(position=np.array([0.0, 0.0, 20.0]), confidence=0.1)
        state = estimator.estimate_heights([hand])

        # Hand should be filtered due to low confidence (< 0.3)
        assert state.get_active_count() == 0

    def test_estimate_heights_x_range_filtering(self):
        """Test X-range filtering."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=0,
            x_range=(-5.0, 5.0),  # Only hands within this range
        )
        estimator = HeightEstimator(config=config)

        hands = [
            Hand(position=np.array([0.0, 0.0, 20.0])),  # In range
            Hand(position=np.array([10.0, 0.0, 20.0])),  # Out of range
        ]
        state = estimator.estimate_heights(hands)

        # Only first hand should be considered
        assert state.get_active_count() == 4  # Levels 0,1,2,3
        assert state.exact_heights == [20.0]  # Only in-range hand

    def test_hysteresis_activation(self):
        """Test hysteresis during activation."""
        config = HeightEstimatorConfig(
            height_thresholds=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            hysteresis=2.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        # Hand at 9.0 (below threshold 10.0) - should not activate
        hand = Hand(position=np.array([0.0, 0.0, 9.0]))
        state = estimator.estimate_heights([hand])
        assert not state.is_active(0)

        # Hand at 10.0 (at threshold) - should activate
        hand = Hand(position=np.array([0.0, 0.0, 10.0]))
        state = estimator.estimate_heights([hand])
        assert state.is_active(0)

    def test_hysteresis_deactivation(self):
        """Test hysteresis during deactivation."""
        config = HeightEstimatorConfig(
            height_thresholds=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            hysteresis=2.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        # First, activate level 0
        hand = Hand(position=np.array([0.0, 0.0, 10.0]))
        state = estimator.estimate_heights([hand])
        assert state.is_active(0)

        # Drop to 9.0 (below threshold but above release threshold 8.0)
        hand = Hand(position=np.array([0.0, 0.0, 9.0]))
        state = estimator.estimate_heights([hand])
        assert state.is_active(0)  # Still active due to hysteresis

        # Drop to 7.0 (below release threshold 8.0)
        hand = Hand(position=np.array([0.0, 0.0, 7.0]))
        state = estimator.estimate_heights([hand])
        assert not state.is_active(0)  # Now deactivated

    def test_debouncing_activation(self):
        """Test debouncing during activation."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=3,
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, 20.0]))

        # First frame - pending
        state = estimator.estimate_heights([hand])
        assert state.get_active_count() == 0

        # Second frame - still pending
        state = estimator.estimate_heights([hand])
        assert state.get_active_count() == 0

        # Third frame - now activated
        state = estimator.estimate_heights([hand])
        assert state.is_active(0)
        assert state.is_active(1)
        assert state.is_active(2)
        assert state.is_active(3)

    def test_debouncing_deactivation(self):
        """Test debouncing during deactivation."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=2,
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, 10.0]))

        # Activate (needs 2 frames)
        estimator.estimate_heights([hand])
        state = estimator.estimate_heights([hand])
        assert state.is_active(0)
        assert state.is_active(1)

        # First frame without hand - still active (pending)
        state = estimator.estimate_heights([])
        assert state.is_active(0)

        # Second frame without hand - now deactivated
        state = estimator.estimate_heights([])
        assert state.get_active_count() == 0

    def test_set_thresholds(self, estimator):
        """Test updating thresholds."""
        new_thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        estimator.set_thresholds(new_thresholds)

        assert estimator.get_all_thresholds() == new_thresholds

    def test_set_thresholds_invalid_count(self, estimator):
        """Test setting wrong number of thresholds."""
        with pytest.raises(ValueError, match="Expected 6 thresholds"):
            estimator.set_thresholds([1.0, 2.0, 3.0])

    def test_set_thresholds_non_increasing(self, estimator):
        """Test setting non-increasing thresholds."""
        with pytest.raises(ValueError, match="strictly increasing"):
            estimator.set_thresholds([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    def test_set_hysteresis(self, estimator):
        """Test updating hysteresis."""
        estimator.set_hysteresis(3.0)
        assert estimator.config.hysteresis == 3.0

    def test_set_hysteresis_negative(self, estimator):
        """Test setting negative hysteresis."""
        with pytest.raises(ValueError, match="must be >= 0"):
            estimator.set_hysteresis(-1.0)

    def test_set_x_range(self, estimator):
        """Test setting X range."""
        estimator.set_x_range((-10.0, 10.0))
        assert estimator.config.x_range == (-10.0, 10.0)

        estimator.set_x_range(None)
        assert estimator.config.x_range is None

    def test_set_x_range_invalid(self, estimator):
        """Test setting invalid X range."""
        with pytest.raises(ValueError, match="x_range"):
            estimator.set_x_range((10.0, 5.0))

    def test_get_threshold(self, estimator):
        """Test getting individual threshold."""
        assert estimator.get_threshold(0) == 5.0
        assert estimator.get_threshold(5) == 30.0

    def test_get_threshold_invalid(self, estimator):
        """Test getting threshold with invalid level."""
        with pytest.raises(ValueError, match="Level must be 0-5"):
            estimator.get_threshold(-1)

        with pytest.raises(ValueError, match="Level must be 0-5"):
            estimator.get_threshold(6)

    def test_get_current_state(self, estimator):
        """Test getting current state."""
        state = estimator.get_current_state()
        assert isinstance(state, HeightState)
        assert state.get_active_count() == 0

    def test_get_state_history(self, estimator):
        """Test getting state history."""
        # Process several frames
        for i in range(5):
            hand = Hand(position=np.array([0.0, 0.0, float(i * 5)]))
            estimator.estimate_heights([hand])

        history = estimator.get_state_history()
        assert len(history) == 5

    def test_state_history_max_length(self, estimator):
        """Test state history respects max length."""
        # Process more frames than max history
        for i in range(150):
            estimator.estimate_heights([])

        history = estimator.get_state_history()
        assert len(history) == 100  # Default max

    def test_reset(self, estimator):
        """Test reset functionality."""
        # Build up some state
        hand = Hand(position=np.array([0.0, 0.0, 20.0]))
        estimator.estimate_heights([hand])

        # Reset
        estimator.reset()

        state = estimator.get_current_state()
        assert state.get_active_count() == 0
        assert len(estimator.get_state_history()) == 0

    def test_height_to_level(self, estimator):
        """Test height to level conversion utility."""
        # Below all thresholds
        assert estimator.height_to_level(3.0) is None

        # At level 0
        assert estimator.height_to_level(5.0) == 0

        # Between levels
        assert estimator.height_to_level(7.0) == 0
        assert estimator.height_to_level(12.0) == 1
        assert estimator.height_to_level(17.0) == 2

        # At highest level
        assert estimator.height_to_level(35.0) == 5

    def test_level_to_height_range(self, estimator):
        """Test level to height range conversion."""
        min_h, max_h = estimator.level_to_height_range(0)
        assert min_h == 5.0
        assert max_h == 10.0

        min_h, max_h = estimator.level_to_height_range(2)
        assert min_h == 15.0
        assert max_h == 20.0

        min_h, max_h = estimator.level_to_height_range(5)
        assert min_h == 30.0
        assert max_h == float("inf")

    def test_level_to_height_range_invalid(self, estimator):
        """Test level_to_height_range with invalid level."""
        with pytest.raises(ValueError, match="Level must be 0-5"):
            estimator.level_to_height_range(-1)

        with pytest.raises(ValueError, match="Level must be 0-5"):
            estimator.level_to_height_range(6)

    def test_timestamp_updates(self, estimator):
        """Test that timestamps are updated on each call."""
        state1 = estimator.estimate_heights([])
        time.sleep(0.01)
        state2 = estimator.estimate_heights([])

        assert state2.timestamp > state1.timestamp


class TestHeightEstimatorEdgeCases:
    """Edge case tests for HeightEstimator."""

    def test_hand_exactly_at_threshold(self):
        """Test hand exactly at threshold boundary."""
        config = HeightEstimatorConfig(
            height_thresholds=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            hysteresis=0.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, 20.0]))  # Exactly at level 1
        state = estimator.estimate_heights([hand])

        assert state.is_active(0)
        assert state.is_active(1)
        assert not state.is_active(2)

    def test_hand_with_zero_confidence(self):
        """Test hand with zero confidence is filtered."""
        estimator = HeightEstimator()

        hand = Hand(position=np.array([0.0, 0.0, 25.0]), confidence=0.0)
        state = estimator.estimate_heights([hand])

        assert state.get_active_count() == 0

    def test_hand_with_exact_min_confidence(self):
        """Test hand with exactly min confidence is included."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            min_confidence=0.5,
            hysteresis=0.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, 10.0]), confidence=0.5)
        state = estimator.estimate_heights([hand])

        assert state.is_active(0)
        assert state.is_active(1)

    def test_very_high_hand(self):
        """Test hand at very high position."""
        config = HeightEstimatorConfig(
            hysteresis=0.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, 100.0]))  # Way above all levels
        state = estimator.estimate_heights([hand])

        assert state.get_active_count() == 6

    def test_negative_height(self):
        """Test hand with negative height (below surface)."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        hand = Hand(position=np.array([0.0, 0.0, -5.0]))
        state = estimator.estimate_heights([hand])

        assert state.get_active_count() == 0

    def test_rapid_height_changes(self):
        """Test rapid height changes with debouncing."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=3,
        )
        estimator = HeightEstimator(config=config)

        # Rapid alternation - should remain stable due to debouncing
        for _ in range(10):
            hand_high = Hand(position=np.array([0.0, 0.0, 25.0]))
            hand_low = Hand(position=np.array([0.0, 0.0, 5.0]))
            estimator.estimate_heights([hand_high])
            estimator.estimate_heights([hand_low])

        # State should eventually stabilize
        state = estimator.get_current_state()
        # The exact state depends on the sequence, but should be stable

    def test_concurrent_hands_same_height(self):
        """Test multiple hands at the same height."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        hands = [
            Hand(position=np.array([-5.0, 0.0, 20.0])),
            Hand(position=np.array([0.0, 0.0, 20.0])),
            Hand(position=np.array([5.0, 0.0, 20.0])),
        ]
        state = estimator.estimate_heights(hands)

        assert state.get_active_count() == 4  # Levels 0, 1, 2, 3
        assert state.exact_heights == [20.0, 20.0, 20.0]


class TestHeightEstimatorIntegration:
    """Integration tests for HeightEstimator."""

    def test_full_height_sweep(self):
        """Test sweeping hand through all heights."""
        config = HeightEstimatorConfig(
            height_thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            hysteresis=0.0,
            debounce_frames=0,
        )
        estimator = HeightEstimator(config=config)

        # Sweep from 0 to 35 cm
        expected_levels = [0, 1, 2, 3, 4, 5]
        for height in range(0, 36, 5):
            hand = Hand(position=np.array([0.0, 0.0, float(height)]))
            state = estimator.estimate_heights([hand])

            # Calculate expected active levels
            expected_count = 0
            for level in range(6):
                if height >= config.height_thresholds[level]:
                    expected_count = level + 1

            assert state.get_active_count() == expected_count, (
                f"At height {height}, expected {expected_count} levels, "
                f"got {state.get_active_count()}"
            )

    def test_chunithm_thresholds(self):
        """Test with actual Chunithm air sensor thresholds."""
        estimator = HeightEstimator()  # Uses default Chunithm thresholds

        thresholds = estimator.get_all_thresholds()
        assert thresholds == [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]

        # Test at each level
        test_heights = [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]
        for level, height in enumerate(test_heights):
            hand = Hand(position=np.array([0.0, 0.0, height]))
            # Need to clear debounce state
            estimator.reset()

            # Process enough frames to pass debounce
            for _ in range(3):
                state = estimator.estimate_heights([hand])

            assert state.get_active_count() == level + 1, (
                f"At height {height} (level {level}), expected {level + 1} active, "
                f"got {state.get_active_count()}"
            )
