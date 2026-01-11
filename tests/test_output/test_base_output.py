"""
Unit tests for BaseOutput abstract class.
"""

import time
from typing import Any, Dict

import pytest

from chunivision.output.base_output import BaseOutput, OutputError, OutputStats
from chunivision.vision.height_estimator import HeightState
from chunivision.vision.touch_detector import TouchState


# Concrete implementation for testing
class DummyOutput(BaseOutput):
    """Test implementation of BaseOutput."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str = "DummyOutput",
        should_fail: bool = False,
    ):
        self.should_fail = should_fail
        self.initialized_called = False
        self.send_called = False
        self.close_called = False
        self.send_count = 0
        super().__init__(config, name)

    def initialize(self) -> bool:
        self.initialized_called = True
        if self.should_fail:
            return False
        self._set_initialized(True)
        self._set_connected(True)
        return True

    def send_state(self, touch_state: TouchState, height_state: HeightState) -> bool:
        self.send_called = True
        self.send_count += 1

        if self.should_fail:
            self._update_stats_failure("Dummy error")
            return False

        # Simulate sending 38 bytes (32 touch + 6 height)
        start = time.time()
        time.sleep(0.001)  # Simulate send delay
        duration = (time.time() - start) * 1000
        self._update_stats_success(38, duration)
        return True

    def close(self) -> None:
        self.close_called = True
        self._set_connected(False)
        self._set_initialized(False)


class TestOutputStats:
    """Tests for OutputStats dataclass."""

    def test_default_initialization(self):
        """Test OutputStats default values."""
        stats = OutputStats()
        assert stats.packets_sent == 0
        assert stats.packets_failed == 0
        assert stats.bytes_sent == 0
        assert stats.last_send_time == 0.0
        assert stats.last_error_time == 0.0
        assert stats.last_error_message == ""
        assert stats.average_send_duration_ms == 0.0
        assert stats.is_connected is False

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = OutputStats(
            packets_sent=100, packets_failed=5, bytes_sent=10000, is_connected=True
        )
        d = stats.to_dict()

        assert d["packets_sent"] == 100
        assert d["packets_failed"] == 5
        assert d["bytes_sent"] == 10000
        assert d["is_connected"] is True
        assert "success_rate" in d
        assert d["success_rate"] == pytest.approx(100 / 105)

    def test_success_rate_calculation(self):
        """Test success rate calculation in to_dict."""
        # No packets
        stats = OutputStats()
        assert stats.to_dict()["success_rate"] == 0.0

        # All successful
        stats = OutputStats(packets_sent=10, packets_failed=0)
        assert stats.to_dict()["success_rate"] == 1.0

        # Some failures
        stats = OutputStats(packets_sent=8, packets_failed=2)
        assert stats.to_dict()["success_rate"] == 0.8


class TestBaseOutput:
    """Tests for BaseOutput abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseOutput cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseOutput({}, "test")  # type: ignore

    def test_dummy_output_creation(self):
        """Test creating a concrete implementation."""
        output = DummyOutput({}, "TestOutput")
        assert output.name == "TestOutput"
        assert output.config == {}
        assert output.is_initialized() is False
        assert output.is_connected() is False

    def test_initialization(self):
        """Test output initialization."""
        output = DummyOutput({}, "TestOutput")
        assert output.is_initialized() is False

        success = output.initialize()
        assert success is True
        assert output.initialized_called is True
        assert output.is_initialized() is True
        assert output.is_connected() is True

    def test_initialization_failure(self):
        """Test failed initialization."""
        output = DummyOutput({}, "TestOutput", should_fail=True)
        success = output.initialize()
        assert success is False

    def test_send_state_success(self):
        """Test successful state sending."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        success = output.send_state(touch_state, height_state)
        assert success is True
        assert output.send_called is True
        assert output.send_count == 1
        assert output.stats.packets_sent == 1
        assert output.stats.packets_failed == 0
        assert output.stats.bytes_sent == 38

    def test_send_state_failure(self):
        """Test failed state sending."""
        output = DummyOutput({}, "TestOutput", should_fail=True)
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        success = output.send_state(touch_state, height_state)
        assert success is False
        assert output.stats.packets_sent == 0
        assert output.stats.packets_failed == 1
        assert output.stats.last_error_message == "Dummy error"

    def test_multiple_sends(self):
        """Test multiple send operations."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        for _ in range(10):
            output.send_state(touch_state, height_state)

        assert output.stats.packets_sent == 10
        assert output.stats.bytes_sent == 380  # 10 * 38

    def test_close(self):
        """Test closing output."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()
        assert output.is_connected() is True

        output.close()
        assert output.close_called is True
        assert output.is_connected() is False
        assert output.is_initialized() is False

    def test_get_stats(self):
        """Test getting statistics."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()
        output.send_state(touch_state, height_state)

        stats = output.get_stats()
        assert isinstance(stats, dict)
        assert stats["packets_sent"] == 1
        assert stats["bytes_sent"] == 38
        assert "success_rate" in stats

    def test_reset_stats(self):
        """Test resetting statistics."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        for _ in range(5):
            output.send_state(touch_state, height_state)

        assert output.stats.packets_sent == 5
        output.reset_stats()
        assert output.stats.packets_sent == 0
        assert output.stats.bytes_sent == 0

    def test_get_config(self):
        """Test getting configuration."""
        config = {"key": "value", "number": 42}
        output = DummyOutput(config, "TestOutput")

        returned_config = output.get_config()
        assert returned_config == config
        # Verify it's a copy
        returned_config["new_key"] = "new_value"
        assert "new_key" not in output.config

    def test_get_name(self):
        """Test getting output name."""
        output = DummyOutput({}, "MyTestOutput")
        assert output.get_name() == "MyTestOutput"

    def test_repr(self):
        """Test string representation."""
        output = DummyOutput({}, "TestOutput")
        repr_str = repr(output)
        assert "DummyOutput" in repr_str
        assert "TestOutput" in repr_str
        assert "not initialized" in repr_str

        output.initialize()
        repr_str = repr(output)
        assert "initialized" in repr_str
        assert "connected" in repr_str

    def test_context_manager_success(self):
        """Test using output as context manager."""
        with DummyOutput({}, "TestOutput") as output:
            assert output.is_initialized() is True
            assert output.is_connected() is True
            touch_state = TouchState()
            height_state = HeightState()
            output.send_state(touch_state, height_state)

        # After context exit
        assert output.close_called is True

    def test_context_manager_init_failure(self):
        """Test context manager with initialization failure."""
        with pytest.raises(OutputError):
            with DummyOutput({}, "TestOutput", should_fail=True) as output:
                pass

    def test_stats_update_moving_average(self):
        """Test that moving average updates correctly."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        # Send 100 times to trigger average calculation
        for _ in range(100):
            output.send_state(touch_state, height_state)

        # Average should be calculated
        assert output.stats.average_send_duration_ms > 0

    def test_invalid_config_validation(self):
        """Test configuration validation."""

        class ValidatingOutput(DummyOutput):
            def _validate_config(self, config: Dict[str, Any]) -> list[str]:
                errors = []
                if "required_key" not in config:
                    errors.append("Missing required_key")
                return errors

        # Should raise on invalid config
        with pytest.raises(OutputError):
            ValidatingOutput({}, "TestOutput")

        # Should succeed with valid config
        output = ValidatingOutput({"required_key": "value"}, "TestOutput")
        assert output is not None

    def test_connection_state_changes(self):
        """Test connection state tracking."""
        output = DummyOutput({}, "TestOutput")
        assert output.is_connected() is False

        output.initialize()
        assert output.is_connected() is True

        output._set_connected(False)
        assert output.is_connected() is False

        output._set_connected(True)
        assert output.is_connected() is True

    def test_error_tracking(self):
        """Test error message tracking."""
        output = DummyOutput({}, "TestOutput", should_fail=True)
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        output.send_state(touch_state, height_state)

        assert output.stats.packets_failed == 1
        assert output.stats.last_error_message == "Dummy error"
        assert output.stats.last_error_time > 0

    def test_timestamps(self):
        """Test timestamp tracking."""
        output = DummyOutput({}, "TestOutput")
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        before = time.time()
        output.send_state(touch_state, height_state)
        after = time.time()

        assert before <= output.stats.last_send_time <= after
