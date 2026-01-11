"""
Unit tests for OutputManager.
"""

import time
from typing import Any, Dict

import pytest

from chunivision.output.base_output import BaseOutput
from chunivision.output.output_manager import OutputManager
from chunivision.vision.height_estimator import HeightState
from chunivision.vision.touch_detector import TouchState


# Test output implementations
class MockOutput(BaseOutput):
    """Mock output for testing."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str = "MockOutput",
        init_success: bool = True,
        send_success: bool = True,
    ):
        self.init_success = init_success
        self.send_success = send_success
        self.initialize_count = 0
        self.send_count = 0
        self.close_count = 0
        super().__init__(config, name)

    def initialize(self) -> bool:
        self.initialize_count += 1
        if self.init_success:
            self._set_initialized(True)
            self._set_connected(True)
            return True
        return False

    def send_state(self, touch_state: TouchState, height_state: HeightState) -> bool:
        self.send_count += 1
        if self.send_success:
            self._update_stats_success(38, 1.0)
            return True
        else:
            self._update_stats_failure("Mock send failure")
            return False

    def close(self) -> None:
        self.close_count += 1
        self._set_connected(False)
        self._set_initialized(False)


class TestOutputManager:
    """Tests for OutputManager."""

    def test_creation(self):
        """Test creating output manager."""
        manager = OutputManager()
        assert manager is not None
        assert manager.count_outputs() == 0
        assert manager.is_initialized() is False

    def test_add_output(self):
        """Test adding output to manager."""
        manager = OutputManager()
        output = MockOutput({}, "test1")

        manager.add_output("test1", output)
        assert manager.count_outputs() == 1
        assert manager.has_output("test1")
        assert "test1" in manager.get_active_outputs()

    def test_add_multiple_outputs(self):
        """Test adding multiple outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")
        output3 = MockOutput({}, "output3")

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.add_output("out3", output3)

        assert manager.count_outputs() == 3
        assert set(manager.get_active_outputs()) == {"out1", "out2", "out3"}

    def test_add_duplicate_name(self):
        """Test adding output with duplicate name."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager.add_output("test", output1)

        with pytest.raises(ValueError):
            manager.add_output("test", output2)

    def test_add_non_output_object(self):
        """Test adding non-BaseOutput object."""
        manager = OutputManager()

        with pytest.raises(TypeError):
            manager.add_output("test", "not an output")  # type: ignore

    def test_remove_output(self):
        """Test removing output."""
        manager = OutputManager()
        output = MockOutput({}, "test")
        manager.add_output("test", output)

        assert manager.has_output("test")
        result = manager.remove_output("test")

        assert result is True
        assert not manager.has_output("test")
        assert manager.count_outputs() == 0
        assert output.close_count == 1

    def test_remove_nonexistent_output(self):
        """Test removing output that doesn't exist."""
        manager = OutputManager()
        result = manager.remove_output("nonexistent")
        assert result is False

    def test_get_output(self):
        """Test getting output by name."""
        manager = OutputManager()
        output = MockOutput({}, "test")
        manager.add_output("test", output)

        retrieved = manager.get_output("test")
        assert retrieved is output

        none_output = manager.get_output("nonexistent")
        assert none_output is None

    def test_has_output(self):
        """Test checking if output exists."""
        manager = OutputManager()
        output = MockOutput({}, "test")

        assert not manager.has_output("test")
        manager.add_output("test", output)
        assert manager.has_output("test")

    def test_get_active_outputs(self):
        """Test getting list of active outputs."""
        manager = OutputManager()
        manager.add_output("out1", MockOutput({}, "output1"))
        manager.add_output("out2", MockOutput({}, "output2"))

        active = manager.get_active_outputs()
        assert len(active) == 2
        assert set(active) == {"out1", "out2"}

    def test_get_connected_outputs(self):
        """Test getting connected outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1", init_success=True)
        output2 = MockOutput({}, "output2", init_success=False)

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        connected = manager.get_connected_outputs()
        assert connected == ["out1"]  # Only out1 initialized successfully

    def test_initialize_all(self):
        """Test initializing all outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1", init_success=True)
        output2 = MockOutput({}, "output2", init_success=True)

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)

        results = manager.initialize_all()

        assert results == {"out1": True, "out2": True}
        assert manager.is_initialized() is True
        assert output1.initialize_count == 1
        assert output2.initialize_count == 1

    def test_initialize_all_with_failures(self):
        """Test initializing when some outputs fail."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1", init_success=True)
        output2 = MockOutput({}, "output2", init_success=False)

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)

        results = manager.initialize_all()

        assert results == {"out1": True, "out2": False}
        assert manager.is_initialized() is True

    def test_close_all(self):
        """Test closing all outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        manager.close_all()

        assert output1.close_count == 1
        assert output2.close_count == 1

    def test_send_state_to_all(self):
        """Test broadcasting state to all outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()

        results = manager.send_state(touch_state, height_state)

        assert results == {"out1": True, "out2": True}
        assert output1.send_count == 1
        assert output2.send_count == 1

    def test_send_state_with_failures(self):
        """Test broadcasting when some outputs fail."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1", send_success=True)
        output2 = MockOutput({}, "output2", send_success=False)

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()

        results = manager.send_state(touch_state, height_state)

        assert results == {"out1": True, "out2": False}
        assert output1.send_count == 1
        assert output2.send_count == 1

    def test_send_state_no_outputs(self):
        """Test broadcasting with no outputs registered."""
        manager = OutputManager()

        touch_state = TouchState()
        height_state = HeightState()

        results = manager.send_state(touch_state, height_state)
        assert results == {}

    def test_send_state_to_specific_output(self):
        """Test sending to a specific output."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()

        result = manager.send_state_to("out1", touch_state, height_state)

        assert result is True
        assert output1.send_count == 1
        assert output2.send_count == 0  # out2 should not have received

    def test_send_state_to_nonexistent_output(self):
        """Test sending to output that doesn't exist."""
        manager = OutputManager()

        touch_state = TouchState()
        height_state = HeightState()

        with pytest.raises(KeyError):
            manager.send_state_to("nonexistent", touch_state, height_state)

    def test_get_all_stats(self):
        """Test getting statistics from all outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()
        manager.send_state(touch_state, height_state)

        all_stats = manager.get_all_stats()

        assert "out1" in all_stats
        assert "out2" in all_stats
        assert all_stats["out1"]["packets_sent"] == 1
        assert all_stats["out2"]["packets_sent"] == 1

    def test_get_stats_for_specific_output(self):
        """Test getting statistics for specific output."""
        manager = OutputManager()
        output = MockOutput({}, "output")

        manager.add_output("test", output)
        manager.initialize_all()

        stats = manager.get_stats("test")
        assert stats is not None
        assert "packets_sent" in stats

        none_stats = manager.get_stats("nonexistent")
        assert none_stats is None

    def test_reset_all_stats(self):
        """Test resetting statistics for all outputs."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()
        manager.send_state(touch_state, height_state)

        assert output1.stats.packets_sent == 1
        assert output2.stats.packets_sent == 1

        manager.reset_all_stats()

        assert output1.stats.packets_sent == 0
        assert output2.stats.packets_sent == 0

    def test_get_summary(self):
        """Test getting manager summary."""
        manager = OutputManager()
        output1 = MockOutput({}, "output1", init_success=True)
        output2 = MockOutput({}, "output2", init_success=False)

        manager.add_output("out1", output1)
        manager.add_output("out2", output2)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()
        manager.send_state(touch_state, height_state)
        manager.send_state(touch_state, height_state)

        summary = manager.get_summary()

        assert summary["total_outputs"] == 2
        assert summary["connected_outputs"] == 1  # Only out1 initialized
        assert summary["initialized"] is True
        assert summary["total_broadcasts"] == 2
        assert "uptime_seconds" in summary
        assert "outputs" in summary
        assert "out1" in summary["outputs"]
        assert "out2" in summary["outputs"]

    def test_context_manager(self):
        """Test using manager as context manager."""
        output1 = MockOutput({}, "output1")
        output2 = MockOutput({}, "output2")

        manager = OutputManager()
        manager.add_output("out1", output1)
        manager.add_output("out2", output2)

        with manager:
            # __enter__ calls initialize_all()
            assert output1.initialize_count == 1
            assert output2.initialize_count == 1

        # After context exit, outputs should be closed
        assert output1.close_count == 1
        assert output2.close_count == 1

    def test_len(self):
        """Test __len__ method."""
        manager = OutputManager()
        assert len(manager) == 0

        manager.add_output("out1", MockOutput({}, "output1"))
        assert len(manager) == 1

        manager.add_output("out2", MockOutput({}, "output2"))
        assert len(manager) == 2

    def test_contains(self):
        """Test __contains__ method."""
        manager = OutputManager()
        output = MockOutput({}, "output")

        assert "test" not in manager
        manager.add_output("test", output)
        assert "test" in manager

    def test_iter(self):
        """Test iterating over output names."""
        manager = OutputManager()
        manager.add_output("out1", MockOutput({}, "output1"))
        manager.add_output("out2", MockOutput({}, "output2"))
        manager.add_output("out3", MockOutput({}, "output3"))

        names = list(manager)
        assert len(names) == 3
        assert set(names) == {"out1", "out2", "out3"}

    def test_repr(self):
        """Test string representation."""
        manager = OutputManager()
        manager.add_output("out1", MockOutput({}, "output1", init_success=True))
        manager.add_output("out2", MockOutput({}, "output2", init_success=False))
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()
        manager.send_state(touch_state, height_state)

        repr_str = repr(manager)
        assert "OutputManager" in repr_str
        assert "1/2 connected" in repr_str
        assert "1 broadcasts" in repr_str

    def test_error_isolation(self):
        """Test that errors in one output don't affect others."""

        class FailingOutput(MockOutput):
            def send_state(self, touch_state, height_state):
                raise RuntimeError("Intentional failure")

        manager = OutputManager()
        failing = FailingOutput({}, "failing")
        working = MockOutput({}, "working")

        manager.add_output("failing", failing)
        manager.add_output("working", working)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()

        # Should not raise exception, just return failure for one output
        results = manager.send_state(touch_state, height_state)

        assert results["failing"] is False
        assert results["working"] is True
        assert working.send_count == 1

    def test_multiple_broadcasts(self):
        """Test multiple broadcast operations."""
        manager = OutputManager()
        output = MockOutput({}, "output")

        manager.add_output("test", output)
        manager.initialize_all()

        touch_state = TouchState()
        height_state = HeightState()

        for i in range(10):
            manager.send_state(touch_state, height_state)

        assert output.send_count == 10
        summary = manager.get_summary()
        assert summary["total_broadcasts"] == 10

    def test_count_outputs(self):
        """Test counting outputs."""
        manager = OutputManager()
        assert manager.count_outputs() == 0

        manager.add_output("out1", MockOutput({}, "output1"))
        assert manager.count_outputs() == 1

        manager.add_output("out2", MockOutput({}, "output2"))
        assert manager.count_outputs() == 2

        manager.remove_output("out1")
        assert manager.count_outputs() == 1
