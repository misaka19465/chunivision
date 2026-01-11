"""
Unit tests for ConsoleOutput.
"""

import sys
from io import StringIO
from typing import Any, Dict

import pytest

from chunivision.output.console_output import ConsoleOutput
from chunivision.vision.height_estimator import HeightState
from chunivision.vision.touch_detector import TouchState


class TestConsoleOutput:
    """Tests for ConsoleOutput."""

    def test_creation(self):
        """Test creating console output."""
        output = ConsoleOutput({})
        assert output is not None
        assert output.name == "Console"

    def test_creation_with_config(self):
        """Test creating with custom config."""
        config = {
            "update_interval": 0.1,
            "show_stats": False,
            "color_enabled": False,
        }
        output = ConsoleOutput(config, "MyConsole")
        assert output.name == "MyConsole"
        assert output._update_interval == 0.1
        assert output._show_stats is False
        assert output._color_enabled is False

    def test_invalid_update_interval(self):
        """Test invalid update interval."""
        with pytest.raises(Exception):  # OutputError
            ConsoleOutput({"update_interval": -1})

        with pytest.raises(Exception):  # OutputError
            ConsoleOutput({"update_interval": "invalid"})

    def test_initialization(self):
        """Test initialization."""
        output = ConsoleOutput({})
        assert output.is_initialized() is False

        success = output.initialize()
        assert success is True
        assert output.is_initialized() is True
        assert output.is_connected() is True

    def test_send_state(self, monkeypatch):
        """Test sending state."""
        # Capture stdout
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": False})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        success = output.send_state(touch_state, height_state)
        assert success is True

        # Check output was written
        display = captured_output.getvalue()
        assert len(display) > 0
        assert "Touch Zones" in display
        assert "Air Sensors" in display

    def test_send_state_with_touches(self, monkeypatch):
        """Test displaying active touches."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": False})
        output.initialize()

        # Create state with some touches
        import numpy as np

        zones = np.zeros(32, dtype=bool)
        zones[0] = True  # Zone 1
        zones[15] = True  # Zone 16
        zones[31] = True  # Zone 32
        touch_state = TouchState(zones=zones)
        height_state = HeightState()

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        assert "█" in display  # Should have active touch indicators

    def test_send_state_with_heights(self, monkeypatch):
        """Test displaying active height levels."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": False})
        output.initialize()

        # Create state with some height levels active
        import numpy as np

        touch_state = TouchState()
        levels = np.zeros(6, dtype=bool)
        levels[0] = True
        levels[2] = True
        levels[5] = True
        height_state = HeightState(levels=levels)

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        assert "█" in display  # Should have active height indicators

    def test_rate_limiting(self, monkeypatch):
        """Test update rate limiting."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"update_interval": 1.0})  # 1 second interval
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        # First send should work
        output.send_state(touch_state, height_state)
        first_length = len(captured_output.getvalue())

        # Immediate second send should be skipped (rate limited)
        captured_output.truncate(0)
        captured_output.seek(0)
        output.send_state(touch_state, height_state)
        second_length = len(captured_output.getvalue())

        assert first_length > 0
        assert second_length == 0  # Should be rate limited

    def test_stats_display(self, monkeypatch):
        """Test statistics display."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"show_stats": True, "color_enabled": False})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        # Send multiple times to build stats
        for _ in range(5):
            output._last_update = 0  # Force update
            output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        assert "Sent:" in display
        assert "Failed:" in display
        assert "Rate:" in display

    def test_no_stats_display(self, monkeypatch):
        """Test with stats disabled."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"show_stats": False, "color_enabled": False})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        assert "Sent:" not in display

    def test_close(self, monkeypatch):
        """Test closing output."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()
        output.send_state(touch_state, height_state)

        output.close()
        assert output.is_connected() is False
        assert output.is_initialized() is False

    def test_context_manager(self, monkeypatch):
        """Test using as context manager."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        with ConsoleOutput({}) as output:
            touch_state = TouchState()
            height_state = HeightState()
            output.send_state(touch_state, height_state)

            assert output.is_initialized() is True

        # Should be closed after context
        assert output.is_initialized() is False

    def test_build_touch_display(self):
        """Test touch display building."""
        output = ConsoleOutput({"color_enabled": False})

        import numpy as np

        zones = np.zeros(32, dtype=bool)
        zones[0] = True  # Zone 1 (bottom-left)
        zones[1] = True  # Zone 2 (top-left)
        touch_state = TouchState(zones=zones)

        lines = output._build_touch_display(touch_state)
        assert len(lines) == 3  # Title + 2 rows
        assert "█" in "".join(lines)  # Active zones shown

    def test_build_height_display(self):
        """Test height display building."""
        output = ConsoleOutput({"color_enabled": False})

        import numpy as np

        levels = np.zeros(6, dtype=bool)
        levels[0] = True
        levels[5] = True
        height_state = HeightState(levels=levels)

        line = output._build_height_display(height_state)
        assert "Air Sensors" in line
        assert "█" in line  # Active levels shown

    def test_color_enabled(self, monkeypatch):
        """Test with colors enabled."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": True})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        # Should contain ANSI color codes
        assert "\033[" in display

    def test_color_disabled(self, monkeypatch):
        """Test with colors disabled."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": False})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        # Should not contain ANSI color codes
        assert "\033[" not in display or display.count("\033[") < 5

    def test_statistics_tracking(self, monkeypatch):
        """Test that statistics are tracked."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        # Force multiple updates
        for _ in range(3):
            output._last_update = 0
            output.send_state(touch_state, height_state)

        stats = output.get_stats()
        assert stats["packets_sent"] == 3
        assert stats["packets_failed"] == 0
        assert stats["bytes_sent"] > 0

    def test_display_box_formatting(self, monkeypatch):
        """Test that display box is properly formatted."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": False})
        output.initialize()

        touch_state = TouchState()
        height_state = HeightState()

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        assert "╔" in display  # Top border
        assert "╚" in display  # Bottom border
        assert "│" in display  # Side borders

    def test_all_zones_active(self, monkeypatch):
        """Test display with all zones active."""
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        output = ConsoleOutput({"color_enabled": False})
        output.initialize()

        import numpy as np

        touch_state = TouchState(zones=np.ones(32, dtype=bool))
        height_state = HeightState(levels=np.ones(6, dtype=bool))

        output.send_state(touch_state, height_state)

        display = captured_output.getvalue()
        # Should have many active indicators
        assert display.count("█") > 30

    def test_error_handling(self, monkeypatch):
        """Test error handling during send."""
        output = ConsoleOutput({})
        output.initialize()

        # Mock stdout to raise exception
        def failing_write(data):
            raise IOError("Mock write error")

        monkeypatch.setattr(sys.stdout, "write", failing_write)

        touch_state = TouchState()
        height_state = HeightState()

        success = output.send_state(touch_state, height_state)
        assert success is False
        assert output.stats.packets_failed > 0
