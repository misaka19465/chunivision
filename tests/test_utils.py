"""
Unit tests for utility modules.
"""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path

from chunivision.utils import (
    Logger,
    Point2D,
    Point3D,
    point_in_polygon,
    distance_2d,
    distance_3d,
    StateManager,
    TouchState,
    HeightState,
    PerformanceMonitor,
    PerformanceStats
)


class TestLogger:
    """Tests for Logger class."""

    def test_setup_console_only(self):
        """Test logger setup with console output only."""
        Logger.setup(level="INFO")
        logger = Logger.get_logger("test")
        assert logger is not None
        assert Logger.get_level() == "INFO"

    def test_setup_with_file(self):
        """Test logger setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            Logger.setup(level="DEBUG", log_file=str(log_file))

            logger = Logger.get_logger("test_file")
            logger.info("Test message")

            assert log_file.exists()
            assert Logger.get_level() == "DEBUG"

    def test_set_level(self):
        """Test changing log level dynamically."""
        Logger.setup(level="INFO")
        assert Logger.get_level() == "INFO"

        Logger.set_level("DEBUG")
        assert Logger.get_level() == "DEBUG"

    def test_get_logger(self):
        """Test getting logger instances."""
        Logger.setup()
        logger1 = Logger.get_logger("module1")
        logger2 = Logger.get_logger("module2")

        assert logger1 is not None
        assert logger2 is not None
        assert logger1.name == "module1"
        assert logger2.name == "module2"


class TestGeometry:
    """Tests for geometry utilities."""

    def test_point2d_creation(self):
        """Test Point2D creation and basic operations."""
        p = Point2D(3.0, 4.0)
        assert p.x == 3.0
        assert p.y == 4.0

        arr = p.to_array()
        np.testing.assert_array_equal(arr, np.array([3.0, 4.0]))

    def test_point2d_distance(self):
        """Test Point2D distance calculation."""
        p1 = Point2D(0.0, 0.0)
        p2 = Point2D(3.0, 4.0)

        dist = p1.distance_to(p2)
        assert dist == 5.0

    def test_point2d_operations(self):
        """Test Point2D arithmetic operations."""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(3.0, 4.0)

        p_add = p1 + p2
        assert p_add.x == 4.0 and p_add.y == 6.0

        p_sub = p2 - p1
        assert p_sub.x == 2.0 and p_sub.y == 2.0

        p_mul = p1 * 2.0
        assert p_mul.x == 2.0 and p_mul.y == 4.0

    def test_point3d_creation(self):
        """Test Point3D creation and basic operations."""
        p = Point3D(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

        arr = p.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_point3d_to_2d(self):
        """Test Point3D projection to 2D."""
        p3d = Point3D(1.0, 2.0, 3.0)
        p2d = p3d.to_2d()

        assert isinstance(p2d, Point2D)
        assert p2d.x == 1.0
        assert p2d.y == 2.0

    def test_point_in_polygon(self):
        """Test point-in-polygon test."""
        # Square polygon
        square = [
            Point2D(0.0, 0.0),
            Point2D(2.0, 0.0),
            Point2D(2.0, 2.0),
            Point2D(0.0, 2.0)
        ]

        # Point inside
        assert point_in_polygon(Point2D(1.0, 1.0), square) == True

        # Point outside
        assert point_in_polygon(Point2D(3.0, 3.0), square) == False

        # Point on edge (implementation dependent)
        result = point_in_polygon(Point2D(1.0, 0.0), square)
        assert isinstance(result, bool)

    def test_distance_2d(self):
        """Test 2D distance calculation."""
        p1 = Point2D(0.0, 0.0)
        p2 = Point2D(3.0, 4.0)

        dist = distance_2d(p1, p2)
        assert dist == 5.0

        # Test with numpy arrays
        arr1 = np.array([0.0, 0.0])
        arr2 = np.array([3.0, 4.0])
        dist2 = distance_2d(arr1, arr2)
        assert dist2 == 5.0

    def test_distance_3d(self):
        """Test 3D distance calculation."""
        p1 = Point3D(0.0, 0.0, 0.0)
        p2 = Point3D(2.0, 2.0, 1.0)

        dist = distance_3d(p1, p2)
        assert abs(dist - 3.0) < 1e-6


class TestStateManager:
    """Tests for StateManager class."""

    def test_state_manager_creation(self):
        """Test StateManager initialization."""
        sm = StateManager(debounce_frames=2, history_size=100)
        assert sm is not None

        touch, height = sm.get_current_state()
        assert np.all(touch.zones == False)
        assert np.all(height.levels == 0)

    def test_state_update(self):
        """Test state update and change detection."""
        sm = StateManager(debounce_frames=1)

        # Create initial state
        touch = TouchState(zones=np.zeros(32, dtype=bool))
        height = HeightState(levels=np.zeros(6, dtype=int))

        # First update (no change)
        changed = sm.update_state(touch, height)
        assert changed == False

        # Second update with change
        touch.zones[0] = True
        changed = sm.update_state(touch, height)
        assert changed == True

        # Check changes
        touch_changes, height_changes = sm.get_changes()
        assert touch_changes[0] == 1  # Pressed

    def test_debouncing(self):
        """Test debouncing functionality."""
        sm = StateManager(debounce_frames=3)

        # Send noisy input
        touch1 = TouchState(zones=np.zeros(32, dtype=bool))
        touch2 = TouchState(zones=np.zeros(32, dtype=bool))
        touch2.zones[0] = True

        height = HeightState(levels=np.zeros(6, dtype=int))

        # First frame
        sm.update_state(touch1, height)

        # Second frame (noise)
        sm.update_state(touch2, height)

        # Third frame
        sm.update_state(touch1, height)

        # Should not register as pressed due to debouncing
        curr_touch, _ = sm.get_current_state()
        assert curr_touch.zones[0] == False

    def test_get_pressed_zones(self):
        """Test getting pressed zone indices."""
        sm = StateManager(debounce_frames=1)

        touch = TouchState(zones=np.zeros(32, dtype=bool))
        touch.zones[0] = True  # Zone 1
        touch.zones[5] = True  # Zone 6
        height = HeightState(levels=np.zeros(6, dtype=int))

        sm.update_state(touch, height)
        sm.update_state(touch, height)  # Confirm with debounce

        pressed = sm.get_pressed_zones()
        np.testing.assert_array_equal(pressed, np.array([1, 6]))

    def test_history(self):
        """Test state history tracking."""
        sm = StateManager(debounce_frames=1, history_size=10)

        touch = TouchState(zones=np.zeros(32, dtype=bool))
        height = HeightState(levels=np.zeros(6, dtype=int))

        # Update multiple times
        for i in range(5):
            touch.zones[i] = True
            sm.update_state(touch, height)

        touch_hist, height_hist = sm.get_history()
        assert len(touch_hist) <= 5

    def test_reset(self):
        """Test state manager reset."""
        sm = StateManager(debounce_frames=1)

        touch = TouchState(zones=np.ones(32, dtype=bool))
        height = HeightState(levels=np.ones(6, dtype=int))

        sm.update_state(touch, height)
        sm.reset()

        curr_touch, curr_height = sm.get_current_state()
        assert np.all(curr_touch.zones == False)
        assert np.all(curr_height.levels == 0)


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_performance_monitor_creation(self):
        """Test PerformanceMonitor initialization."""
        pm = PerformanceMonitor(fps_window=60, latency_window=100)
        assert pm is not None
        assert pm.get_fps() == 0.0

    def test_fps_tracking(self):
        """Test FPS tracking."""
        pm = PerformanceMonitor(fps_window=10)

        # Simulate frames at 30 FPS (33.33ms per frame)
        for _ in range(10):
            pm.record_frame()
            time.sleep(0.033)

        fps = pm.get_fps()
        assert 25 < fps < 35  # Allow some tolerance

    def test_latency_recording(self):
        """Test latency recording."""
        pm = PerformanceMonitor()

        pm.record_latency("stage1", 10.5)
        pm.record_latency("stage1", 11.5)
        pm.record_latency("stage2", 5.0)

        latencies = pm.get_stage_latencies()
        assert "stage1" in latencies
        assert "stage2" in latencies
        assert abs(latencies["stage1"] - 11.0) < 0.5

    def test_dropped_frames(self):
        """Test dropped frame counting."""
        pm = PerformanceMonitor()

        pm.record_dropped_frame()
        pm.record_dropped_frame()

        stats = pm.get_stats()
        assert stats.dropped_frames == 2

    def test_get_stats(self):
        """Test getting performance statistics."""
        pm = PerformanceMonitor()

        pm.record_frame()
        time.sleep(0.01)
        pm.record_frame()
        pm.record_latency("test_stage", 5.0)

        stats = pm.get_stats()
        assert isinstance(stats, PerformanceStats)
        assert stats.fps >= 0
        assert "test_stage" in stats.stage_latencies

    def test_reset(self):
        """Test performance monitor reset."""
        pm = PerformanceMonitor()

        pm.record_frame()
        pm.record_latency("stage1", 10.0)
        pm.record_dropped_frame()

        pm.reset()

        stats = pm.get_stats()
        assert stats.dropped_frames == 0
        assert len(stats.stage_latencies) == 0

    def test_export_to_file(self):
        """Test exporting stats to file."""
        pm = PerformanceMonitor()

        pm.record_frame()
        time.sleep(0.01)
        pm.record_frame()
        pm.record_latency("test_stage", 5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "perf_stats.txt"
            pm.export_to_file(str(filepath))

            assert filepath.exists()
            content = filepath.read_text()
            assert "ChunIVision Performance Report" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
