"""Hardware tests for CameraManager - requires physical hardware.

These tests require actual Oculus Rift CV1 cameras to be connected.
Run with: pytest -m hardware -v
Skip with: pytest -m "not hardware" -v
"""

import time

import numpy as np
import pytest

try:
    import usb1
except ImportError:
    pytest.skip("usb1 not installed", allow_module_level=True)

from chunivision.config.camera_config import CameraConfig
from chunivision.oculus import list_oculus_cameras
from chunivision.vision.camera_manager import CameraManager


def is_hardware_available() -> bool:
    """Check if Oculus cameras are connected and available.

    Returns:
        bool: True if at least 2 Oculus cameras are detected (for dual camera setup)
    """
    try:
        cameras = list_oculus_cameras()
        return len(cameras) >= 2
    except Exception:
        return False


@pytest.fixture
def hardware_config():
    """Create config using actual connected cameras."""
    cameras = list_oculus_cameras()
    if len(cameras) < 2:
        pytest.skip("Need at least 2 Oculus cameras for CameraManager tests")

    return CameraConfig(
        left_camera_serial=cameras[0]["serial"],
        right_camera_serial=cameras[1]["serial"],
        resolution=(1280, 960),
        fps=60,
        exposure=0.0,
        left_camera_position=np.array([0.0, 0.0, 0.0]),
        right_camera_position=np.array([63.5, 0.0, 0.0]),
        baseline_distance=63.5,
        left_camera_angle=0.0,
        right_camera_angle=0.0,
    )


@pytest.mark.hardware
@pytest.mark.skipif(not is_hardware_available(), reason="hardware needed.")
class TestCameraManagerHardware:
    """Hardware tests for CameraManager with real Oculus cameras."""

    def test_camera_manager_initialization(self, hardware_config):
        """Test CameraManager initialization with real cameras."""
        manager = CameraManager(hardware_config)
        try:
            result = manager.initialize()
            assert result is True, "CameraManager initialization failed"
            assert manager.is_ready() is True
        finally:
            manager.release()

    def test_camera_manager_initialization_fails_invalid_serial(self):
        """Test that initialization fails with invalid serial numbers."""
        config = CameraConfig(
            left_camera_serial="INVALID_LEFT_SERIAL",
            right_camera_serial="INVALID_RIGHT_SERIAL",
            resolution=(1280, 960),
            fps=60,
            exposure=0.0,
            left_camera_position=np.array([0.0, 0.0, 0.0]),
            right_camera_position=np.array([63.5, 0.0, 0.0]),
            baseline_distance=63.5,
            left_camera_angle=0.0,
            right_camera_angle=0.0,
        )

        manager = CameraManager(config)
        try:
            result = manager.initialize()
            assert result is False, "Should fail with invalid serials"
            assert manager.is_ready() is False
        finally:
            manager.release()

    def test_camera_manager_get_camera_info(self, hardware_config):
        """Test getting camera information from real devices."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()
            info = manager.get_camera_info()

            assert "left_camera" in info
            assert "right_camera" in info
            assert info["left_camera"]["resolution"] == (1280, 960)
            assert info["right_camera"]["resolution"] == (1280, 960)
            assert info["left_camera"]["serial"] == hardware_config.left_camera_serial
            assert info["right_camera"]["serial"] == hardware_config.right_camera_serial
        finally:
            manager.release()

    def test_camera_manager_streaming(self, hardware_config):
        """Test streaming from dual cameras."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()
            manager.start_streaming()

            # Wait for frames to arrive
            time.sleep(0.5)

            # Try to get a frame pair
            frames = manager.get_frame_pair()
            assert frames is not None, "No frames received"

            left_frame, right_frame, timestamp = frames
            assert left_frame is not None
            assert right_frame is not None
            assert timestamp > 0

            # Verify frame sizes
            expected_size = 1280 * 960
            assert len(left_frame) == expected_size
            assert len(right_frame) == expected_size

            manager.stop_streaming()
        finally:
            manager.release()

    def test_camera_manager_synchronized_frames(self, hardware_config):
        """Test that frames from both cameras are synchronized."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()
            manager.start_streaming()

            # Collect multiple frame pairs
            time.sleep(0.5)

            frame_pairs = []
            for _ in range(5):
                frames = manager.get_frame_pair()
                if frames:
                    frame_pairs.append(frames)
                time.sleep(0.016)  # ~60 FPS

            assert len(frame_pairs) > 0, "No synchronized frames received"

            # Verify all pairs have valid data
            for left, right, ts in frame_pairs:
                assert left is not None
                assert right is not None
                assert ts > 0

            manager.stop_streaming()
        finally:
            manager.release()

    def test_camera_manager_fps_measurement(self, hardware_config):
        """Test FPS measurement with real cameras."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()
            manager.start_streaming()

            # Let it run for a bit to accumulate FPS data
            time.sleep(1.0)

            left_fps, right_fps = manager.get_fps()

            # FPS should be positive and reasonable (allow for some variance)
            assert left_fps > 0, "Left camera FPS should be positive"
            assert right_fps > 0, "Right camera FPS should be positive"

            # FPS should be close to configured value (within 50%)
            # Real hardware may not achieve exact target FPS
            assert 30 < left_fps < 90, f"Left FPS {left_fps} outside expected range"
            assert 30 < right_fps < 90, f"Right FPS {right_fps} outside expected range"

            manager.stop_streaming()
        finally:
            manager.release()

    def test_camera_manager_context_manager(self, hardware_config):
        """Test CameraManager as context manager with real hardware."""
        with CameraManager(hardware_config) as manager:
            manager.initialize()
            assert manager.is_ready()

            manager.start_streaming()
            time.sleep(0.2)

            frames = manager.get_frame_pair()
            assert frames is not None

        # Manager should be released after context exit

    def test_camera_manager_release_while_streaming(self, hardware_config):
        """Test releasing manager while streaming."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()
            manager.start_streaming()
            time.sleep(0.2)
        finally:
            # Should handle cleanup even if streaming
            manager.release()

        assert manager.is_ready() is False

    def test_camera_manager_multiple_start_stop_cycles(self, hardware_config):
        """Test multiple streaming start/stop cycles."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()

            for cycle in range(3):
                manager.start_streaming()
                time.sleep(0.3)

                frames = manager.get_frame_pair()
                assert frames is not None, f"Cycle {cycle}: No frames received"

                manager.stop_streaming()
                time.sleep(0.1)
        finally:
            manager.release()

    def test_camera_manager_reconnection(self, hardware_config):
        """Test releasing and re-initializing cameras."""
        # First session
        manager1 = CameraManager(hardware_config)
        manager1.initialize()
        assert manager1.is_ready()
        manager1.release()
        assert not manager1.is_ready()

        # Brief pause to ensure resources are released
        time.sleep(0.2)

        # Second session with same config
        manager2 = CameraManager(hardware_config)
        manager2.initialize()
        assert manager2.is_ready()
        manager2.release()

    def test_camera_manager_frame_data_integrity(self, hardware_config):
        """Test that frame data from real cameras is valid."""
        manager = CameraManager(hardware_config)
        try:
            manager.initialize()
            manager.start_streaming()
            time.sleep(0.5)

            frames = manager.get_frame_pair()
            assert frames is not None

            left_frame, right_frame, _ = frames

            # Convert to numpy array for analysis
            left_array = np.frombuffer(left_frame, dtype=np.uint8)
            right_array = np.frombuffer(right_frame, dtype=np.uint8)

            # Check that frames contain actual data (not all zeros or all same value)
            assert left_array.std() > 0, "Left frame appears to be uniform/invalid"
            assert right_array.std() > 0, "Right frame appears to be uniform/invalid"

            # Values should be in valid range for 8-bit grayscale
            assert left_array.min() >= 0
            assert left_array.max() <= 255
            assert right_array.min() >= 0
            assert right_array.max() <= 255

            manager.stop_streaming()
        finally:
            manager.release()
