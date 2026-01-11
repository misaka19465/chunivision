"""
Tests for CameraManager module.
"""

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from chunivision.config.camera_config import CameraConfig
from chunivision.vision.camera_manager import CameraInitError, CameraManager


class MockOculusCamera:
    """Mock Oculus camera for testing."""

    def __init__(
        self, serial_number="MOCK_SERIAL", fail_init=False, fail_streaming=False
    ):
        """Initialize mock camera.

        Args:
            serial_number: Camera serial number
            fail_init: If True, raise error during initialization
            fail_streaming: If True, raise error during streaming
        """
        if fail_init:
            from chunivision.oculus.exceptions import DeviceInitializationError

            raise DeviceInitializationError("Mock initialization failure")

        self.serial_number = serial_number
        self.fail_streaming = fail_streaming
        self.handle = MagicMock()
        self.streaming = False
        self.streaming_callback = None

    def get_frame_size(self):
        """Return mock frame size."""
        return (1280, 960)

    def get_frame_width(self):
        """Return mock frame width."""
        return 1280

    def get_frame_height(self):
        """Return mock frame height."""
        return 960

    def start_streaming(self, callback):
        """Start streaming with callback.

        Args:
            callback: Callback function for frames
        """
        if self.fail_streaming:
            from chunivision.oculus.exceptions import StreamingError

            raise StreamingError("Mock streaming failure")

        self.streaming = True
        self.streaming_callback = callback

        # Simulate streaming in background thread
        def stream():
            frame_count = 0
            while self.streaming and frame_count < 10:
                # Create mock frame (8-bit grayscale)
                frame_data = np.random.randint(
                    0, 255, (960, 1280), dtype=np.uint8
                ).tobytes()
                metadata = {"frame_id": frame_count, "presentation_time": time.time()}
                if self.streaming_callback:
                    self.streaming_callback(frame_data, metadata)
                frame_count += 1
                time.sleep(0.016)  # ~60 FPS

        self._stream_thread = threading.Thread(target=stream, daemon=True)
        self._stream_thread.start()

    def stop_streaming(self):
        """Stop streaming."""
        self.streaming = False
        if hasattr(self, "_stream_thread"):
            self._stream_thread.join(timeout=1.0)

    def close(self):
        """Close camera."""
        if self.streaming:
            self.stop_streaming()
        self.handle = None


@pytest.fixture
def valid_config():
    """Create valid camera configuration."""
    return CameraConfig(
        left_camera_serial="MOCK_LEFT_SERIAL",
        right_camera_serial="MOCK_RIGHT_SERIAL",
        resolution=(640, 480),
        fps=60,
        exposure=-1,
    )


@pytest.fixture
def invalid_config():
    """Create invalid camera configuration."""
    # Create config without validation (for testing invalid configs)
    config = CameraConfig.__new__(CameraConfig)
    config.left_camera_serial = "SAME_SERIAL"
    config.right_camera_serial = "SAME_SERIAL"  # Same as left - invalid
    config.resolution = (640, 480)
    config.fps = 60
    config.exposure = -1
    config.left_camera_position = np.array([-10.0, 15.0, 30.0])
    config.right_camera_position = np.array([10.0, 15.0, 30.0])
    config.baseline_distance = 20.0
    config.left_camera_angle = 45.0
    config.right_camera_angle = 45.0
    return config
    return config


class TestCameraManagerInit:
    """Tests for CameraManager initialization."""

    def test_init_with_valid_config(self, valid_config):
        """Test initialization with valid configuration."""
        manager = CameraManager(valid_config)

        assert manager.config == valid_config
        assert manager.left_camera is None
        assert manager.right_camera is None
        assert not manager._streaming

    def test_init_with_invalid_config(self, invalid_config):
        """Test initialization with invalid configuration."""
        with pytest.raises(CameraInitError) as exc_info:
            CameraManager(invalid_config)

        assert "Invalid camera configuration" in str(exc_info.value)


class TestCameraManagerInitialize:
    """Tests for camera initialization."""

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_initialize_success(self, mock_camera_class, valid_config):
        """Test successful camera initialization."""
        # Setup mocks
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        result = manager.initialize()

        assert result is True
        assert manager.left_camera is not None
        assert manager.right_camera is not None
        assert manager.is_ready()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_initialize_left_camera_fails(self, mock_camera_class, valid_config):
        """Test initialization when left camera fails."""

        # Setup mocks - left camera fails
        def camera_factory(device_index, **kwargs):
            if device_index == 0:
                return MockOculusCamera(device_index, fail_init=True)
            return MockOculusCamera(device_index)

        mock_camera_class.side_effect = camera_factory

        manager = CameraManager(valid_config)
        with pytest.raises(CameraInitError) as exc_info:
            manager.initialize()

        assert "Left camera initialization failed" in str(exc_info.value)
        assert manager.left_camera is None
        assert manager.right_camera is None

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_initialize_right_camera_fails(self, mock_camera_class, valid_config):
        """Test initialization when right camera fails."""

        # Setup mocks - right camera fails
        def camera_factory(device_index, **kwargs):
            if device_index == 1:
                return MockOculusCamera(device_index, fail_init=True)
            return MockOculusCamera(device_index)

        mock_camera_class.side_effect = camera_factory

        manager = CameraManager(valid_config)
        with pytest.raises(CameraInitError) as exc_info:
            manager.initialize()

        assert "Right camera initialization failed" in str(exc_info.value)
        # Left camera should be cleaned up
        assert manager.left_camera is None
        assert manager.right_camera is None


class TestCameraManagerStreaming:
    """Tests for camera streaming."""

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_start_streaming_success(self, mock_camera_class, valid_config):
        """Test starting streaming successfully."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()
        manager.start_streaming()

        assert manager._streaming is True

        # Clean up
        manager.stop_streaming()
        manager.release()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_start_streaming_not_initialized(self, mock_camera_class, valid_config):
        """Test starting streaming without initialization."""
        manager = CameraManager(valid_config)

        with pytest.raises(CameraInitError):
            manager.start_streaming()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_stop_streaming(self, mock_camera_class, valid_config):
        """Test stopping streaming."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()
        manager.start_streaming()

        assert manager._streaming is True

        manager.stop_streaming()

        assert manager._streaming is False

        # Clean up
        manager.release()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_stop_streaming_when_not_streaming(self, mock_camera_class, valid_config):
        """Test stopping streaming when not streaming."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()

        # Should not raise error
        manager.stop_streaming()

        assert manager._streaming is False

        # Clean up
        manager.release()


class TestCameraManagerFrameCapture:
    """Tests for frame capture."""

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_get_frame_pair_success(self, mock_camera_class, valid_config):
        """Test getting synchronized frame pair."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()
        manager.start_streaming()

        # Wait for frames to arrive
        time.sleep(0.1)

        result = manager.get_frame_pair()

        assert result is not None
        left_frame, right_frame, timestamp = result

        assert isinstance(left_frame, np.ndarray)
        assert isinstance(right_frame, np.ndarray)
        assert left_frame.shape == (960, 1280)
        assert right_frame.shape == (960, 1280)
        assert isinstance(timestamp, float)
        assert timestamp > 0

        # Clean up
        manager.stop_streaming()
        manager.release()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_get_frame_pair_not_streaming(self, mock_camera_class, valid_config):
        """Test getting frame pair when not streaming."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()

        result = manager.get_frame_pair()

        assert result is None

        # Clean up
        manager.release()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_get_frame_pair_no_frames_yet(self, mock_camera_class, valid_config):
        """Test getting frame pair before frames arrive."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()
        manager.start_streaming()

        # Immediately try to get frame (before callback fires)
        result = manager.get_frame_pair()

        # May be None if no frames received yet
        if result is not None:
            assert len(result) == 3

        # Clean up
        manager.stop_streaming()
        manager.release()


class TestCameraManagerInfo:
    """Tests for camera information."""

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_get_camera_info(self, mock_camera_class, valid_config):
        """Test getting camera information."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()

        info = manager.get_camera_info()

        assert info["left_camera_ready"] is True
        assert info["right_camera_ready"] is True
        assert info["streaming"] is False
        assert "left_camera" in info
        assert "right_camera" in info
        assert info["left_camera"]["index"] == 0
        assert info["right_camera"]["index"] == 1
        assert info["left_camera"]["resolution"] == (1280, 960)
        assert info["right_camera"]["resolution"] == (1280, 960)

        # Clean up
        manager.release()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_is_ready(self, mock_camera_class, valid_config):
        """Test is_ready method."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)

        assert not manager.is_ready()

        manager.initialize()

        assert manager.is_ready()

        manager.release()

        assert not manager.is_ready()

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_get_fps(self, mock_camera_class, valid_config):
        """Test FPS tracking."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()
        manager.start_streaming()

        # Wait for some frames
        time.sleep(0.2)

        # Get frames to trigger FPS calculation
        for _ in range(5):
            manager.get_frame_pair()
            time.sleep(0.016)

        fps = manager.get_fps()
        assert fps > 0  # Should have some FPS value

        # Clean up
        manager.stop_streaming()
        manager.release()


class TestCameraManagerResourceManagement:
    """Tests for resource management."""

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_release(self, mock_camera_class, valid_config):
        """Test releasing camera resources."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()

        assert manager.left_camera is not None
        assert manager.right_camera is not None

        manager.release()

        assert manager.left_camera is None
        assert manager.right_camera is None

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_release_while_streaming(self, mock_camera_class, valid_config):
        """Test releasing resources while streaming."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        manager = CameraManager(valid_config)
        manager.initialize()
        manager.start_streaming()

        assert manager._streaming is True

        manager.release()

        assert manager._streaming is False
        assert manager.left_camera is None
        assert manager.right_camera is None

    @patch("chunivision.vision.camera_manager.OculusRiftCV1Camera")
    def test_context_manager(self, mock_camera_class, valid_config):
        """Test context manager protocol."""
        mock_camera_class.side_effect = lambda device_index, **kwargs: MockOculusCamera(
            device_index
        )

        with CameraManager(valid_config) as manager:
            assert manager.is_ready()
            assert manager.left_camera is not None
            assert manager.right_camera is not None

        # Resources should be released after exiting context
        assert manager.left_camera is None
        assert manager.right_camera is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
