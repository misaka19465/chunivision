"""Hardware tests for Oculus cameras - requires physical hardware.

These tests require actual Oculus Rift CV1 cameras to be connected.
Run with: pytest -m hardware -v
Skip with: pytest -m "not hardware" -v
"""

import time

import pytest

try:
    import usb1
except ImportError:
    pytest.skip("usb1 not installed", allow_module_level=True)

from chunivision.oculus import OculusRiftCV1Camera, list_oculus_cameras
from chunivision.oculus.exceptions import DeviceNotFoundError


def is_hardware_available() -> bool:
    """Check if Oculus cameras are connected and available.

    Returns:
        bool: True if at least one Oculus camera is detected
    """
    try:
        cameras = list_oculus_cameras()
        return len(cameras) > 0
    except Exception:
        return False


@pytest.mark.hardware
@pytest.mark.skipif(not is_hardware_available(), reason="hardware needed.")
class TestOculusCameraHardware:
    """Hardware tests for real Oculus Rift CV1 cameras."""

    def test_list_oculus_cameras(self):
        """Test listing connected Oculus cameras."""
        cameras = list_oculus_cameras()
        assert isinstance(cameras, list)
        # At least log what we found
        print(f"\nFound {len(cameras)} Oculus camera(s)")
        for cam_info in cameras:
            print(
                f"  - Serial: {cam_info['serial']}, Bus: {cam_info['bus']}, "
                f"Address: {cam_info['address']}"
            )

    def test_camera_detection(self):
        """Test that at least one Oculus camera can be detected."""
        cameras = list_oculus_cameras()
        assert (
            len(cameras) > 0
        ), "No Oculus cameras detected. Connect at least one camera."

    def test_dual_camera_detection(self):
        """Test that dual cameras are available (required for stereo vision)."""
        cameras = list_oculus_cameras()
        assert (
            len(cameras) >= 2
        ), f"Only {len(cameras)} camera(s) detected. Dual cameras required for stereo vision."

    def test_camera_serial_numbers_unique(self):
        """Test that each camera has a unique serial number."""
        cameras = list_oculus_cameras()
        if len(cameras) < 2:
            pytest.skip("Need at least 2 cameras to test uniqueness")

        serials = [cam["serial"] for cam in cameras]
        assert len(serials) == len(
            set(serials)
        ), "Cameras must have unique serial numbers"

    def test_open_camera_by_serial(self):
        """Test opening a camera using its serial number."""
        cameras = list_oculus_cameras()
        if not cameras:
            pytest.skip("No cameras available for testing")

        serial = cameras[0]["serial"]
        camera = OculusRiftCV1Camera(serial_number=serial)
        try:
            assert camera is not None
            assert camera.get_frame_width() == 1280
            assert camera.get_frame_height() == 960
        finally:
            camera.close()

    def test_open_nonexistent_serial(self):
        """Test that opening with invalid serial raises DeviceNotFoundError."""
        with pytest.raises(DeviceNotFoundError):
            OculusRiftCV1Camera(serial_number="INVALID_SERIAL_NUMBER_9999")

    def test_context_manager_with_real_camera(self):
        """Test using camera as context manager."""
        cameras = list_oculus_cameras()
        if not cameras:
            pytest.skip("No cameras available for testing")

        serial = cameras[0]["serial"]
        with OculusRiftCV1Camera(serial_number=serial) as camera:
            assert camera is not None
            # Camera should be properly initialized
            assert camera.get_frame_size() == (1280, 960)
        # Camera should be closed after context exit

    def test_dual_camera_initialization(self):
        """Test that both cameras can be opened simultaneously."""
        cameras = list_oculus_cameras()
        if len(cameras) < 2:
            pytest.skip("Need at least 2 cameras for dual camera test")

        left_serial = cameras[0]["serial"]
        right_serial = cameras[1]["serial"]

        left_camera = OculusRiftCV1Camera(serial_number=left_serial)
        right_camera = OculusRiftCV1Camera(serial_number=right_serial)

        try:
            assert left_camera is not None
            assert right_camera is not None
            # Both should have same resolution
            assert left_camera.get_frame_size() == right_camera.get_frame_size()
        finally:
            left_camera.close()
            right_camera.close()

    def test_camera_streaming_start_stop(self):
        """Test starting and stopping camera streaming."""
        cameras = list_oculus_cameras()
        if not cameras:
            pytest.skip("No cameras available for testing")

        serial = cameras[0]["serial"]
        camera = OculusRiftCV1Camera(serial_number=serial)

        frame_received = False

        def frame_callback(frame_data: bytes):
            nonlocal frame_received
            frame_received = True
            # Verify frame data size matches expected resolution
            expected_size = 1280 * 960  # Grayscale 8-bit
            assert len(frame_data) == expected_size

        try:
            camera.start_streaming(frame_callback)
            # Wait briefly for at least one frame
            time.sleep(0.5)
            camera.stop_streaming()

            assert frame_received, "No frames received during streaming"
        finally:
            camera.close()

    def test_undistort_with_real_calibration(self):
        """Test undistortion with actual camera calibration data."""
        cameras = list_oculus_cameras()
        if not cameras:
            pytest.skip("No cameras available for testing")

        serial = cameras[0]["serial"]
        camera = OculusRiftCV1Camera(serial_number=serial)

        try:
            # Test undistortion at image center (should always be valid)
            center_point = (655.0, 475.0)
            assert camera.can_undistort(center_point)

            undistorted = camera.undistort(center_point)
            assert isinstance(undistorted, tuple)
            assert len(undistorted) == 2

            # Undistorted coordinates should be reasonable
            assert 0 <= undistorted[0] <= 1280
            assert 0 <= undistorted[1] <= 960
        finally:
            camera.close()

    def test_camera_properties(self):
        """Test reading camera properties from real device."""
        cameras = list_oculus_cameras()
        if not cameras:
            pytest.skip("No cameras available for testing")

        serial = cameras[0]["serial"]
        camera = OculusRiftCV1Camera(serial_number=serial)

        try:
            # Verify expected Oculus Rift CV1 camera properties
            assert camera.get_frame_width() == 1280
            assert camera.get_frame_height() == 960
            assert camera.get_frame_size() == (1280, 960)

            # Check that calibration data exists
            assert hasattr(camera, "max_r2")
            assert camera.max_r2 > 0
        finally:
            camera.close()
