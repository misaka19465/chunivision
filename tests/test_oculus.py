"""Comprehensive tests for oculus camera library."""

import threading
import time
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

# Test imports
try:
    import usb1
except ImportError:
    pytest.skip("usb1 not installed", allow_module_level=True)

from chunivision.oculus.camera import AR0134, ESP770U, OculusRiftCV1Camera, UVC
from chunivision.oculus.exceptions import (
    CalibrationError,
    CommunicationError,
    DeviceInitializationError,
    DeviceNotFoundError,
    InvalidParameterError,
    StreamingError,
)
from chunivision.oculus.triple_buffer import TripleBuffer


class TestTripleBuffer:
    """Tests for TripleBuffer thread-safe frame buffering."""

    def test_initial_state(self):
        """Test buffer starts empty."""
        buffer = TripleBuffer[int]()
        assert buffer.get_latest() is None

    def test_publish_and_get(self):
        """Test publishing and retrieving values."""
        buffer = TripleBuffer[int]()
        buffer.publish(42)
        assert buffer.get_latest() == 42

    def test_multiple_publishes(self):
        """Test that latest value overwrites previous."""
        buffer = TripleBuffer[int]()
        buffer.publish(1)
        buffer.publish(2)
        buffer.publish(3)
        assert buffer.get_latest() == 3

    def test_get_latest_idempotent(self):
        """Test that get_latest() can be called multiple times."""
        buffer = TripleBuffer[int]()
        buffer.publish(100)
        assert buffer.get_latest() == 100
        assert buffer.get_latest() == 100
        assert buffer.get_latest() == 100

    def test_clear(self):
        """Test clearing buffer."""
        buffer = TripleBuffer[int]()
        buffer.publish(42)
        assert buffer.get_latest() == 42
        buffer.clear()
        assert buffer.get_latest() is None

    def test_thread_safety(self):
        """Test concurrent access from multiple threads."""
        buffer = TripleBuffer[int]()
        iterations = 1000
        errors = []

        def publisher():
            try:
                for i in range(iterations):
                    buffer.publish(i)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(iterations):
                    buffer.get_latest()
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publisher),
            threading.Thread(target=publisher),
            threading.Thread(target=consumer),
            threading.Thread(target=consumer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"

    def test_different_types(self):
        """Test buffer works with different types."""
        # Bytes
        buffer_bytes = TripleBuffer[bytes]()
        buffer_bytes.publish(b"hello")
        assert buffer_bytes.get_latest() == b"hello"

        # String
        buffer_str = TripleBuffer[str]()
        buffer_str.publish("world")
        assert buffer_str.get_latest() == "world"

        # Complex object
        buffer_obj = TripleBuffer[dict]()
        data = {"key": "value"}
        buffer_obj.publish(data)
        assert buffer_obj.get_latest() == data


class TestUVC:
    """Tests for UVC control transfer helpers."""

    @patch("usb1.USBDeviceHandle")
    def test_set_cur_success(self, mock_handle):
        """Test successful SET_CUR operation."""
        mock_handle.controlWrite.return_value = None
        UVC.set_cur(mock_handle, 0, 4, 3, b"\x01\x02\x03")
        mock_handle.controlWrite.assert_called_once()

    @patch("usb1.USBDeviceHandle")
    def test_set_cur_usb_error(self, mock_handle):
        """Test SET_CUR with USB error."""
        mock_handle.controlWrite.side_effect = usb1.USBError("USB error")
        with pytest.raises(CommunicationError, match="Failed to set UVC control"):
            UVC.set_cur(mock_handle, 0, 4, 3, b"\x01\x02\x03")

    @patch("usb1.USBDeviceHandle")
    def test_get_cur_success(self, mock_handle):
        """Test successful GET_CUR operation."""
        mock_handle.controlRead.return_value = b"\x01\x02\x03\x04"
        result = UVC.get_cur(mock_handle, 0, 4, 3, 4)
        assert result == b"\x01\x02\x03\x04"

    @patch("usb1.USBDeviceHandle")
    def test_get_cur_invalid_length(self, mock_handle):
        """Test GET_CUR with invalid length parameter."""
        with pytest.raises(InvalidParameterError, match="Length must be > 0"):
            UVC.get_cur(mock_handle, 0, 4, 3, 0)
        with pytest.raises(InvalidParameterError):
            UVC.get_cur(mock_handle, 0, 4, 3, -1)

    @patch("usb1.USBDeviceHandle")
    def test_get_cur_usb_error(self, mock_handle):
        """Test GET_CUR with USB error."""
        mock_handle.controlRead.side_effect = usb1.USBError("USB error")
        with pytest.raises(CommunicationError, match="Failed to get UVC control"):
            UVC.get_cur(mock_handle, 0, 4, 3, 4)

    @patch("usb1.USBDeviceHandle")
    def test_get_len_success(self, mock_handle):
        """Test successful GET_LEN operation."""
        mock_handle.controlRead.return_value = b"\x0a\x00"  # Length = 10
        result = UVC.get_len(mock_handle, 0, 4, 3)
        assert result == 10

    @patch("usb1.USBDeviceHandle")
    def test_get_len_invalid_response(self, mock_handle):
        """Test GET_LEN with invalid response length."""
        mock_handle.controlRead.return_value = b"\x0a"  # Only 1 byte
        with pytest.raises(CommunicationError, match="unexpected length"):
            UVC.get_len(mock_handle, 0, 4, 3)

    @patch("usb1.USBDeviceHandle")
    def test_get_len_usb_error(self, mock_handle):
        """Test GET_LEN with USB error."""
        mock_handle.controlRead.side_effect = usb1.USBError("USB error")
        with pytest.raises(
            CommunicationError, match="Failed to get UVC control length"
        ):
            UVC.get_len(mock_handle, 0, 4, 3)


class TestESP770U:
    """Tests for ESP770U camera controller."""

    @pytest.fixture
    def mock_handle(self):
        """Create mock USB handle."""
        return MagicMock(spec=usb1.USBDeviceHandle)

    @pytest.fixture
    def controller(self, mock_handle):
        """Create ESP770U controller with mocked handle."""
        return ESP770U(mock_handle)

    def test_init(self, mock_handle):
        """Test controller initialization."""
        controller = ESP770U(mock_handle)
        assert controller.handle == mock_handle

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_read_register_success(self, mock_get, mock_set, controller):
        """Test successful register read."""
        mock_get.return_value = b"\x82\x42\x00\x00"
        result = controller.read_register(0xF05A)
        assert result == 0x42

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_read_register_invalid_response(self, mock_get, mock_set, controller):
        """Test register read with invalid response."""
        mock_get.return_value = b"\x00\x42\x00\x00"  # Wrong command byte
        with pytest.raises(CommunicationError, match="invalid response"):
            controller.read_register(0xF05A)

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_write_register_success(self, mock_get, mock_set, controller):
        """Test successful register write."""
        mock_get.return_value = b"\x02\xf0\x5a\x01"
        controller.write_register(0xF05A, 0x01)
        # Should call SET_CUR and GET_CUR for verification
        assert mock_set.call_count >= 1
        assert mock_get.call_count >= 1

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_write_register_verification_failure(self, mock_get, mock_set, controller):
        """Test register write with verification failure."""
        mock_get.return_value = b"\x02\xf0\x5a\xff"  # Wrong value
        with pytest.raises(CommunicationError, match="invalid response"):
            controller.write_register(0xF05A, 0x01)

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_retry_logic(self, mock_get, mock_set, controller):
        """Test retry logic on failures."""
        # Fail twice, then succeed
        mock_get.side_effect = [
            usb1.USBError("timeout"),
            usb1.USBError("timeout"),
            b"\x82\x42\x00\x00",
        ]
        result = controller.read_register(0xF05A)
        assert result == 0x42
        assert mock_get.call_count == 3

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_retry_exhaustion(self, mock_get, mock_set, controller):
        """Test all retries exhausted."""
        mock_get.side_effect = usb1.USBError("persistent error")
        with pytest.raises(
            CommunicationError, match="Failed to execute set/get current after"
        ):
            controller.read_register(0xF05A)
        assert mock_get.call_count == 3  # max_retries

    def test_write_radio_invalid_data(self, controller):
        """Test _write_radio with invalid data."""
        with pytest.raises(InvalidParameterError, match="cannot be empty"):
            controller._write_radio(b"")
        with pytest.raises(InvalidParameterError, match="too large"):
            controller._write_radio(b"x" * 127)

    @patch.object(UVC, "set_cur")
    @patch.object(UVC, "get_cur")
    def test_query_firmware_version(self, mock_get, mock_set, controller):
        """Test firmware version query."""
        mock_get.return_value = b"\xa0\x05\x00\x00"  # Version 5
        version = controller.query_firmware_version()
        assert version == 5


class TestAR0134:
    """Tests for AR0134 imaging sensor."""

    @pytest.fixture
    def mock_controller(self):
        """Create mock ESP770U controller."""
        return MagicMock(spec=ESP770U)

    @pytest.fixture
    def sensor(self, mock_controller):
        """Create AR0134 sensor with mocked controller."""
        return AR0134(mock_controller)

    def test_init(self, mock_controller):
        """Test sensor initialization."""
        sensor = AR0134(mock_controller)
        assert sensor.controller == mock_controller

    def test_read_register(self, sensor, mock_controller):
        """Test register read delegation."""
        mock_controller.read_i2c.return_value = 0x1234
        result = sensor.read_register(AR0134.CHIP_VERSION_REG)
        mock_controller.read_i2c.assert_called_once_with(0x20, AR0134.CHIP_VERSION_REG)
        assert result == 0x1234

    def test_write_register(self, sensor, mock_controller):
        """Test register write delegation."""
        sensor.write_register(AR0134.GLOBAL_GAIN, 128)
        mock_controller.write_i2c.assert_called_once_with(0x20, AR0134.GLOBAL_GAIN, 128)

    def test_init_success(self, sensor, mock_controller):
        """Test successful sensor initialization."""
        mock_controller.read_i2c.side_effect = [
            0x2406,  # CHIP_VERSION_REG
            0x1300,  # REVISION_NUMBER
            0x0080,  # DIGITAL_TEST (MONO_CHROME)
            0x0000,  # EMBEDDED_DATA_CONTROL
        ]
        sensor.init()
        assert mock_controller.read_i2c.call_count == 4
        assert mock_controller.write_i2c.call_count == 1

    def test_init_invalid_version(self, sensor, mock_controller):
        """Test sensor init with wrong chip version."""
        mock_controller.read_i2c.side_effect = [
            0x0000,  # Wrong chip version
            0x1300,
        ]
        with pytest.raises(DeviceInitializationError, match="unsupported chip version"):
            sensor.init()

    def test_flip_controls(self, sensor, mock_controller):
        """Test horizontal/vertical flip controls."""
        # Test get horizontal flip
        mock_controller.read_i2c.return_value = AR0134.HORIZ_MIRROR
        assert sensor.get_horizontal_flip() is True

        mock_controller.read_i2c.return_value = 0x0000
        assert sensor.get_horizontal_flip() is False

        # Test get vertical flip
        mock_controller.read_i2c.return_value = AR0134.VERT_FLIP
        assert sensor.get_vertical_flip() is True

        # Test set_flip
        mock_controller.read_i2c.return_value = 0x0000
        sensor.set_flip(True, True)
        # Should write READ_MODE with both flip bits set
        call_args = mock_controller.write_i2c.call_args[0]
        assert call_args[0] == 0x20  # I2C address
        assert call_args[1] == AR0134.READ_MODE
        assert call_args[2] & (AR0134.HORIZ_MIRROR | AR0134.VERT_FLIP)

    def test_auto_exposure_controls(self, sensor, mock_controller):
        """Test auto exposure enable/disable."""
        # Test get
        mock_controller.read_i2c.return_value = AR0134.AE_ENABLE
        assert sensor.get_auto_exposure() is True

        mock_controller.read_i2c.return_value = 0x0000
        assert sensor.get_auto_exposure() is False

        # Test set
        mock_controller.read_i2c.return_value = 0x0000
        sensor.set_auto_exposure(True, True, True)
        call_args = mock_controller.write_i2c.call_args[0]
        assert call_args[2] & AR0134.AE_ENABLE
        assert call_args[2] & AR0134.AUTO_AG_EN
        assert call_args[2] & AR0134.AUTO_DG_EN

    def test_gain_controls(self, sensor, mock_controller):
        """Test gain get/set."""
        mock_controller.read_i2c.return_value = 128
        assert sensor.get_gain() == 128

        sensor.set_gain(256)
        mock_controller.write_i2c.assert_called_with(0x20, AR0134.GLOBAL_GAIN, 256)

        # Test validation of out-of-range value
        with pytest.raises(InvalidParameterError, match="Gain must be in range"):
            sensor.set_gain(0x1FFFF)  # Value larger than 16 bits

    def test_exposure_time_controls(self, sensor, mock_controller):
        """Test exposure time calculations."""
        # Setup frame width
        mock_controller.read_i2c.side_effect = [
            1388,  # LINE_LENGTH_PCK (frame width)
            800,  # COARSE_INTEGRATION_TIME
            15,  # FINE_INTEGRATION_TIME
        ]
        exposure = sensor.get_exposure_time()
        assert exposure == 800 * 1388 + 15

        # Test set_exposure_time - reset side_effect
        mock_controller.read_i2c.side_effect = [1388]  # LINE_LENGTH_PCK
        mock_controller.read_i2c.return_value = 1388
        sensor.set_exposure_time(1000000)
        # Should split into coarse and fine
        assert mock_controller.write_i2c.call_count >= 2

    def test_window_controls(self, sensor, mock_controller):
        """Test sensor window/ROI configuration."""
        sensor.set_window(100, 200, 640, 480)
        assert mock_controller.write_i2c.call_count == 4
        # Verify calls for X/Y start/end
        calls = mock_controller.write_i2c.call_args_list
        registers = [call[0][1] for call in calls]
        assert AR0134.Y_ADDR_START in registers
        assert AR0134.X_ADDR_START in registers
        assert AR0134.Y_ADDR_END in registers
        assert AR0134.X_ADDR_END in registers


class TestOculusRiftCV1Camera:
    """Tests for high-level camera interface."""

    @pytest.fixture
    def mock_context(self):
        """Create mock USB context."""
        context = MagicMock(spec=usb1.USBContext)
        return context

    @pytest.fixture
    def mock_device(self):
        """Create mock USB device."""
        device = MagicMock()
        device.getVendorID.return_value = 0x2833
        device.getProductID.return_value = 0x0211
        return device

    @pytest.fixture
    def mock_handle(self):
        """Create mock USB device handle."""
        handle = MagicMock(spec=usb1.USBDeviceHandle)
        handle.controlRead.return_value = b"\x00" * 128
        handle.controlWrite.return_value = None
        handle.getConfiguration.return_value = 1
        return handle

    def test_empty_serial_number(self, mock_context):
        """Test initialization with empty serial number."""
        with pytest.raises(InvalidParameterError, match="serial_number is required"):
            OculusRiftCV1Camera(serial_number="", context=mock_context)

    @patch("chunivision.oculus.oculus_camera.usb1.USBContext")
    def test_device_not_found(self, mock_usb_context_class):
        """Test initialization when device not found."""
        mock_context = MagicMock()
        mock_context.getDeviceList.return_value = []
        mock_usb_context_class.return_value = mock_context

        with pytest.raises(DeviceNotFoundError):
            OculusRiftCV1Camera(serial_number="NONEXISTENT_SERIAL")

    def test_context_manager(self, mock_context, mock_device, mock_handle):
        """Test camera as context manager."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            with OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            ) as camera:
                assert camera is not None
            # close() should have been called
            assert mock_handle.close.called

    def test_frame_size_getters(self, mock_context, mock_device, mock_handle):
        """Test frame size accessors."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )
            assert camera.get_frame_size() == (1280, 960)
            assert camera.get_frame_width() == 1280
            assert camera.get_frame_height() == 960
            camera.close()

    def test_can_undistort(self, mock_context, mock_device, mock_handle):
        """Test undistortion validity check."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )
            # Set max_r2 to a known value for testing
            camera.max_r2 = 655 * 655 + 475 * 475
            # Center pixel should be valid
            assert camera.can_undistort((655.0, 475.0))
            # Far corner should be invalid (beyond max_r2)
            assert not camera.can_undistort((0.0, 0.0))
            camera.close()

    def test_undistort_distort_roundtrip(self, mock_context, mock_device, mock_handle):
        """Test undistort/distort form approximate inverse."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )
            camera.max_r2 = 655 * 655 + 475 * 475  # Allow all pixels

            # Test roundtrip for several points
            test_points = [
                (655.0, 475.0),  # Center
                (700.0, 500.0),
                (600.0, 450.0),
                (800.0, 600.0),
            ]

            for original in test_points:
                undistorted = camera.undistort(original)
                back_to_distorted = camera.distort(undistorted)

                # Should be close (within 1 pixel)
                assert abs(back_to_distorted[0] - original[0]) < 1.0
                assert abs(back_to_distorted[1] - original[1]) < 1.0

            camera.close()

    def test_start_streaming_twice_error(self, mock_context, mock_device, mock_handle):
        """Test that starting streaming twice raises error."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        def dummy_callback(frame: bytes) -> None:
            pass

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            with patch.object(OculusRiftCV1Camera, "_allocate_transfers"):
                camera = OculusRiftCV1Camera(
                    serial_number="TEST_SERIAL", context=mock_context
                )
                camera.streaming = True  # Simulate already streaming

                with pytest.raises(StreamingError, match="Already streaming"):
                    camera.start_streaming(dummy_callback)

                camera.close()

    def test_stop_streaming_when_not_streaming(
        self, mock_context, mock_device, mock_handle
    ):
        """Test that stop_streaming is safe when not streaming."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )
            # Should not raise
            camera.stop_streaming()
            camera.stop_streaming()  # Safe to call multiple times
            camera.close()

    def test_threading_lock_protection(self, mock_context, mock_device, mock_handle):
        """Test that streaming uses thread-safe event."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )
            assert hasattr(camera, "_stop_event")
            assert isinstance(camera._stop_event, threading.Event)
            camera.close()

    def test_close_cleanup(self, mock_context, mock_device, mock_handle):
        """Test that close() properly releases resources."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )
            camera.close()

            # Should release interface and close handle
            assert mock_handle.releaseInterface.called
            assert mock_handle.close.called

    def test_close_handles_errors(self, mock_context, mock_device, mock_handle):
        """Test that close() handles errors gracefully."""
        mock_device.open.return_value = mock_handle
        mock_device.getSerialNumber.return_value = "TEST_SERIAL"
        mock_context.getDeviceList.return_value = [mock_device]

        with patch.object(OculusRiftCV1Camera, "_setup_device"):
            camera = OculusRiftCV1Camera(
                serial_number="TEST_SERIAL", context=mock_context
            )

            # Make releaseInterface and close raise errors
            mock_handle.releaseInterface.side_effect = usb1.USBError("error")
            mock_handle.close.side_effect = usb1.USBError("error")

            # Should not raise
            camera.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
