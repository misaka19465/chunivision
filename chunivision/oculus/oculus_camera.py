"""Oculus Rift CV1 tracking camera interface."""

from __future__ import annotations

import math
import struct
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import usb1

from ..utils.logger import Logger
from .ar0134 import AR0134
from .esp770u import ESP770U
from .exceptions import (
    CalibrationError,
    CommunicationError,
    DeviceInitializationError,
    DeviceNotFoundError,
    InvalidParameterError,
    StreamingError,
)
from .triple_buffer import TripleBuffer
from .uvc import UVC

if TYPE_CHECKING:
    StreamingCallback = Callable[[bytes], None]

logger = Logger.get_logger(__name__)


class OculusRiftCV1Camera:
    """High-level camera interface matching the standalone C++ API.

    This class provides a high-level interface to the Oculus Rift CV1 camera,
    managing USB communication, streaming, calibration, and sensor control.

    Supports context manager protocol for automatic resource cleanup:
        with OculusRiftCV1Camera() as camera:
            camera.start_streaming(callback)
            # ... use camera ...
        # Resources automatically released
    """

    VENDOR_ID = 0x2833
    PRODUCT_ID = 0x0211

    def __init__(
        self,
        serial_number: str,
        context: Optional[usb1.USBContext] = None,
    ) -> None:
        """Initialize camera interface.

        Args:
            serial_number: USB serial number of the camera device
            context: Optional USB context (creates new if None)

        Raises:
            InvalidParameterError: If serial_number is invalid
            DeviceNotFoundError: If requested camera is not found
            DeviceInitializationError: If camera initialization fails
        """
        if not serial_number:
            raise InvalidParameterError("serial_number is required and cannot be empty")

        logger.info(f"Initializing OculusRiftCV1Camera (serial={serial_number})")

        try:
            self.context = context or usb1.USBContext()
            self.handle = self._open_device(serial_number)
            self.frame_size = (1280, 960)
            self.fx = 0.0
            self.fy = 0.0
            self.cx = 0.0
            self.cy = 0.0
            self.k = [0.0, 0.0, 0.0, 0.0]
            self.max_r2 = 0.0
            self.streaming = False
            self.streaming_callback: Optional[StreamingCallback] = None
            self._frame_buffer = bytearray()
            self._frame_ptr = 0
            self._frame_remainder = 0
            self._frame_id = 0
            self._frame_presentation_time = 0
            self._transfers: list[usb1.USBTransfer] = []
            self._transfer_buffers: list[bytearray] = []
            self._stop_event = threading.Event()
            self._setup_device()
            logger.info("OculusRiftCV1Camera initialized successfully")
        except (DeviceNotFoundError, DeviceInitializationError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise DeviceInitializationError(f"Camera initialization failed: {e}") from e

    def __enter__(self) -> "OculusRiftCV1Camera":
        """Enter context manager.

        Returns:
            Self for context manager protocol
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.close()

    # ---- public API -------------------------------------------------
    def get_frame_size(self) -> tuple[int, int]:
        """Get camera frame size.

        Returns:
            Tuple of (width, height) in pixels
        """
        return self.frame_size

    def get_frame_width(self) -> int:
        """Get camera frame width.

        Returns:
            Frame width in pixels
        """
        return self.frame_size[0]

    def get_frame_height(self) -> int:
        """Get camera frame height.

        Returns:
            Frame height in pixels
        """
        return self.frame_size[1]

    def get_calibration_params(self) -> dict[str, Any]:
        """Get camera calibration parameters for distortion correction.

        Returns:
            Dictionary containing:
                - fx, fy: Focal lengths
                - cx, cy: Optical center
                - k: Lens distortion coefficients [k0, k1, k2, k3]
                - max_r2: Maximum squared radius for valid undistortion
                - frame_size: (width, height) tuple
        """
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k": self.k.copy(),
            "max_r2": self.max_r2,
            "frame_size": self.frame_size,
        }

    def start_streaming(self, callback: StreamingCallback) -> None:
        """Start video streaming.

        Args:
            callback: Function to call for each frame

        Raises:
            InvalidParameterError: If callback is None
            StreamingError: If already streaming or streaming start fails
        """
        if callback is None:
            raise InvalidParameterError("Callback cannot be None")

        if self.streaming:
            raise StreamingError("Already streaming")

        logger.info("Starting video streaming")

        try:
            self.streaming_callback = callback
            controller = ESP770U(self.handle)
            sensor = AR0134(controller)
            sensor.init()
            sensor.set_frame_timings(True)
            sensor.set_gain(128)
            sensor.set_coarse_exposure_time(400)
            sensor.set_fine_exposure_time(15)
            sensor.set_sync(False)
            self.handle.claimInterface(1)
            uvc_probe = bytearray(26)
            uvc_probe[2] = 1  # bFormatIndex
            uvc_probe[3] = 4  # bFrameIndex
            frame_interval = 192000
            uvc_probe[4] = frame_interval & 0xFF
            uvc_probe[5] = (frame_interval >> 8) & 0xFF
            uvc_probe[6] = (frame_interval >> 16) & 0xFF
            uvc_probe[7] = (frame_interval >> 24) & 0xFF
            max_frame = self.frame_size[0] * self.frame_size[1]
            uvc_probe[18] = max_frame & 0xFF
            uvc_probe[19] = (max_frame >> 8) & 0xFF
            uvc_probe[20] = (max_frame >> 16) & 0xFF
            uvc_probe[21] = (max_frame >> 24) & 0xFF
            uvc_probe[22] = 0x00
            uvc_probe[23] = 0x0C  # dwMaxPayloadTransferSize = 3072
            UVC.set_cur(self.handle, 1, 0, 1, bytes(uvc_probe))
            uvc_probe_result = bytearray(
                UVC.get_cur(self.handle, 1, 0, 1, len(uvc_probe))
            )
            UVC.set_cur(self.handle, 1, 0, 2, bytes(uvc_probe_result))
            self.handle.setInterfaceAltSetting(1, 2)
            self._frame_buffer = bytearray(max_frame)
            self._frame_ptr = 0
            self._frame_remainder = max_frame
            self._frame_id = 0
            self._stop_event.clear()
            self._allocate_transfers()
            self.streaming = True
            for transfer in self._transfers:
                transfer.submit()
            time.sleep(1.0)
            sensor.set_coarse_exposure_time(800)
            sensor.set_fine_exposure_time(0)
            sensor.set_gain(128)
            logger.info("Video streaming started successfully")
        except (InvalidParameterError, StreamingError):
            raise
        except (CommunicationError, usb1.USBError) as e:
            logger.error(f"Failed to start streaming: {e}")
            self.streaming = False
            raise StreamingError(f"Failed to start streaming: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error starting streaming: {e}")
            self.streaming = False
            raise StreamingError(f"Unexpected error starting streaming: {e}") from e

    def stop_streaming(self) -> None:
        """Stop video streaming.

        Safe to call multiple times. Will attempt to clean up resources
        even if errors occur.
        """
        if not self.streaming:
            logger.debug("stop_streaming called but not streaming")
            return

        logger.info("Stopping video streaming")
        self.streaming = False
        self._stop_event.set()

        # Cancel all transfers
        for transfer in self._transfers:
            try:
                transfer.cancel()
            except (usb1.USBError, Exception) as e:
                logger.debug(f"Error canceling transfer: {e}")

        # Give libusb a moment to process cancellations
        time.sleep(0.1)

        self._transfers.clear()
        self._transfer_buffers.clear()

        # Release interface
        try:
            self.handle.releaseInterface(1)
        except (usb1.USBError, Exception) as e:
            logger.debug(f"Error releasing interface: {e}")

        self.streaming_callback = None
        logger.info("Video streaming stopped")

    def get_horizontal_flip(self) -> bool:
        """Get horizontal flip state.

        Returns:
            True if horizontal flip is enabled

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug("get_horizontal_flip")
        try:
            sensor = AR0134(ESP770U(self.handle))
            return sensor.get_horizontal_flip()
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"get_horizontal_flip failed: {e}")
            raise CommunicationError("Failed to get horizontal flip state") from e

    def get_vertical_flip(self) -> bool:
        """Get vertical flip state.

        Returns:
            True if vertical flip is enabled

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug("get_vertical_flip")
        try:
            sensor = AR0134(ESP770U(self.handle))
            return sensor.get_vertical_flip()
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"get_vertical_flip failed: {e}")
            raise CommunicationError("Failed to get vertical flip state") from e

    def set_flip(self, horizontal: bool, vertical: bool) -> None:
        """Set horizontal and vertical flip modes.

        Args:
            horizontal: Enable horizontal flip
            vertical: Enable vertical flip

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug(f"set_flip: horizontal={horizontal}, vertical={vertical}")
        try:
            sensor = AR0134(ESP770U(self.handle))
            sensor.set_flip(horizontal, vertical)
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"set_flip failed: {e}")
            raise CommunicationError("Failed to set flip modes") from e

    def get_auto_exposure(self) -> bool:
        """Get auto-exposure enable state.

        Returns:
            True if auto-exposure is enabled

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug("get_auto_exposure")
        try:
            sensor = AR0134(ESP770U(self.handle))
            return sensor.get_auto_exposure()
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"get_auto_exposure failed: {e}")
            raise CommunicationError("Failed to get auto-exposure state") from e

    def set_auto_exposure(
        self,
        enable: bool,
        adjust_analog_gain: bool = False,
        adjust_digital_gain: bool = False,
    ) -> None:
        """Configure auto-exposure settings.

        Args:
            enable: Enable auto-exposure
            adjust_analog_gain: Allow adjustment of analog gain
            adjust_digital_gain: Allow adjustment of digital gain

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug(
            f"set_auto_exposure: enable={enable}, analog={adjust_analog_gain}, "
            f"digital={adjust_digital_gain}"
        )
        try:
            sensor = AR0134(ESP770U(self.handle))
            sensor.set_auto_exposure(enable, adjust_analog_gain, adjust_digital_gain)
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"set_auto_exposure failed: {e}")
            raise CommunicationError("Failed to set auto-exposure") from e

    def get_gain(self) -> int:
        """Get current gain value.

        Returns:
            Current gain value (0-65535)

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug("get_gain")
        try:
            sensor = AR0134(ESP770U(self.handle))
            return sensor.get_gain()
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"get_gain failed: {e}")
            raise CommunicationError("Failed to get gain") from e

    def set_gain(self, gain: int) -> None:
        """Set gain value.

        Args:
            gain: Gain value (must be in range 0-65535)

        Raises:
            InvalidParameterError: If gain is out of range
            CommunicationError: If operation fails
        """
        if not (0 <= gain <= 65535):
            raise InvalidParameterError(f"Gain must be in range 0-65535, got {gain}")
        logger.debug(f"set_gain: {gain}")
        try:
            sensor = AR0134(ESP770U(self.handle))
            sensor.set_gain(gain)
        except (InvalidParameterError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"set_gain failed: {e}")
            raise CommunicationError("Failed to set gain") from e

    def get_exposure_time(self) -> int:
        """Get total exposure time.

        Returns:
            Total exposure time in pixel clocks

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug("get_exposure_time")
        try:
            sensor = AR0134(ESP770U(self.handle))
            return sensor.get_exposure_time()
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"get_exposure_time failed: {e}")
            raise CommunicationError("Failed to get exposure time") from e

    def set_exposure_time(self, exposure: int) -> None:
        """Set total exposure time.

        Args:
            exposure: Total exposure time in pixel clocks (must be >= 0)

        Raises:
            InvalidParameterError: If exposure is invalid
            CommunicationError: If operation fails
        """
        if exposure < 0:
            raise InvalidParameterError(f"Exposure time must be >= 0, got {exposure}")
        logger.debug(f"set_exposure_time: {exposure}")
        try:
            sensor = AR0134(ESP770U(self.handle))
            sensor.set_exposure_time(exposure)
        except (InvalidParameterError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"set_exposure_time failed: {e}")
            raise CommunicationError("Failed to set exposure time") from e

    # ---- internal helpers -------------------------------------------
    def _open_device(self, serial_number: str) -> usb1.USBDeviceHandle:
        """Open USB device by serial number.

        Args:
            serial_number: USB serial string to match

        Returns:
            USB device handle

        Raises:
            DeviceNotFoundError: If device is not found
        """

        def _get_serial_for_device(device) -> Optional[str]:
            try:
                # Some backends expose serial directly
                serial = device.getSerialNumber()
                if serial:
                    return serial
            except Exception:
                pass
            try:
                # Try to open and query string descriptor
                h = device.open()
                try:
                    idx = device.getSerialNumber()
                    if idx:
                        try:
                            return h.getStringDescriptor(idx)
                        except Exception:
                            return None
                finally:
                    try:
                        h.close()
                    except Exception:
                        pass
            except Exception:
                pass
            return None

        logger.debug(f"Opening device by serial: {serial_number}")

        try:
            match_count = 0
            handle: Optional[usb1.USBDeviceHandle] = None
            for device in self.context.getDeviceList(skip_on_error=True):
                if (
                    device.getVendorID() == self.VENDOR_ID
                    and device.getProductID() == self.PRODUCT_ID
                ):
                    match_count += 1
                    dev_serial = _get_serial_for_device(device)
                    if dev_serial is not None and dev_serial == serial_number:
                        handle = device.open()
                        break

            if handle is None:
                error_msg = (
                    f"OculusRiftCV1Camera: no device with serial '{serial_number}' found "
                    f"(checked {match_count} Oculus cameras)"
                )
                logger.error(error_msg)
                raise DeviceNotFoundError(error_msg)
            logger.info(f"Device opened successfully (serial={serial_number})")
            return handle
        except DeviceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to open device: {e}")
            raise DeviceNotFoundError(f"Failed to open device: {e}") from e

    def _setup_device(self) -> None:
        """Setup and initialize camera device.

        Performs full device initialization including reset, configuration,
        controller and radio setup, and calibration data loading.

        Raises:
            DeviceInitializationError: If setup fails
            CalibrationError: If calibration data is invalid
            CommunicationError: If communication with device fails
        """
        logger.info("Setting up camera device")

        try:
            self.handle.setAutoDetachKernelDriver(True)
            print("Resetting device...")
            try:
                self.handle.resetDevice()
            except usb1.USBError as exc:
                if exc.value != usb1.ERROR_NOT_FOUND:
                    print(f"Warning: reset failed: {exc}")
            time.sleep(2.0)
            try:
                config = self.handle.getConfiguration()
                if config != 1:
                    print("Setting configuration to 1...")
                    self.handle.setConfiguration(1)
                    time.sleep(0.2)
            except usb1.USBError as exc:
                print(f"Warning: could not read/set configuration: {exc}")
            self.handle.claimInterface(0)
            print("Successfully claimed interface 0")
            print("Waiting for device to initialize...")
            time.sleep(2.0)
            controller = ESP770U(self.handle)
            print("Warming up UVC interface...")
            try:
                UVC.get_cur(self.handle, 0, 4, 3, 4)
            except Exception:
                pass
            time.sleep(0.5)
            print("Querying firmware version...")
            firmware_version = None
            for attempt in range(5):
                try:
                    if attempt:
                        print(f"Retry attempt {attempt + 1}/5...")
                        time.sleep(0.5 * attempt)
                    firmware_version = controller.query_firmware_version()
                    break
                except Exception as exc:
                    if attempt == 4:
                        print(
                            "Error: Failed to query firmware version after 5 attempts"
                        )
                        raise
                    print(f"Attempt {attempt + 1} failed: {exc}")
            if firmware_version is not None:
                print(f"Firmware version: {firmware_version}")
            controller.init_controller()
            controller.init_radio()

            logger.info("Reading calibration data")
            cal_data = controller.read_memory(0x1D000, 128)

            def _read_float(offset: int) -> float:
                if offset + 4 > len(cal_data):
                    raise CalibrationError(
                        f"Calibration data too short for offset {offset}"
                    )
                val = cal_data[offset : offset + 4]
                return struct.unpack_from("<f", val)[0]

            self.fy = self.fx = _read_float(0x30)
            self.cx = _read_float(0x34)
            self.cy = _read_float(0x38)
            self.k[0] = _read_float(0x48)
            self.k[1] = _read_float(0x4C)
            self.k[2] = _read_float(0x50)
            self.k[3] = _read_float(0x54)
            print("Calibration parameters:")
            print(f"  Focal length: {self.fx}, {self.fy}")
            print(f"  Focus point: {self.cx}, {self.cy}")
            print(
                f"  Lens distortion: {self.k[0]}, {self.k[1]}, {self.k[2]}, {self.k[3]}"
            )
            kappas = (5.16403e-07, 2.44492e-13, 6.881e-19)
            center = (655.052, 475.083)
            max_r = math.sqrt(center[0] * center[0] + center[1] * center[1])
            last_corrected_r = 0.0
            for r in range(1, int(max_r)):
                r2 = float(r * r)
                radial = 0.0
                for kappa in reversed(kappas):
                    radial = (radial + kappa) * r2
                radial += 1.0
                corrected_r = r * radial
                if corrected_r <= last_corrected_r:
                    max_r = r - 1
                    break
                last_corrected_r = corrected_r
            self.max_r2 = float(max_r * max_r)

            logger.info("Camera device setup complete")

        except (DeviceInitializationError, CalibrationError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"Device setup failed: {e}")
            raise DeviceInitializationError(f"Device setup failed: {e}") from e

    def handle_events(self, timeout: float = 0.01) -> None:
        """Process pending USB events.

        Call this periodically when streaming to process USB transfers.

        Args:
            timeout: Timeout in seconds (must be >= 0)

        Raises:
            InvalidParameterError: If timeout is negative
        """
        if timeout < 0:
            raise InvalidParameterError(f"Timeout must be >= 0, got {timeout}")

        if self.context:
            self.context.handleEventsTimeout(int(timeout * 1000000))

    def _allocate_transfers(self) -> None:
        self._transfers = []
        self._transfer_buffers = []
        num_transfers = 7
        packets = 24
        packet_size = 16384
        for _ in range(num_transfers):
            buf = bytearray(packets * packet_size)
            transfer = self.handle.getTransfer(iso_packets=packets)
            transfer.setIsochronous(
                0x81,
                buf,
                callback=self._on_transfer,
                timeout=1000,
                iso_transfer_length_list=[packet_size] * packets,
            )
            self._transfers.append(transfer)
            self._transfer_buffers.append(buf)

    def _on_transfer(self, transfer: usb1.USBTransfer) -> None:
        if self._stop_event.is_set():
            return
        status = transfer.getStatus()
        if status == usb1.TRANSFER_COMPLETED:
            for packet_status, packet_data in transfer.iterISO():
                if packet_status == usb1.TRANSFER_COMPLETED and len(packet_data) >= 12:
                    self._consume_packet(packet_data)
        try:
            if self.streaming:
                transfer.submit()
        except usb1.USBError as exc:
            print(f"Failed to resubmit transfer: {exc}")

    def _consume_packet(self, packet: bytes) -> None:
        header_size = packet[0]
        if header_size < 12 or len(packet) < header_size:
            return
        if packet[1] & 0x40:
            return
        packet_frame_id = packet[1] & 0x01
        if self._frame_id != packet_frame_id:
            self._frame_id = packet_frame_id
            self._frame_ptr = 0
            self._frame_remainder = self.frame_size[0] * self.frame_size[1]
            pt = packet[5]
            pt = (pt << 8) | packet[4]
            pt = (pt << 8) | packet[3]
            pt = (pt << 8) | packet[2]
            self._frame_presentation_time = pt
        if self._frame_remainder <= 0:
            return
        payload = packet[header_size:]
        if len(payload) <= self._frame_remainder:
            self._frame_buffer[self._frame_ptr : self._frame_ptr + len(payload)] = (
                payload
            )
            self._frame_ptr += len(payload)
            self._frame_remainder -= len(payload)
            if self._frame_remainder == 0 and self.streaming_callback:
                finished = bytes(self._frame_buffer)
                self._frame_buffer = bytearray(self.frame_size[0] * self.frame_size[1])
                self.streaming_callback(finished)
        else:
            self._frame_remainder = 0

    def close(self) -> None:
        """Close camera and release all resources.

        Safe to call multiple times. Attempts to release all resources
        even if errors occur during cleanup.
        """
        logger.info("Closing camera")

        # Stop streaming if active
        if self.streaming:
            try:
                self.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping streaming during close: {e}")

        # Release interface 0
        try:
            if hasattr(self, "handle") and self.handle:
                self.handle.releaseInterface(0)
        except (usb1.USBError, Exception) as e:
            logger.debug(f"Error releasing interface 0: {e}")

        # Close handle
        try:
            if hasattr(self, "handle") and self.handle:
                self.handle.close()
        except (usb1.USBError, Exception) as e:
            logger.debug(f"Error closing handle: {e}")

        # Clear context reference
        if hasattr(self, "context"):
            self.context = None  # type: ignore

        logger.info("Camera closed")


def list_oculus_cameras(context: Optional[usb1.USBContext] = None) -> list[dict]:
    """List connected Oculus Rift CV1 cameras.

    Returns a list of dictionaries with keys: `index`, `vendor_id`, `product_id`, `serial_number`.

    Note: this function does not prompt the user; it only enumerates devices.
    """
    ctx = context or usb1.USBContext()
    devices = []
    try:
        idx = 0
        for device in ctx.getDeviceList(skip_on_error=True):
            try:
                if (
                    device.getVendorID() == OculusRiftCV1Camera.VENDOR_ID
                    and device.getProductID() == OculusRiftCV1Camera.PRODUCT_ID
                ):
                    serial = None
                    try:
                        serial = device.getSerialNumber()
                    except Exception:
                        # try via handle
                        try:
                            h = device.open()
                            try:
                                sidx = device.getSerialNumber()
                                if sidx:
                                    serial = h.getStringDescriptor(sidx)
                            finally:
                                try:
                                    h.close()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    devices.append(
                        {
                            "index": idx,
                            "vendor_id": device.getVendorID(),
                            "product_id": device.getProductID(),
                            "serial_number": serial,
                        }
                    )
                    idx += 1
            except Exception:
                # skip this device on error, continue enumeration
                pass
    except Exception:
        # best-effort listing; return what we have
        pass
    return devices


__all__ = ["OculusRiftCV1Camera", "list_oculus_cameras"]
