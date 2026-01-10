"""Python rewrite of the Oculus Rift CV1 camera driver.

This module mirrors the minimal standalone C++ implementation using libusb.
It uses the `usb1` bindings (libusb-1.0) and keeps the public surface area
similar to the original: `OculusRiftCV1Camera` exposes streaming controls,
exposure/gain, and simple undistortion helpers.

The implementation closely follows the logic in `src/OculusRiftCV1Camera.cpp`.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from typing import Callable, Optional

try:
    import usb1
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "usb1 (libusb1 Python bindings) is required. Install with 'pip install libusb1'."
    ) from exc

StreamingCallback = Callable[[bytes], None]


class UVC:
    """Helpers for UVC control transfers."""

    @staticmethod
    def set_cur(handle: usb1.USBDeviceHandle, interface: int, entity: int, selector: int, data: bytes) -> None:
        request_type = usb1.TYPE_CLASS | usb1.RECIPIENT_INTERFACE
        request = 0x01  # SET_CUR
        handle.controlWrite(request_type, request, selector << 8, (entity << 8) | interface, data, timeout=1000)

    @staticmethod
    def get_cur(handle: usb1.USBDeviceHandle, interface: int, entity: int, selector: int, length: int) -> bytes:
        request_type = usb1.ENDPOINT_IN | usb1.TYPE_CLASS | usb1.RECIPIENT_INTERFACE
        request = 0x81  # GET_CUR
        return handle.controlRead(request_type, request, selector << 8, (entity << 8) | interface, length, timeout=1000)

    @staticmethod
    def get_len(handle: usb1.USBDeviceHandle, interface: int, entity: int, selector: int) -> int:
        request_type = usb1.TYPE_CLASS | usb1.RECIPIENT_INTERFACE
        request = 0x85  # GET_LEN
        data = handle.controlRead(request_type, request, selector << 8, (entity << 8) | interface, 2, timeout=1000)
        if len(data) != 2:
            raise RuntimeError("UVC::get_len returned unexpected length")
        return data[0] | (data[1] << 8)


class ESP770U:
    """Camera controller access via extension unit controls."""

    EXTENSION_UNIT = 4
    I2C = 2
    REGISTER = 3
    COUNTER = 10
    CONTROL = 11
    DATA = 12

    def __init__(self, handle: usb1.USBDeviceHandle) -> None:
        self.handle = handle

    def _set_get_cur(self, selector: int, buffer: bytearray) -> None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                UVC.set_cur(self.handle, 0, self.EXTENSION_UNIT, selector, bytes(buffer))
                time.sleep(0.1)
                data = UVC.get_cur(self.handle, 0, self.EXTENSION_UNIT, selector, len(buffer))
                buffer[:] = data
                return
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.3 * (attempt + 1))

    def read_register(self, register_index: int) -> int:
        command = bytearray([0x82, (register_index >> 8) & 0xFF, register_index & 0xFF, 0x00])
        self._set_get_cur(self.REGISTER, command)
        if command[0] != 0x82 or command[2] != 0x00:
            raise RuntimeError("ESP770U::read_register: invalid response")
        return command[1]

    def write_register(self, register_index: int, value: int) -> None:
        command = bytearray([0x02, (register_index >> 8) & 0xFF, register_index & 0xFF, value & 0xFF])
        self._set_get_cur(self.REGISTER, command)
        if command[0] != 0x02 or command[1] != (register_index >> 8) & 0xFF or command[2] != register_index & 0xFF or command[3] != value:
            raise RuntimeError("ESP770U::write_register: invalid response")

    def get_counter(self) -> int:
        data = UVC.get_cur(self.handle, 0, self.EXTENSION_UNIT, self.COUNTER, 1)
        return data[0]

    def set_counter(self, new_counter: int) -> None:
        UVC.set_cur(self.handle, 0, self.EXTENSION_UNIT, self.COUNTER, bytes([new_counter & 0xFF]))

    def _spi_set_control(self, handle: int, length: int) -> None:
        command = bytearray(16)
        command[0] = 0x00
        command[1] = handle & 0xFF
        command[2] = 0x80
        command[3] = 0x01
        command[9] = length & 0xFF
        UVC.set_cur(self.handle, 0, self.EXTENSION_UNIT, self.CONTROL, bytes(command))

    def _spi_set_data(self, data: bytes) -> None:
        UVC.set_cur(self.handle, 0, self.EXTENSION_UNIT, self.DATA, data)

    def _spi_get_data(self, length: int) -> bytes:
        return UVC.get_cur(self.handle, 0, self.EXTENSION_UNIT, self.DATA, length)

    def _write_radio(self, data: bytes) -> None:
        if len(data) > 126:
            raise ValueError("ESP770U::write_radio: block too large")
        buffer = bytearray(127)
        checksum = 0
        for i, b in enumerate(data):
            buffer[i] = b
            checksum -= b
        buffer[126] = checksum & 0xFF

        self._spi_set_control(0x81, len(buffer))
        self._spi_set_data(buffer)
        self._spi_set_control(0x41, len(buffer))
        buffer = bytearray(self._spi_get_data(len(buffer)))

        self._spi_set_control(0x81, len(buffer))
        self._spi_set_data(bytes(len(buffer)))
        self._spi_set_control(0x41, len(buffer))
        buffer = bytearray(self._spi_get_data(len(buffer)))

        if sum(buffer) & 0xFF or buffer[0] != data[0] or buffer[1] != data[1]:
            raise RuntimeError("ESP770U::write_radio: invalid return buffer")

    def query_firmware_version(self) -> int:
        command = bytearray([0xA0, 0x03, 0x00, 0x00])
        self._set_get_cur(self.REGISTER, command)
        if command[0] != 0xA0 or command[2] != 0x00 or command[3] != 0x00:
            raise RuntimeError("ESP770U::query_firmware_version: invalid response")
        return command[1]

    def read_memory(self, address: int, length: int) -> bytes:
        counter = self.get_counter()
        command = bytearray(16)
        command[0] = counter
        command[1] = 0x41
        command[2] = 0x03
        command[3] = 0x01
        command[5] = (address >> 16) & 0xFF
        command[6] = (address >> 8) & 0xFF
        command[7] = address & 0xFF
        command[8] = (length >> 8) & 0xFF
        command[9] = length & 0xFF
        UVC.set_cur(self.handle, 0, self.EXTENSION_UNIT, self.CONTROL, bytes(command))
        data = UVC.get_cur(self.handle, 0, self.EXTENSION_UNIT, self.DATA, length)
        self.set_counter(counter)
        return data

    def init_controller(self) -> None:
        value = self.read_register(0xF05A)
        if value not in (0x01, 0x03):
            print(f"ESP770U::initController: unexpected 0x{value:02x} in 0xF05A")
        self.write_register(0xF05A, 0x01)
        value = self.read_register(0xF018)
        self.write_register(0xF018, 0x0F)
        value = self.read_register(0xF017)
        if value not in (0xEC, 0xED):
            print(f"ESP770U::initController: unexpected 0x{value:02x} in 0xF017")
        self.write_register(0xF017, value | 0x01)
        self.write_register(0xF017, value & ~0x01)
        self.write_register(0xF018, 0x0E)

    def init_radio(self) -> None:
        time.sleep(0.05)
        for payload in (b"\x01\x01", b"\x11\x01"):
            self._write_radio(payload)
        value = self.read_register(0xF014)
        if value not in (0x1A, 0x1B):
            print(f"ESP770U::initRadio: unexpected 0x{value:02x} in 0xF014")
        self._write_radio(b"\x21\x01")
        self._write_radio(b"\x31\x01\x00")

    def setup_radio(self, radio_id: int) -> None:
        command0 = bytes([0x40, 0x10, radio_id & 0xFF, (radio_id >> 8) & 0xFF, (radio_id >> 16) & 0xFF, (radio_id >> 24) & 0xFF, 0x8C])
        command1 = bytes([0x50, 0x11, 0xF4, 0x01, 0x00, 0x00, 0x67, 0xFF, 0xFF, 0xFF])
        self._write_radio(command0)
        self._write_radio(command1)
        self._write_radio(b"\x61\x12")
        self._write_radio(b"\x71\x85")
        self._write_radio(b"\x81\x86")

    def read_i2c(self, address: int, register_index: int) -> int:
        command = bytearray([0x86, address & 0xFF, (register_index >> 8) & 0xFF, register_index & 0xFF, 0x00, 0x00])
        self._set_get_cur(self.I2C, command)
        if command[0] != 0x86 or command[4] != 0x00 or command[5] != 0x00:
            raise RuntimeError("ESP770U::read_i2c: invalid response")
        return (command[2] << 8) | command[1]

    def write_i2c(self, address: int, register_index: int, value: int) -> None:
        command = bytearray([0x06, address & 0xFF, (register_index >> 8) & 0xFF, register_index & 0xFF, (value >> 8) & 0xFF, value & 0xFF])
        self._set_get_cur(self.I2C, command)
        if command[0] != 0x06 or command[1] != (address & 0xFF) or command[2] != (register_index >> 8) & 0xFF or command[3] != register_index & 0xFF or command[4] != (value >> 8) & 0xFF or command[5] != value & 0xFF:
            raise RuntimeError("ESP770U::write_i2c: invalid response")


class AR0134:
    """Imaging sensor access via I2C through the controller."""

    I2C_ADDRESS = 0x20

    CHIP_VERSION_REG = 0x3000
    Y_ADDR_START = 0x3002
    X_ADDR_START = 0x3004
    Y_ADDR_END = 0x3006
    X_ADDR_END = 0x3008
    FRAME_LENGTH_LINES = 0x300A
    LINE_LENGTH_PCK = 0x300C
    REVISION_NUMBER = 0x300E
    COARSE_INTEGRATION_TIME = 0x3012
    FINE_INTEGRATION_TIME = 0x3014
    RESET_REGISTER = 0x301A
    DATA_PEDESTAL = 0x301E
    GPI_STATUS = 0x3026
    DIGITAL_BINNING = 0x3032
    FRAME_COUNT = 0x303A
    FRAME_STATUS = 0x303C
    READ_MODE = 0x3040
    DARK_CONTROL = 0x3044
    FLASH = 0x3046
    GLOBAL_GAIN = 0x305E
    EMBEDDED_DATA_CONTROL = 0x3064
    DATAPATH_SELECT = 0x306E
    TEST_PATTERN_MODE = 0x3070
    DIGITAL_TEST = 0x30B0
    TEMPSENS_DATA = 0x30B2
    COLUMN_CORRECTION = 0x30D4
    AE_CTRL_REG = 0x3100
    AE_LUMA_TARGET_REG = 0x3102
    AE_MIN_EV_STEP_REG = 0x3108
    AE_MAX_EV_STEP_REG = 0x310A
    AE_MAX_EXPOSURE_REG = 0x311C
    AE_MIN_EXPOSURE_REG = 0x311E

    RESET = 0x0001
    RESTART = 0x0002
    STREAM = 0x0004
    LOCK_REG = 0x0008
    STDBY_EOF = 0x0010
    DRIVE_PINS = 0x0040
    PARALLEL_EN = 0x0080
    GPI_EN = 0x0100
    MASK_BAD = 0x0200
    RESTART_BAD = 0x0400
    FORCED_PLL_ON = 0x0800
    SMIA_SERIALISER_DIS = 0x1000
    GROUPED_PARAMETER_HOLD = 0x8000

    HORIZ_MIRROR = 0x4000
    VERT_FLIP = 0x8000

    EMBEDDED_STATS_EN = 0x0080
    EMBEDDED_DATA = 0x0100

    COL_GAIN_MASK = 0x0030
    MONO_CHROME = 0x0080
    COL_GAIN_CB_MASK = 0x0300
    ENABLE_SHORT_LLPCK = 0x0400
    CONTEXT_B = 0x2000
    PLL_COMPLETE_BYPASS = 0x4000

    AE_ENABLE = 0x0001
    AUTO_AG_EN = 0x0002
    AUTO_DG_EN = 0x0010
    MIN_ANA_GAIN_MASK = 0x0060

    def __init__(self, controller: ESP770U) -> None:
        self.controller = controller

    def read_register(self, register_index: int) -> int:
        return self.controller.read_i2c(self.I2C_ADDRESS, register_index)

    def write_register(self, register_index: int, value: int) -> None:
        self.controller.write_i2c(self.I2C_ADDRESS, register_index, value)

    def init(self) -> None:
        time.sleep(0.1)
        version = self.read_register(self.CHIP_VERSION_REG)
        revision = self.read_register(self.REVISION_NUMBER)
        if version != 0x2406 or revision != 0x1300:
            raise RuntimeError(f"AR0134::init: unsupported chip version {version}.{revision}")
        test_mode = self.read_register(self.DIGITAL_TEST)
        if test_mode != self.MONO_CHROME:
            print(f"AR0134::init: unexpected camera mode 0x{test_mode:04x}, expected 0x{self.MONO_CHROME:04x}; continuing anyway")
        edc = self.read_register(self.EMBEDDED_DATA_CONTROL)
        self.write_register(self.EMBEDDED_DATA_CONTROL, edc | self.EMBEDDED_STATS_EN | self.EMBEDDED_DATA)

    def get_horizontal_flip(self) -> bool:
        return bool(self.read_register(self.READ_MODE) & self.HORIZ_MIRROR)

    def get_vertical_flip(self) -> bool:
        return bool(self.read_register(self.READ_MODE) & self.VERT_FLIP)

    def set_flip(self, horizontal: bool, vertical: bool) -> None:
        read_mode = self.read_register(self.READ_MODE)
        if horizontal:
            read_mode |= self.HORIZ_MIRROR
        else:
            read_mode &= ~self.HORIZ_MIRROR
        if vertical:
            read_mode |= self.VERT_FLIP
        else:
            read_mode &= ~self.VERT_FLIP
        self.write_register(self.READ_MODE, read_mode)

    def get_auto_exposure(self) -> bool:
        return bool(self.read_register(self.AE_CTRL_REG) & self.AE_ENABLE)

    def set_auto_exposure(self, enable: bool, adjust_analog_gain: bool, adjust_digital_gain: bool) -> None:
        ae = self.read_register(self.AE_CTRL_REG)
        ae &= ~(self.AE_ENABLE | self.AUTO_AG_EN | self.AUTO_DG_EN)
        if enable:
            ae |= self.AE_ENABLE
        if adjust_analog_gain:
            ae |= self.AUTO_AG_EN
        if adjust_digital_gain:
            ae |= self.AUTO_DG_EN
        self.write_register(self.AE_CTRL_REG, ae)

    def get_gain(self) -> int:
        return self.read_register(self.GLOBAL_GAIN)

    def set_gain(self, gain: int) -> None:
        self.write_register(self.GLOBAL_GAIN, gain & 0xFFFF)

    def set_window(self, x: int, y: int, width: int, height: int) -> None:
        self.write_register(self.Y_ADDR_START, y)
        self.write_register(self.X_ADDR_START, x)
        self.write_register(self.Y_ADDR_END, y + height - 1)
        self.write_register(self.X_ADDR_END, x + width - 1)

    def get_total_width(self) -> int:
        return self.read_register(self.LINE_LENGTH_PCK)

    def get_total_height(self) -> int:
        return self.read_register(self.FRAME_LENGTH_LINES)

    def set_frame_timings(self, min_blank: bool) -> None:
        self.set_window(0, 0, 1280, 960)
        self.write_register(self.LINE_LENGTH_PCK, 1280 + (108 if min_blank else 218))
        dt = self.read_register(self.DIGITAL_TEST)
        if min_blank:
            dt |= self.ENABLE_SHORT_LLPCK
        else:
            dt &= ~self.ENABLE_SHORT_LLPCK
        self.write_register(self.DIGITAL_TEST, dt)
        self.write_register(self.FRAME_LENGTH_LINES, 960 + (23 if min_blank else 37))

    def get_coarse_exposure_time(self) -> int:
        return self.read_register(self.COARSE_INTEGRATION_TIME)

    def set_coarse_exposure_time(self, coarse: int) -> None:
        self.write_register(self.COARSE_INTEGRATION_TIME, coarse & 0xFFFF)

    def get_fine_exposure_time(self) -> int:
        return self.read_register(self.FINE_INTEGRATION_TIME)

    def set_fine_exposure_time(self, fine: int) -> None:
        self.write_register(self.FINE_INTEGRATION_TIME, fine & 0xFFFF)

    def get_exposure_time(self) -> int:
        frame_width = self.read_register(self.LINE_LENGTH_PCK)
        coarse = self.read_register(self.COARSE_INTEGRATION_TIME)
        fine = self.read_register(self.FINE_INTEGRATION_TIME)
        return coarse * frame_width + fine

    def set_exposure_time(self, exposure: int) -> None:
        frame_width = self.read_register(self.LINE_LENGTH_PCK)
        self.write_register(self.COARSE_INTEGRATION_TIME, (exposure // frame_width) & 0xFFFF)
        self.write_register(self.FINE_INTEGRATION_TIME, (exposure % frame_width) & 0xFFFF)

    def set_sync(self, enable: bool) -> None:
        r = self.read_register(self.RESET_REGISTER)
        r &= ~(self.STREAM | self.GPI_EN | self.FORCED_PLL_ON)
        if enable:
            r |= self.GPI_EN | self.FORCED_PLL_ON
        else:
            r |= self.STREAM
        self.write_register(self.RESET_REGISTER, r)


class OculusRiftCV1Camera:
    """High-level camera interface matching the standalone C++ API."""

    VENDOR_ID = 0x2833
    PRODUCT_ID = 0x0211

    def __init__(self, device_index: int = 0, context: Optional[usb1.USBContext] = None) -> None:
        self.context = context or usb1.USBContext()
        self.handle = self._open_device(device_index)
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

    # ---- public API -------------------------------------------------
    def get_frame_size(self) -> tuple[int, int]:
        return self.frame_size

    def get_frame_width(self) -> int:
        return self.frame_size[0]

    def get_frame_height(self) -> int:
        return self.frame_size[1]

    def can_undistort(self, pixel: tuple[float, float]) -> bool:
        center = (655.052, 475.083)
        dx = pixel[0] - center[0]
        dy = pixel[1] - center[1]
        return (dx * dx + dy * dy) < self.max_r2

    def undistort(self, pixel: tuple[float, float]) -> tuple[float, float]:
        """Convert a distorted pixel coordinate to undistorted coordinate.
        
        Args:
            pixel: (x, y) coordinate in distorted (raw camera) space
            
        Returns:
            (x, y) coordinate in undistorted (corrected) space
        """
        center = (655.052, 475.083)
        kappas = (5.16403e-07, 2.44492e-13, 6.881e-19)
        rhos = (-8.66716e-07, 8.37108e-07)
        dx = pixel[0] - center[0]
        dy = pixel[1] - center[1]
        r2 = dx * dx + dy * dy
        radial = 0.0
        for kappa in reversed(kappas):
            radial = (radial + kappa) * r2
        radial += 1.0
        return (
            center[0] + dx * radial + 2.0 * rhos[0] * dx * dy + rhos[1] * (r2 + 2.0 * dx * dx),
            center[1] + dy * radial + rhos[0] * (r2 + 2.0 * dy * dy) + 2.0 * rhos[1] * dx * dy,
        )

    def distort(self, pixel: tuple[float, float], max_iterations: int = 20, tolerance: float = 1e-6) -> tuple[float, float]:
        """Convert an undistorted pixel coordinate to distorted coordinate (inverse of undistort).
        
        This is needed for creating remap tables with OpenCV, where for each output (undistorted)
        pixel we need to find which input (distorted) pixel to sample from.
        
        Args:
            pixel: (x, y) coordinate in undistorted (corrected) space
            max_iterations: Maximum number of Newton-Raphson iterations
            tolerance: Convergence tolerance in pixels
            
        Returns:
            (x, y) coordinate in distorted (raw camera) space
        """
        # Use fixed-point iteration to find the distorted point
        # Start with the undistorted point as initial guess
        distorted_x, distorted_y = pixel
        
        for _ in range(max_iterations):
            # Compute undistorted position from current distorted guess
            undistorted_x, undistorted_y = self.undistort((distorted_x, distorted_y))
            
            # Compute error
            error_x = undistorted_x - pixel[0]
            error_y = undistorted_y - pixel[1]
            
            # Check for convergence
            error = math.sqrt(error_x * error_x + error_y * error_y)
            if error < tolerance:
                break
            
            # Update distorted position (simple fixed-point iteration)
            # This works because the distortion is small
            distorted_x -= error_x
            distorted_y -= error_y
        
        # Clamp to valid image bounds to prevent out-of-bounds sampling
        distorted_x = max(0.0, min(float(self.frame_size[0] - 1), distorted_x))
        distorted_y = max(0.0, min(float(self.frame_size[1] - 1), distorted_y))
        
        return (distorted_x, distorted_y)

    def start_streaming(self, callback: StreamingCallback) -> None:
        if self.streaming:
            raise RuntimeError("Already streaming")
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
        uvc_probe[2] = 1     # bFormatIndex
        uvc_probe[3] = 4     # bFrameIndex
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
        uvc_probe_result = bytearray(UVC.get_cur(self.handle, 1, 0, 1, len(uvc_probe)))
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

    def stop_streaming(self) -> None:
        if not self.streaming:
            return
        self.streaming = False
        self._stop_event.set()
        for transfer in self._transfers:
            try:
                transfer.cancel()
            except usb1.USBError:
                pass
        # Give libusb a moment to process cancellations
        time.sleep(0.1)
        self._transfers.clear()
        self._transfer_buffers.clear()
        try:
            self.handle.releaseInterface(1)
        except usb1.USBError:
            pass
        self.streaming_callback = None

    def get_horizontal_flip(self) -> bool:
        sensor = AR0134(ESP770U(self.handle))
        return sensor.get_horizontal_flip()

    def get_vertical_flip(self) -> bool:
        sensor = AR0134(ESP770U(self.handle))
        return sensor.get_vertical_flip()

    def set_flip(self, horizontal: bool, vertical: bool) -> None:
        sensor = AR0134(ESP770U(self.handle))
        sensor.set_flip(horizontal, vertical)

    def get_auto_exposure(self) -> bool:
        sensor = AR0134(ESP770U(self.handle))
        return sensor.get_auto_exposure()

    def set_auto_exposure(self, enable: bool, adjust_analog_gain: bool = False, adjust_digital_gain: bool = False) -> None:
        sensor = AR0134(ESP770U(self.handle))
        sensor.set_auto_exposure(enable, adjust_analog_gain, adjust_digital_gain)

    def get_gain(self) -> int:
        sensor = AR0134(ESP770U(self.handle))
        return sensor.get_gain()

    def set_gain(self, gain: int) -> None:
        sensor = AR0134(ESP770U(self.handle))
        sensor.set_gain(gain)

    def get_exposure_time(self) -> int:
        sensor = AR0134(ESP770U(self.handle))
        return sensor.get_exposure_time()

    def set_exposure_time(self, exposure: int) -> None:
        sensor = AR0134(ESP770U(self.handle))
        sensor.set_exposure_time(exposure)

    # ---- internal helpers -------------------------------------------
    def _open_device(self, device_index: int) -> usb1.USBDeviceHandle:
        match_count = 0
        handle: Optional[usb1.USBDeviceHandle] = None
        for device in self.context.getDeviceList(skip_on_error=True):
            if device.getVendorID() == self.VENDOR_ID and device.getProductID() == self.PRODUCT_ID:
                if match_count == device_index:
                    handle = device.open()
                    break
                match_count += 1
        if handle is None:
            raise RuntimeError(f"OculusRiftCV1Camera: fewer than {device_index + 1} cameras detected")
        return handle

    def _setup_device(self) -> None:
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
                    print("Error: Failed to query firmware version after 5 attempts")
                    raise
                print(f"Attempt {attempt + 1} failed: {exc}")
        if firmware_version is not None:
            print(f"Firmware version: {firmware_version}")
        controller.init_controller()
        controller.init_radio()
        cal_data = controller.read_memory(0x1D000, 128)
        def _read_float(offset: int) -> float:
            val = cal_data[offset : offset + 4]
            return struct.unpack_from('<f', val)[0]
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
        print(f"  Lens distortion: {self.k[0]}, {self.k[1]}, {self.k[2]}, {self.k[3]}")
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

    def handle_events(self, timeout: float = 0.01) -> None:
        """Process pending USB events. Call this periodically when streaming."""
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
                iso_transfer_length_list=[packet_size] * packets
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
            self._frame_buffer[self._frame_ptr : self._frame_ptr + len(payload)] = payload
            self._frame_ptr += len(payload)
            self._frame_remainder -= len(payload)
            if self._frame_remainder == 0 and self.streaming_callback:
                finished = bytes(self._frame_buffer)
                self._frame_buffer = bytearray(self.frame_size[0] * self.frame_size[1])
                self.streaming_callback(finished)
        else:
            self._frame_remainder = 0

    def close(self) -> None:
        if self.streaming:
            self.stop_streaming()
        try:
            self.handle.releaseInterface(0)
        except usb1.USBError:
            pass
        self.handle.close()
        self.context = None  # type: ignore


__all__ = [
    "OculusRiftCV1Camera",
    "StreamingCallback",
]
