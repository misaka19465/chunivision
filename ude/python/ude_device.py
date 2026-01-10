"""
Main Python interface for ChunithmUDE driver.

Provides high-level API for sending Chunithm controller state
to the UDE kernel driver.

Author: Misaka 19465
"""

import ctypes
from ctypes import wintypes
import struct
from typing import List, Optional, Dict, Any

from .constants import (
    CHUNITHM_INPUT_REPORT_SIZE,
    NUM_TOUCH_ZONES,
    NUM_AIR_SENSORS,
    TOUCH_VALUE_RELEASED,
    TOUCH_VALUE_PRESSED,
    DEFAULT_DEVICE_PATH,
    IOCTL_CHUNITHM_SEND_REPORT,
    IOCTL_CHUNITHM_GET_STATUS,
    IOCTL_CHUNITHM_RESET,
    GENERIC_READ,
    GENERIC_WRITE,
    OPEN_EXISTING,
    INVALID_HANDLE_VALUE,
)
from .exceptions import (
    DeviceNotFoundError,
    DeviceNotReadyError,
    InvalidReportError,
    DriverCommunicationError,
)


class ChunithmUDEDevice:
    """
    Interface to ChunithmUDE kernel driver.
    
    Provides methods to send controller state (touch zones and air sensors)
    to the emulated Chunithm USB device.
    
    Example:
        >>> device = ChunithmUDEDevice()
        >>> touch_zones = [False] * 32
        >>> air_sensors = [False] * 6
        >>> touch_zones[0] = True  # Zone 1
        >>> device.send_report(touch_zones, air_sensors)
        >>> device.close()
    
    Attributes:
        device_path: Path to device interface
        handle: Windows handle to device
    """
    
    def __init__(self, device_path: str = DEFAULT_DEVICE_PATH):
        """
        Initialize connection to UDE driver.
        
        Args:
            device_path: Device interface path (default: r"\\.\ChunithmController")
        
        Raises:
            DeviceNotFoundError: If device cannot be opened
        """
        self.device_path = device_path
        self.handle: Optional[int] = None
        self._kernel32 = ctypes.windll.kernel32
        
        self._open_device()
    
    def _open_device(self) -> None:
        """
        Open handle to UDE driver device.
        
        Raises:
            DeviceNotFoundError: If CreateFile fails
        """
        self.handle = self._kernel32.CreateFileW(
            self.device_path,
            GENERIC_READ | GENERIC_WRITE,
            0,  # No sharing
            None,  # Default security
            OPEN_EXISTING,
            0,  # No special flags
            None  # No template
        )
        
        if self.handle == INVALID_HANDLE_VALUE:
            error_code = ctypes.get_last_error()
            self.handle = None
            raise DeviceNotFoundError(self.device_path, error_code)
    
    def send_report(
        self,
        touch_zones: List[bool],
        air_sensors: List[bool]
    ) -> bool:
        """
        Send controller state to UDE driver.
        
        Args:
            touch_zones: List of 32 boolean values (zone 1-32)
            air_sensors: List of 6 boolean values (air 0-5)
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            DeviceNotReadyError: If device not opened
            InvalidReportError: If input data is invalid
            DriverCommunicationError: If IOCTL fails
        """
        if self.handle is None:
            raise DeviceNotReadyError("Device not opened")
        
        # Validate input
        if len(touch_zones) != NUM_TOUCH_ZONES:
            raise InvalidReportError(
                f"Expected {NUM_TOUCH_ZONES} touch zones, got {len(touch_zones)}"
            )
        
        if len(air_sensors) != NUM_AIR_SENSORS:
            raise InvalidReportError(
                f"Expected {NUM_AIR_SENSORS} air sensors, got {len(air_sensors)}"
            )
        
        # Build HID report
        report = self._build_report(touch_zones, air_sensors)
        
        # Send via IOCTL
        return self._send_ioctl(IOCTL_CHUNITHM_SEND_REPORT, report)
    
    def _build_report(
        self,
        touch_zones: List[bool],
        air_sensors: List[bool]
    ) -> bytes:
        """
        Build 45-byte HID input report.
        
        Args:
            touch_zones: Touch zone states
            air_sensors: Air sensor states
        
        Returns:
            45-byte packed report
        """
        # Build IR value (6 air sensors in low 6 bits)
        ir_value = 0
        for i in range(NUM_AIR_SENSORS):
            if air_sensors[i]:
                ir_value |= (1 << i)
        
        # Build touch values (0x00 = released, 0x64 = pressed)
        touch_values = [
            TOUCH_VALUE_PRESSED if touch_zones[i] else TOUCH_VALUE_RELEASED
            for i in range(NUM_TOUCH_ZONES)
        ]
        
        # Pack report structure
        # Format: BB32B11B = 1 + 1 + 32 + 1 + 10 = 45 bytes
        report = struct.pack(
            'BB32B11B',
            ir_value,           # IRValue (1 byte)
            0x00,               # Buttons (1 byte) - not used
            *touch_values,      # TouchValue[32] (32 bytes)
            0,                  # CardStatus (1 byte) - always 0
            *([0] * 10)         # CardID[10] (10 bytes) - all zeros
        )
        
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get driver status information.
        
        Returns:
            Dictionary with status fields:
                - device_ready: bool
                - reports_sent: int
                - error_count: int
        
        Raises:
            DeviceNotReadyError: If device not opened
            DriverCommunicationError: If IOCTL fails
        """
        if self.handle is None:
            raise DeviceNotReadyError("Device not opened")
        
        # Prepare output buffer for status structure
        buffer = ctypes.create_string_buffer(12)  # 3 DWORDs = 12 bytes
        bytes_returned = wintypes.DWORD()
        
        success = self._kernel32.DeviceIoControl(
            self.handle,
            IOCTL_CHUNITHM_GET_STATUS,
            None,  # No input
            0,
            buffer,
            len(buffer),
            ctypes.byref(bytes_returned),
            None
        )
        
        if not success:
            error_code = ctypes.get_last_error()
            raise DriverCommunicationError(IOCTL_CHUNITHM_GET_STATUS, error_code)
        
        # Unpack status structure
        device_ready, reports_sent, error_count = struct.unpack('III', buffer.raw[:12])
        
        return {
            'device_ready': bool(device_ready),
            'reports_sent': reports_sent,
            'error_count': error_count
        }
    
    def reset(self) -> bool:
        """
        Reset device state.
        
        Returns:
            True if successful
        
        Raises:
            DeviceNotReadyError: If device not opened
            DriverCommunicationError: If IOCTL fails
        """
        if self.handle is None:
            raise DeviceNotReadyError("Device not opened")
        
        return self._send_ioctl(IOCTL_CHUNITHM_RESET, None)
    
    def _send_ioctl(
        self,
        ioctl_code: int,
        input_buffer: Optional[bytes]
    ) -> bool:
        """
        Send IOCTL to driver.
        
        Args:
            ioctl_code: IOCTL control code
            input_buffer: Input data (or None)
        
        Returns:
            True if successful
        
        Raises:
            DriverCommunicationError: If DeviceIoControl fails
        """
        bytes_returned = wintypes.DWORD()
        
        success = self._kernel32.DeviceIoControl(
            self.handle,
            ioctl_code,
            input_buffer if input_buffer else None,
            len(input_buffer) if input_buffer else 0,
            None,  # No output
            0,
            ctypes.byref(bytes_returned),
            None
        )
        
        if not success:
            error_code = ctypes.get_last_error()
            raise DriverCommunicationError(ioctl_code, error_code)
        
        return True
    
    def close(self) -> None:
        """Close device handle and cleanup resources."""
        if self.handle is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor - ensure handle is closed."""
        self.close()
