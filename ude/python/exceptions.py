"""
Custom exceptions for ChunithmUDE driver communication.

Author: Misaka 19465
"""


class ChunithmUDEError(Exception):
    """Base exception for ChunithmUDE errors."""
    pass


class DeviceNotFoundError(ChunithmUDEError):
    """Raised when UDE driver device cannot be found or opened."""
    
    def __init__(self, device_path: str, error_code: int = 0):
        self.device_path = device_path
        self.error_code = error_code
        message = f"Cannot open ChunithmUDE device '{device_path}'"
        if error_code:
            message += f" (error code: {error_code})"
        super().__init__(message)


class DeviceNotReadyError(ChunithmUDEError):
    """Raised when device exists but is not ready for operation."""
    
    def __init__(self, message: str = "Device is not ready"):
        super().__init__(message)


class InvalidReportError(ChunithmUDEError):
    """Raised when HID report data is invalid."""
    
    def __init__(self, message: str):
        super().__init__(f"Invalid report data: {message}")


class DriverCommunicationError(ChunithmUDEError):
    """Raised when communication with driver fails."""
    
    def __init__(self, ioctl_code: int, error_code: int):
        self.ioctl_code = ioctl_code
        self.error_code = error_code
        message = f"Driver IOCTL 0x{ioctl_code:08X} failed (error: {error_code})"
        super().__init__(message)
