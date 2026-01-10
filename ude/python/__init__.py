"""
ChunithmUDE Python Bindings

Provides Python interface to ChunithmUDE kernel driver for
Chunithm controller emulation via Windows USB Device Emulation.

Author: Misaka 19465
Platform: Windows 10/11 only

Example:
    >>> from ude.python import ChunithmUDEDevice
    >>> device = ChunithmUDEDevice()
    >>> device.send_report([False]*32, [False]*6)
    >>> device.close()
"""

from .ude_device import ChunithmUDEDevice
from .exceptions import (
    ChunithmUDEError,
    DeviceNotFoundError,
    DeviceNotReadyError,
    InvalidReportError,
    DriverCommunicationError
)
from .constants import (
    CHUNITHM_INPUT_REPORT_SIZE,
    CHUNITHM_OUTPUT_REPORT_SIZE,
    IOCTL_CHUNITHM_SEND_REPORT,
    IOCTL_CHUNITHM_GET_STATUS,
    IOCTL_CHUNITHM_RESET
)

__version__ = "1.0.0"
__author__ = "Misaka 19465"

__all__ = [
    "ChunithmUDEDevice",
    "ChunithmUDEError",
    "DeviceNotFoundError",
    "DeviceNotReadyError",
    "InvalidReportError",
    "DriverCommunicationError",
    "CHUNITHM_INPUT_REPORT_SIZE",
    "CHUNITHM_OUTPUT_REPORT_SIZE",
    "IOCTL_CHUNITHM_SEND_REPORT",
    "IOCTL_CHUNITHM_GET_STATUS",
    "IOCTL_CHUNITHM_RESET",
]
