"""Oculus Rift CV1 camera driver library.

This package provides a Python interface to Oculus Rift CV1 tracking cameras
using libusb-1.0 through the usb1 Python bindings. It supports low-level USB
control, camera streaming, lens undistortion, and frame buffering.

Main Components:
    - OculusRiftCV1Camera: High-level camera interface
    - TripleBuffer: Thread-safe triple buffering for frame data
    - UVC: UVC control transfer helpers
    - ESP770U: Camera controller interface
    - AR0134: Imaging sensor interface

Example:
    >>> from chunivision.oculus import OculusRiftCV1Camera
    >>> def on_frame(frame_bytes):
    ...     print(f"Received frame: {len(frame_bytes)} bytes")
    >>>
    >>> with OculusRiftCV1Camera() as camera:
    ...     camera.start_streaming(on_frame)
    ...     # Process frames...
    ...     camera.stop_streaming()
"""

__version__ = "0.1.0"
__author__ = "Misaka 19465"

from .ar0134 import AR0134
from .esp770u import ESP770U
from .oculus_camera import OculusRiftCV1Camera
from .oculus_camera import list_oculus_cameras
from .uvc import UVC
from .exceptions import (
    CalibrationError,
    CommunicationError,
    DeviceInitializationError,
    DeviceNotFoundError,
    InvalidParameterError,
    OculusError,
    StreamingError,
)
from .triple_buffer import TripleBuffer

__all__ = [
    # Main camera interface
    "OculusRiftCV1Camera",
    # Low-level interfaces
    "UVC",
    "ESP770U",
    "AR0134",
    # Utilities
    "TripleBuffer",
    # Exceptions
    "OculusError",
    "DeviceNotFoundError",
    "DeviceInitializationError",
    "CommunicationError",
    "InvalidParameterError",
    "StreamingError",
    "CalibrationError",
    "list_oculus_cameras",
]
