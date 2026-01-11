"""Custom exceptions for oculus camera operations."""


class OculusError(Exception):
    """Base exception for all Oculus camera errors."""


class DeviceNotFoundError(OculusError):
    """Raised when the requested camera device cannot be found."""


class DeviceInitializationError(OculusError):
    """Raised when camera initialization fails."""


class CommunicationError(OculusError):
    """Raised when USB communication with the camera fails."""


class InvalidParameterError(OculusError):
    """Raised when invalid parameters are provided."""


class StreamingError(OculusError):
    """Raised when streaming operations fail."""


class CalibrationError(OculusError):
    """Raised when calibration data is invalid or cannot be read."""
