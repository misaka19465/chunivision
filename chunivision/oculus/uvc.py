"""USB Video Class (UVC) control transfer helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import usb1
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "usb1 (libusb1 Python bindings) is required. Install with 'pip install libusb1'."
    ) from exc

from ..utils.logger import Logger
from .exceptions import CommunicationError, InvalidParameterError

if TYPE_CHECKING:
    pass

logger = Logger.get_logger(__name__)


class UVC:
    """Helpers for UVC control transfers.

    This class provides static methods for USB Video Class (UVC) control
    transfers, including SET_CUR, GET_CUR, and GET_LEN operations.
    """

    @staticmethod
    def set_cur(
        handle: usb1.USBDeviceHandle,
        interface: int,
        entity: int,
        selector: int,
        data: bytes,
    ) -> None:
        """Set current value of a UVC control.

        Args:
            handle: USB device handle
            interface: Interface number (must be >= 0)
            entity: Entity ID (must be >= 0)
            selector: Selector value (must be >= 0)
            data: Data to send (must not be empty)

        Raises:
            InvalidParameterError: If parameters are invalid
            CommunicationError: If USB communication fails
        """
        # Validate inputs
        if handle is None:
            raise InvalidParameterError("USB device handle cannot be None")
        if interface < 0:
            raise InvalidParameterError(f"Interface must be >= 0, got {interface}")
        if entity < 0:
            raise InvalidParameterError(f"Entity must be >= 0, got {entity}")
        if selector < 0:
            raise InvalidParameterError(f"Selector must be >= 0, got {selector}")
        if not data:
            raise InvalidParameterError("Data cannot be empty")

        logger.debug(
            f"UVC.set_cur: interface={interface}, entity={entity}, "
            f"selector={selector}, data_len={len(data)}"
        )

        try:
            request_type = usb1.TYPE_CLASS | usb1.RECIPIENT_INTERFACE
            request = 0x01  # SET_CUR
            handle.controlWrite(
                request_type,
                request,
                selector << 8,
                (entity << 8) | interface,
                data,
                timeout=1000,
            )
        except usb1.USBError as e:
            logger.error(f"UVC.set_cur failed: {e}")
            raise CommunicationError(f"Failed to set UVC control: {e}") from e

    @staticmethod
    def get_cur(
        handle: usb1.USBDeviceHandle,
        interface: int,
        entity: int,
        selector: int,
        length: int,
    ) -> bytes:
        """Get current value of a UVC control.

        Args:
            handle: USB device handle
            interface: Interface number (must be >= 0)
            entity: Entity ID (must be >= 0)
            selector: Selector value (must be >= 0)
            length: Expected data length (must be > 0)

        Returns:
            Data received from the control

        Raises:
            InvalidParameterError: If parameters are invalid
            CommunicationError: If USB communication fails
        """
        # Validate inputs
        if handle is None:
            raise InvalidParameterError("USB device handle cannot be None")
        if interface < 0:
            raise InvalidParameterError(f"Interface must be >= 0, got {interface}")
        if entity < 0:
            raise InvalidParameterError(f"Entity must be >= 0, got {entity}")
        if selector < 0:
            raise InvalidParameterError(f"Selector must be >= 0, got {selector}")
        if length <= 0:
            raise InvalidParameterError(f"Length must be > 0, got {length}")

        logger.debug(
            f"UVC.get_cur: interface={interface}, entity={entity}, "
            f"selector={selector}, length={length}"
        )

        try:
            request_type = usb1.ENDPOINT_IN | usb1.TYPE_CLASS | usb1.RECIPIENT_INTERFACE
            request = 0x81  # GET_CUR
            return handle.controlRead(
                request_type,
                request,
                selector << 8,
                (entity << 8) | interface,
                length,
                timeout=1000,
            )
        except usb1.USBError as e:
            logger.error(f"UVC.get_cur failed: {e}")
            raise CommunicationError(f"Failed to get UVC control: {e}") from e

    @staticmethod
    def get_len(
        handle: usb1.USBDeviceHandle, interface: int, entity: int, selector: int
    ) -> int:
        """Get length of a UVC control.

        Args:
            handle: USB device handle
            interface: Interface number (must be >= 0)
            entity: Entity ID (must be >= 0)
            selector: Selector value (must be >= 0)

        Returns:
            Length of the control data

        Raises:
            InvalidParameterError: If parameters are invalid
            CommunicationError: If USB communication fails or response is invalid
        """
        # Validate inputs
        if handle is None:
            raise InvalidParameterError("USB device handle cannot be None")
        if interface < 0:
            raise InvalidParameterError(f"Interface must be >= 0, got {interface}")
        if entity < 0:
            raise InvalidParameterError(f"Entity must be >= 0, got {entity}")
        if selector < 0:
            raise InvalidParameterError(f"Selector must be >= 0, got {selector}")

        logger.debug(
            f"UVC.get_len: interface={interface}, entity={entity}, selector={selector}"
        )

        try:
            request_type = usb1.TYPE_CLASS | usb1.RECIPIENT_INTERFACE
            request = 0x85  # GET_LEN
            data = handle.controlRead(
                request_type,
                request,
                selector << 8,
                (entity << 8) | interface,
                2,
                timeout=1000,
            )
        except usb1.USBError as e:
            logger.error(f"UVC.get_len failed: {e}")
            raise CommunicationError(f"Failed to get UVC control length: {e}") from e

        if len(data) != 2:
            logger.error(f"UVC.get_len returned unexpected length: {len(data)}")
            raise CommunicationError(
                f"UVC::get_len returned unexpected length: {len(data)}"
            )
        return data[0] | (data[1] << 8)


__all__ = ["UVC"]
