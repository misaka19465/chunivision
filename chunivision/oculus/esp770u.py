"""ESP770U camera controller access via extension unit controls."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    import usb1
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "usb1 (libusb1 Python bindings) is required. Install with 'pip install libusb1'."
    ) from exc

from ..utils.logger import Logger
from .exceptions import CommunicationError, InvalidParameterError
from .uvc import UVC

if TYPE_CHECKING:
    pass

logger = Logger.get_logger(__name__)


class ESP770U:
    """Camera controller access via extension unit controls.

    This class provides low-level access to the ESP770U camera controller
    through USB extension unit controls. It supports register access, I2C
    communication, SPI operations, and radio control.
    """

    EXTENSION_UNIT = 4
    I2C = 2
    REGISTER = 3
    COUNTER = 10
    CONTROL = 11
    DATA = 12

    def __init__(self, handle: usb1.USBDeviceHandle) -> None:
        """Initialize ESP770U controller interface.

        Args:
            handle: USB device handle

        Raises:
            InvalidParameterError: If handle is None
        """
        if handle is None:
            raise InvalidParameterError("USB device handle cannot be None")
        self.handle = handle
        logger.debug("ESP770U controller initialized")

    def _set_get_cur(self, selector: int, buffer: bytearray) -> None:
        """Execute set/get current sequence with retry logic.

        Args:
            selector: Selector value (must be >= 0)
            buffer: Buffer to send and receive (must not be empty)

        Raises:
            InvalidParameterError: If parameters are invalid
            CommunicationError: If operation fails after retries
        """
        if selector < 0:
            raise InvalidParameterError(f"Selector must be >= 0, got {selector}")
        if not buffer:
            raise InvalidParameterError("Buffer cannot be empty")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"ESP770U._set_get_cur: selector={selector}, "
                    f"attempt={attempt + 1}/{max_retries}"
                )
                UVC.set_cur(
                    self.handle, 0, self.EXTENSION_UNIT, selector, bytes(buffer)
                )
                time.sleep(0.1)
                data = UVC.get_cur(
                    self.handle, 0, self.EXTENSION_UNIT, selector, len(buffer)
                )
                buffer[:] = data
                return
            except (usb1.USBError, CommunicationError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"ESP770U._set_get_cur failed after {max_retries} attempts: {e}"
                    )
                    raise CommunicationError(
                        f"Failed to execute set/get current after {max_retries} attempts"
                    ) from e
                logger.warning(
                    f"ESP770U._set_get_cur attempt {attempt + 1} failed: {e}"
                )
                time.sleep(0.3 * (attempt + 1))

    def read_register(self, register_index: int) -> int:
        """Read a register value from the camera controller.

        Args:
            register_index: Register address (must be in range 0x0000-0xFFFF)

        Returns:
            Register value (0-255)

        Raises:
            InvalidParameterError: If register_index is out of range
            CommunicationError: If read operation fails or returns invalid response
        """
        if not (0x0000 <= register_index <= 0xFFFF):
            raise InvalidParameterError(
                f"Register index must be in range 0x0000-0xFFFF, got 0x{register_index:04X}"
            )

        logger.debug(f"ESP770U.read_register: 0x{register_index:04X}")

        try:
            command = bytearray(
                [0x82, (register_index >> 8) & 0xFF, register_index & 0xFF, 0x00]
            )
            self._set_get_cur(self.REGISTER, command)
            if command[0] != 0x82 or command[2] != 0x00:
                logger.error(f"ESP770U.read_register invalid response: {command.hex()}")
                raise CommunicationError(
                    f"ESP770U::read_register: invalid response for register 0x{register_index:04X}"
                )
            return command[1]
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"ESP770U.read_register failed: {e}")
            raise CommunicationError(
                f"Failed to read register 0x{register_index:04X}"
            ) from e

    def write_register(self, register_index: int, value: int) -> None:
        """Write a value to a camera controller register.

        Args:
            register_index: Register address (must be in range 0x0000-0xFFFF)
            value: Value to write (must be in range 0-255)

        Raises:
            InvalidParameterError: If parameters are out of range
            CommunicationError: If write operation fails or returns invalid response
        """
        if not (0x0000 <= register_index <= 0xFFFF):
            raise InvalidParameterError(
                f"Register index must be in range 0x0000-0xFFFF, got 0x{register_index:04X}"
            )
        if not (0 <= value <= 255):
            raise InvalidParameterError(f"Value must be in range 0-255, got {value}")

        logger.debug(f"ESP770U.write_register: 0x{register_index:04X} = 0x{value:02X}")

        try:
            command = bytearray(
                [
                    0x02,
                    (register_index >> 8) & 0xFF,
                    register_index & 0xFF,
                    value & 0xFF,
                ]
            )
            self._set_get_cur(self.REGISTER, command)
            if (
                command[0] != 0x02
                or command[1] != (register_index >> 8) & 0xFF
                or command[2] != register_index & 0xFF
                or command[3] != value
            ):
                logger.error(
                    f"ESP770U.write_register invalid response: {command.hex()}"
                )
                raise CommunicationError(
                    f"ESP770U::write_register: invalid response for register 0x{register_index:04X}"
                )
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"ESP770U.write_register failed: {e}")
            raise CommunicationError(
                f"Failed to write register 0x{register_index:04X}"
            ) from e

    def get_counter(self) -> int:
        """Get current counter value.

        Returns:
            Counter value (0-255)

        Raises:
            CommunicationError: If read operation fails
        """
        logger.debug("ESP770U.get_counter")
        try:
            data = UVC.get_cur(self.handle, 0, self.EXTENSION_UNIT, self.COUNTER, 1)
            return data[0]
        except (usb1.USBError, CommunicationError) as e:
            logger.error(f"ESP770U.get_counter failed: {e}")
            raise CommunicationError("Failed to get counter") from e

    def set_counter(self, new_counter: int) -> None:
        """Set counter value.

        Args:
            new_counter: Counter value to set (must be in range 0-255)

        Raises:
            InvalidParameterError: If new_counter is out of range
            CommunicationError: If write operation fails
        """
        if not (0 <= new_counter <= 255):
            raise InvalidParameterError(
                f"Counter value must be in range 0-255, got {new_counter}"
            )

        logger.debug(f"ESP770U.set_counter: {new_counter}")
        try:
            UVC.set_cur(
                self.handle,
                0,
                self.EXTENSION_UNIT,
                self.COUNTER,
                bytes([new_counter & 0xFF]),
            )
        except (usb1.USBError, CommunicationError) as e:
            logger.error(f"ESP770U.set_counter failed: {e}")
            raise CommunicationError("Failed to set counter") from e

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
        """Write data to radio via SPI.

        Args:
            data: Data to write (must not be empty, max 126 bytes)

        Raises:
            InvalidParameterError: If data is invalid
            CommunicationError: If write operation fails
        """
        if not data:
            raise InvalidParameterError("Radio data cannot be empty")
        if len(data) > 126:
            raise InvalidParameterError(
                f"Radio data too large: {len(data)} bytes (max 126)"
            )
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
            logger.error(f"ESP770U._write_radio: invalid return buffer checksum")
            raise CommunicationError("ESP770U::write_radio: invalid return buffer")

    def query_firmware_version(self) -> int:
        """Query camera firmware version.

        Returns:
            Firmware version number

        Raises:
            CommunicationError: If query fails or returns invalid response
        """
        logger.debug("ESP770U.query_firmware_version")
        try:
            command = bytearray([0xA0, 0x03, 0x00, 0x00])
            self._set_get_cur(self.REGISTER, command)
            if command[0] != 0xA0 or command[2] != 0x00 or command[3] != 0x00:
                logger.error(
                    f"ESP770U.query_firmware_version invalid response: {command.hex()}"
                )
                raise CommunicationError(
                    "ESP770U::query_firmware_version: invalid response"
                )
            return command[1]
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"ESP770U.query_firmware_version failed: {e}")
            raise CommunicationError("Failed to query firmware version") from e

    def read_memory(self, address: int, length: int) -> bytes:
        """Read data from camera memory.

        Args:
            address: Memory address (must be in range 0x000000-0xFFFFFF)
            length: Number of bytes to read (must be > 0)

        Returns:
            Data read from memory

        Raises:
            InvalidParameterError: If parameters are out of range
            CommunicationError: If read operation fails
        """
        if not (0x000000 <= address <= 0xFFFFFF):
            raise InvalidParameterError(
                f"Memory address must be in range 0x000000-0xFFFFFF, got 0x{address:06X}"
            )
        if length <= 0:
            raise InvalidParameterError(f"Length must be > 0, got {length}")
        if length > 0xFFFF:
            raise InvalidParameterError(f"Length too large: {length} (max 65535)")

        logger.debug(f"ESP770U.read_memory: address=0x{address:06X}, length={length}")

        try:
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
            UVC.set_cur(
                self.handle, 0, self.EXTENSION_UNIT, self.CONTROL, bytes(command)
            )
            data = UVC.get_cur(self.handle, 0, self.EXTENSION_UNIT, self.DATA, length)
            self.set_counter(counter)
            return data
        except (InvalidParameterError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"ESP770U.read_memory failed: {e}")
            raise CommunicationError(f"Failed to read memory at 0x{address:06X}") from e

    def init_controller(self) -> None:
        """Initialize camera controller hardware.

        Performs hardware initialization sequence including register
        configuration and timing setup.

        Raises:
            CommunicationError: If initialization fails
        """
        logger.info("ESP770U.init_controller: starting controller initialization")
        try:
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
            logger.info("ESP770U.init_controller: controller initialized successfully")
        except (InvalidParameterError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"ESP770U.init_controller failed: {e}")
            raise CommunicationError("Failed to initialize controller") from e

    def init_radio(self) -> None:
        """Initialize radio communication hardware.

        Sets up radio for communication with external devices.

        Raises:
            CommunicationError: If initialization fails
        """
        logger.info("ESP770U.init_radio: starting radio initialization")
        try:
            time.sleep(0.05)
            for payload in (b"\x01\x01", b"\x11\x01"):
                self._write_radio(payload)
            value = self.read_register(0xF014)
            if value not in (0x1A, 0x1B):
                print(f"ESP770U::initRadio: unexpected 0x{value:02x} in 0xF014")
            self._write_radio(b"\x21\x01")
            self._write_radio(b"\x31\x01\x00")
            logger.info("ESP770U.init_radio: radio initialized successfully")
        except (InvalidParameterError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"ESP770U.init_radio failed: {e}")
            raise CommunicationError("Failed to initialize radio") from e

    def setup_radio(self, radio_id: int) -> None:
        """Configure radio with specific ID.

        Args:
            radio_id: Radio identifier (32-bit unsigned integer)

        Raises:
            InvalidParameterError: If radio_id is out of range
            CommunicationError: If setup fails
        """
        if not (0 <= radio_id <= 0xFFFFFFFF):
            raise InvalidParameterError(
                f"Radio ID must be in range 0-0xFFFFFFFF, got 0x{radio_id:08X}"
            )

        logger.info(f"ESP770U.setup_radio: radio_id=0x{radio_id:08X}")

        try:
            command0 = bytes(
                [
                    0x40,
                    0x10,
                    radio_id & 0xFF,
                    (radio_id >> 8) & 0xFF,
                    (radio_id >> 16) & 0xFF,
                    (radio_id >> 24) & 0xFF,
                    0x8C,
                ]
            )
            command1 = bytes(
                [0x50, 0x11, 0xF4, 0x01, 0x00, 0x00, 0x67, 0xFF, 0xFF, 0xFF]
            )
            self._write_radio(command0)
            self._write_radio(command1)
            self._write_radio(b"\x61\x12")
            self._write_radio(b"\x71\x85")
            self._write_radio(b"\x81\x86")
            logger.info("ESP770U.setup_radio: radio setup complete")
        except (InvalidParameterError, CommunicationError):
            raise
        except Exception as e:
            logger.error(f"ESP770U.setup_radio failed: {e}")
            raise CommunicationError("Failed to setup radio") from e

    def read_i2c(self, address: int, register_index: int) -> int:
        """Read value from I2C device.

        Args:
            address: I2C device address (must be in range 0-255)
            register_index: Register address (must be in range 0x0000-0xFFFF)

        Returns:
            16-bit register value

        Raises:
            InvalidParameterError: If parameters are out of range
            CommunicationError: If read operation fails or returns invalid response
        """
        if not (0 <= address <= 255):
            raise InvalidParameterError(
                f"I2C address must be in range 0-255, got {address}"
            )
        if not (0x0000 <= register_index <= 0xFFFF):
            raise InvalidParameterError(
                f"Register index must be in range 0x0000-0xFFFF, got 0x{register_index:04X}"
            )

        logger.debug(
            f"ESP770U.read_i2c: address=0x{address:02X}, register=0x{register_index:04X}"
        )

        try:
            command = bytearray(
                [
                    0x86,
                    address & 0xFF,
                    (register_index >> 8) & 0xFF,
                    register_index & 0xFF,
                    0x00,
                    0x00,
                ]
            )
            self._set_get_cur(self.I2C, command)
            if command[0] != 0x86 or command[4] != 0x00 or command[5] != 0x00:
                logger.error(f"ESP770U.read_i2c invalid response: {command.hex()}")
                raise CommunicationError("ESP770U::read_i2c: invalid response")
            return (command[2] << 8) | command[1]
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"ESP770U.read_i2c failed: {e}")
            raise CommunicationError(
                f"Failed to read I2C register 0x{register_index:04X} from device 0x{address:02X}"
            ) from e

    def write_i2c(self, address: int, register_index: int, value: int) -> None:
        """Write value to I2C device.

        Args:
            address: I2C device address (must be in range 0-255)
            register_index: Register address (must be in range 0x0000-0xFFFF)
            value: Value to write (must be in range 0x0000-0xFFFF)

        Raises:
            InvalidParameterError: If parameters are out of range
            CommunicationError: If write operation fails or returns invalid response
        """
        if not (0 <= address <= 255):
            raise InvalidParameterError(
                f"I2C address must be in range 0-255, got {address}"
            )
        if not (0x0000 <= register_index <= 0xFFFF):
            raise InvalidParameterError(
                f"Register index must be in range 0x0000-0xFFFF, got 0x{register_index:04X}"
            )
        if not (0x0000 <= value <= 0xFFFF):
            raise InvalidParameterError(
                f"Value must be in range 0x0000-0xFFFF, got 0x{value:04X}"
            )

        logger.debug(
            f"ESP770U.write_i2c: address=0x{address:02X}, "
            f"register=0x{register_index:04X}, value=0x{value:04X}"
        )

        try:
            command = bytearray(
                [
                    0x06,
                    address & 0xFF,
                    (register_index >> 8) & 0xFF,
                    register_index & 0xFF,
                    (value >> 8) & 0xFF,
                    value & 0xFF,
                ]
            )
            self._set_get_cur(self.I2C, command)
            if (
                command[0] != 0x06
                or command[1] != (address & 0xFF)
                or command[2] != (register_index >> 8) & 0xFF
                or command[3] != register_index & 0xFF
                or command[4] != (value >> 8) & 0xFF
                or command[5] != value & 0xFF
            ):
                logger.error(f"ESP770U.write_i2c invalid response: {command.hex()}")
                raise CommunicationError("ESP770U::write_i2c: invalid response")
        except CommunicationError:
            raise
        except Exception as e:
            logger.error(f"ESP770U.write_i2c failed: {e}")
            raise CommunicationError(
                f"Failed to write I2C register 0x{register_index:04X} on device 0x{address:02X}"
            ) from e


__all__ = ["ESP770U"]
