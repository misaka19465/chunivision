"""AR0134 imaging sensor access via I2C."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..utils.logger import Logger
from .exceptions import (
    CommunicationError,
    DeviceInitializationError,
    InvalidParameterError,
)

if TYPE_CHECKING:
    from .esp770u import ESP770U

logger = Logger.get_logger(__name__)


class AR0134:
    """Imaging sensor access via I2C through the controller.

    This class provides access to the AR0134 image sensor through I2C
    communication. It supports configuration of exposure, gain, windowing,
    flip modes, and auto-exposure settings.
    """

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
        """Initialize AR0134 sensor interface.

        Args:
            controller: ESP770U controller instance

        Raises:
            InvalidParameterError: If controller is None
        """
        if controller is None:
            raise InvalidParameterError("Controller cannot be None")
        self.controller = controller
        logger.debug("AR0134 sensor interface initialized")

    def read_register(self, register_index: int) -> int:
        """Read sensor register value.

        Args:
            register_index: Register address (must be in range 0x0000-0xFFFF)

        Returns:
            16-bit register value

        Raises:
            InvalidParameterError: If register_index is out of range
            CommunicationError: If read operation fails
        """
        if not (0x0000 <= register_index <= 0xFFFF):
            raise InvalidParameterError(
                f"Register index must be in range 0x0000-0xFFFF, got 0x{register_index:04X}"
            )
        logger.debug(f"AR0134.read_register: 0x{register_index:04X}")
        return self.controller.read_i2c(self.I2C_ADDRESS, register_index)

    def write_register(self, register_index: int, value: int) -> None:
        """Write value to sensor register.

        Args:
            register_index: Register address (must be in range 0x0000-0xFFFF)
            value: Value to write (must be in range 0x0000-0xFFFF)

        Raises:
            InvalidParameterError: If parameters are out of range
            CommunicationError: If write operation fails
        """
        if not (0x0000 <= register_index <= 0xFFFF):
            raise InvalidParameterError(
                f"Register index must be in range 0x0000-0xFFFF, got 0x{register_index:04X}"
            )
        if not (0x0000 <= value <= 0xFFFF):
            raise InvalidParameterError(
                f"Value must be in range 0x0000-0xFFFF, got 0x{value:04X}"
            )
        logger.debug(f"AR0134.write_register: 0x{register_index:04X} = 0x{value:04X}")
        self.controller.write_i2c(self.I2C_ADDRESS, register_index, value)

    def init(self) -> None:
        """Initialize AR0134 sensor.

        Verifies chip version and configures embedded data.

        Raises:
            DeviceInitializationError: If sensor version is unsupported
            CommunicationError: If initialization fails
        """
        logger.info("AR0134.init: initializing sensor")
        try:
            time.sleep(0.1)
            version = self.read_register(self.CHIP_VERSION_REG)
            revision = self.read_register(self.REVISION_NUMBER)
            if version != 0x2406 or revision != 0x1300:
                error_msg = (
                    f"AR0134::init: unsupported chip version 0x{version:04X}.0x{revision:04X} "
                    f"(expected 0x2406.0x1300)"
                )
                logger.error(error_msg)
                raise DeviceInitializationError(error_msg)
            test_mode = self.read_register(self.DIGITAL_TEST)
            if test_mode != self.MONO_CHROME:
                logger.warning(
                    f"AR0134::init: unexpected camera mode 0x{test_mode:04x}, "
                    f"expected 0x{self.MONO_CHROME:04x}; continuing anyway"
                )
            edc = self.read_register(self.EMBEDDED_DATA_CONTROL)
            self.write_register(
                self.EMBEDDED_DATA_CONTROL,
                edc | self.EMBEDDED_STATS_EN | self.EMBEDDED_DATA,
            )
            logger.info("AR0134.init: sensor initialized successfully")
        except (InvalidParameterError, CommunicationError, DeviceInitializationError):
            raise
        except Exception as e:
            logger.error(f"AR0134.init failed: {e}")
            raise CommunicationError("Failed to initialize AR0134 sensor") from e

    def get_horizontal_flip(self) -> bool:
        """Get horizontal flip state.

        Returns:
            True if horizontal flip is enabled

        Raises:
            CommunicationError: If read operation fails
        """
        logger.debug("AR0134.get_horizontal_flip")
        return bool(self.read_register(self.READ_MODE) & self.HORIZ_MIRROR)

    def get_vertical_flip(self) -> bool:
        """Get vertical flip state.

        Returns:
            True if vertical flip is enabled

        Raises:
            CommunicationError: If read operation fails
        """
        logger.debug("AR0134.get_vertical_flip")
        return bool(self.read_register(self.READ_MODE) & self.VERT_FLIP)

    def set_flip(self, horizontal: bool, vertical: bool) -> None:
        """Set horizontal and vertical flip modes.

        Args:
            horizontal: Enable horizontal flip
            vertical: Enable vertical flip

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug(f"AR0134.set_flip: horizontal={horizontal}, vertical={vertical}")
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
        """Get auto-exposure enable state.

        Returns:
            True if auto-exposure is enabled

        Raises:
            CommunicationError: If read operation fails
        """
        logger.debug("AR0134.get_auto_exposure")
        return bool(self.read_register(self.AE_CTRL_REG) & self.AE_ENABLE)

    def set_auto_exposure(
        self, enable: bool, adjust_analog_gain: bool, adjust_digital_gain: bool
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
            f"AR0134.set_auto_exposure: enable={enable}, "
            f"analog_gain={adjust_analog_gain}, digital_gain={adjust_digital_gain}"
        )
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
        """Get current gain value.

        Returns:
            Current gain value (0-65535)

        Raises:
            CommunicationError: If read operation fails
        """
        logger.debug("AR0134.get_gain")
        return self.read_register(self.GLOBAL_GAIN)

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
        logger.debug(f"AR0134.set_gain: {gain}")
        self.write_register(self.GLOBAL_GAIN, gain & 0xFFFF)

    def set_window(self, x: int, y: int, width: int, height: int) -> None:
        """Set sensor readout window.

        Args:
            x: X coordinate (must be >= 0)
            y: Y coordinate (must be >= 0)
            width: Window width (must be > 0)
            height: Window height (must be > 0)

        Raises:
            InvalidParameterError: If parameters are out of range
            CommunicationError: If operation fails
        """
        if x < 0:
            raise InvalidParameterError(f"X coordinate must be >= 0, got {x}")
        if y < 0:
            raise InvalidParameterError(f"Y coordinate must be >= 0, got {y}")
        if width <= 0:
            raise InvalidParameterError(f"Width must be > 0, got {width}")
        if height <= 0:
            raise InvalidParameterError(f"Height must be > 0, got {height}")

        logger.debug(f"AR0134.set_window: x={x}, y={y}, width={width}, height={height}")
        self.write_register(self.Y_ADDR_START, y)
        self.write_register(self.X_ADDR_START, x)
        self.write_register(self.Y_ADDR_END, y + height - 1)
        self.write_register(self.X_ADDR_END, x + width - 1)

    def get_total_width(self) -> int:
        """Get total frame width including blanking.

        Returns:
            Total frame width in pixels

        Raises:
            CommunicationError: If read operation fails
        """
        return self.read_register(self.LINE_LENGTH_PCK)

    def get_total_height(self) -> int:
        """Get total frame height including blanking.

        Returns:
            Total frame height in lines

        Raises:
            CommunicationError: If read operation fails
        """
        return self.read_register(self.FRAME_LENGTH_LINES)

    def set_frame_timings(self, min_blank: bool) -> None:
        """Configure frame timing parameters.

        Args:
            min_blank: Use minimum blanking intervals

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug(f"AR0134.set_frame_timings: min_blank={min_blank}")
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
        """Get coarse exposure time.

        Returns:
            Coarse exposure time value

        Raises:
            CommunicationError: If read operation fails
        """
        return self.read_register(self.COARSE_INTEGRATION_TIME)

    def set_coarse_exposure_time(self, coarse: int) -> None:
        """Set coarse exposure time.

        Args:
            coarse: Coarse exposure value (must be in range 0-65535)

        Raises:
            InvalidParameterError: If coarse is out of range
            CommunicationError: If operation fails
        """
        if not (0 <= coarse <= 65535):
            raise InvalidParameterError(
                f"Coarse exposure must be in range 0-65535, got {coarse}"
            )
        logger.debug(f"AR0134.set_coarse_exposure_time: {coarse}")
        self.write_register(self.COARSE_INTEGRATION_TIME, coarse & 0xFFFF)

    def get_fine_exposure_time(self) -> int:
        """Get fine exposure time.

        Returns:
            Fine exposure time value

        Raises:
            CommunicationError: If read operation fails
        """
        return self.read_register(self.FINE_INTEGRATION_TIME)

    def set_fine_exposure_time(self, fine: int) -> None:
        """Set fine exposure time.

        Args:
            fine: Fine exposure value (must be in range 0-65535)

        Raises:
            InvalidParameterError: If fine is out of range
            CommunicationError: If operation fails
        """
        if not (0 <= fine <= 65535):
            raise InvalidParameterError(
                f"Fine exposure must be in range 0-65535, got {fine}"
            )
        logger.debug(f"AR0134.set_fine_exposure_time: {fine}")
        self.write_register(self.FINE_INTEGRATION_TIME, fine & 0xFFFF)

    def get_exposure_time(self) -> int:
        """Get total exposure time.

        Returns:
            Total exposure time in pixel clocks

        Raises:
            CommunicationError: If read operation fails
        """
        frame_width = self.read_register(self.LINE_LENGTH_PCK)
        coarse = self.read_register(self.COARSE_INTEGRATION_TIME)
        fine = self.read_register(self.FINE_INTEGRATION_TIME)
        return coarse * frame_width + fine

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
        logger.debug(f"AR0134.set_exposure_time: {exposure}")
        frame_width = self.read_register(self.LINE_LENGTH_PCK)
        self.write_register(
            self.COARSE_INTEGRATION_TIME, (exposure // frame_width) & 0xFFFF
        )
        self.write_register(
            self.FINE_INTEGRATION_TIME, (exposure % frame_width) & 0xFFFF
        )

    def set_sync(self, enable: bool) -> None:
        """Configure sensor synchronization mode.

        Args:
            enable: Enable external synchronization

        Raises:
            CommunicationError: If operation fails
        """
        logger.debug(f"AR0134.set_sync: enable={enable}")
        r = self.read_register(self.RESET_REGISTER)
        r &= ~(self.STREAM | self.GPI_EN | self.FORCED_PLL_ON)
        if enable:
            r |= self.GPI_EN | self.FORCED_PLL_ON
        else:
            r |= self.STREAM
        self.write_register(self.RESET_REGISTER, r)


__all__ = ["AR0134"]
