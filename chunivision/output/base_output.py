"""
Base Output Adapter for ChunIVision.

Provides the abstract base class that all output adapters must implement.
Handles common functionality like statistics tracking and error handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..utils.logger import Logger
from ..vision.height_estimator import HeightState
from ..vision.touch_detector import TouchState

logger = Logger.get_logger(__name__)


class OutputError(Exception):
    """Base exception for output-related errors."""

    pass


@dataclass
class OutputStats:
    """
    Statistics for an output adapter.

    Attributes:
        packets_sent: Total number of packets successfully sent
        packets_failed: Total number of packets that failed to send
        bytes_sent: Total bytes transmitted
        last_send_time: Timestamp of last successful send
        last_error_time: Timestamp of last error
        last_error_message: Message from last error
        average_send_duration_ms: Moving average of send operation duration
        is_connected: Whether output is currently connected
    """

    packets_sent: int = 0
    packets_failed: int = 0
    bytes_sent: int = 0
    last_send_time: float = 0.0
    last_error_time: float = 0.0
    last_error_message: str = ""
    average_send_duration_ms: float = 0.0
    is_connected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary format."""
        return {
            "packets_sent": self.packets_sent,
            "packets_failed": self.packets_failed,
            "bytes_sent": self.bytes_sent,
            "last_send_time": self.last_send_time,
            "last_error_time": self.last_error_time,
            "last_error_message": self.last_error_message,
            "average_send_duration_ms": self.average_send_duration_ms,
            "is_connected": self.is_connected,
            "success_rate": (
                self.packets_sent / (self.packets_sent + self.packets_failed)
                if (self.packets_sent + self.packets_failed) > 0
                else 0.0
            ),
        }


class BaseOutput(ABC):
    """
    Abstract base class for output adapters.

    All output implementations (Serial, HID, Keyboard, UDP) must inherit from this class
    and implement the abstract methods.

    The base class provides:
    - Statistics tracking
    - Common error handling
    - Connection state management
    - Performance monitoring

    Attributes:
        config: Output-specific configuration dictionary
        name: Human-readable name for this output
        stats: OutputStats tracking packet counts, errors, etc.
    """

    def __init__(self, config: Dict[str, Any], name: str = "BaseOutput"):
        """
        Initialize output adapter with configuration.

        Args:
            config: Output-specific configuration parameters
            name: Human-readable name for this output (e.g., "Serial COM10")

        Raises:
            OutputError: If configuration is invalid
        """
        self.config = config
        self.name = name
        self.stats = OutputStats()
        self._initialized = False
        self._last_state: Optional[tuple[TouchState, HeightState]] = None
        self._send_count_for_avg = 0
        self._send_duration_sum = 0.0

        # Validate configuration
        validation_errors = self._validate_config(config)
        if validation_errors:
            raise OutputError(
                f"Invalid configuration for {name}: {', '.join(validation_errors)}"
            )

        logger.info(f"Created output adapter: {name}")

    def _validate_config(self, config: Dict[str, Any]) -> list[str]:
        """
        Validate configuration parameters.

        Subclasses can override to add specific validation.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation error messages (empty if valid)
        """
        # Base class has no required config
        return []

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the output connection/device.

        This method must be called before send_state().
        Should handle connection setup, device creation, etc.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            OutputError: If initialization fails critically
        """
        pass

    @abstractmethod
    def send_state(self, touch_state: TouchState, height_state: HeightState) -> bool:
        """
        Send current state to output.

        This is the main method called by the vision pipeline to transmit
        game controller state.

        Args:
            touch_state: Current touch zone states (32 zones)
            height_state: Current height level states (6 levels)

        Returns:
            True if send successful, False otherwise

        Raises:
            OutputError: If send fails critically (connection lost, etc.)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close output connection and cleanup resources.

        Should be called when shutting down the application.
        Must be safe to call multiple times.
        """
        pass

    def is_connected(self) -> bool:
        """
        Check if output is connected/ready.

        Default implementation returns connection state from stats.
        Subclasses can override for more sophisticated checks.

        Returns:
            True if connected and ready to send, False otherwise
        """
        return self.stats.is_connected

    def is_initialized(self) -> bool:
        """
        Check if output has been initialized.

        Returns:
            True if initialize() has been called successfully
        """
        return self._initialized

    def get_stats(self) -> Dict[str, Any]:
        """
        Get output statistics.

        Returns:
            Dictionary with statistics (packets sent, errors, etc.)
        """
        return self.stats.to_dict()

    def get_config(self) -> Dict[str, Any]:
        """
        Get output configuration.

        Returns:
            Configuration dictionary (copy)
        """
        return self.config.copy()

    def get_name(self) -> str:
        """
        Get output adapter name.

        Returns:
            Human-readable name
        """
        return self.name

    def _update_stats_success(self, bytes_sent: int, duration_ms: float) -> None:
        """
        Update statistics after successful send.

        Args:
            bytes_sent: Number of bytes transmitted
            duration_ms: Send operation duration in milliseconds
        """
        self.stats.packets_sent += 1
        self.stats.bytes_sent += bytes_sent
        self.stats.last_send_time = time.time()

        # Update moving average of send duration (last 100 sends)
        self._send_count_for_avg += 1
        self._send_duration_sum += duration_ms
        if self._send_count_for_avg >= 100:
            self.stats.average_send_duration_ms = (
                self._send_duration_sum / self._send_count_for_avg
            )
            self._send_count_for_avg = 0
            self._send_duration_sum = 0.0

    def _update_stats_failure(self, error_message: str) -> None:
        """
        Update statistics after failed send.

        Args:
            error_message: Description of the error
        """
        self.stats.packets_failed += 1
        self.stats.last_error_time = time.time()
        self.stats.last_error_message = error_message
        logger.warning(f"{self.name}: Send failed - {error_message}")

    def _set_connected(self, connected: bool) -> None:
        """
        Update connection state.

        Args:
            connected: Whether output is currently connected
        """
        if self.stats.is_connected != connected:
            self.stats.is_connected = connected
            status = "connected" if connected else "disconnected"
            logger.info(f"{self.name}: {status}")

    def _set_initialized(self, initialized: bool) -> None:
        """
        Update initialization state.

        Args:
            initialized: Whether output is initialized
        """
        self._initialized = initialized

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self.stats = OutputStats(is_connected=self.stats.is_connected)
        self._send_count_for_avg = 0
        self._send_duration_sum = 0.0
        logger.info(f"{self.name}: Statistics reset")

    def __repr__(self) -> str:
        """String representation of output adapter."""
        status = "initialized" if self._initialized else "not initialized"
        connected = "connected" if self.stats.is_connected else "disconnected"
        return f"{self.__class__.__name__}(name='{self.name}', {status}, {connected})"

    def __enter__(self):
        """Context manager entry - initialize output."""
        if not self.initialize():
            raise OutputError(f"Failed to initialize {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close output."""
        self.close()
        return False
