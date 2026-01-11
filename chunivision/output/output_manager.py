"""
Output Manager for ChunIVision.

Manages multiple output adapters simultaneously, broadcasting state updates
to all active outputs. Provides error isolation so one failing output doesn't
affect others.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..utils.logger import Logger
from ..vision.height_estimator import HeightState
from ..vision.touch_detector import TouchState
from .base_output import BaseOutput, OutputError

logger = Logger.get_logger(__name__)


class OutputManager:
    """
    Manages multiple output adapters simultaneously.

    Broadcasts state updates to all registered outputs. Each output is isolated,
    so errors in one output don't affect others. Collects statistics from all
    outputs for monitoring.

    Example:
        >>> manager = OutputManager()
        >>> manager.add_output("udp", UDPOutput(config))
        >>> manager.add_output("serial", SerialOutput(config))
        >>> manager.initialize_all()
        >>> manager.send_state(touch_state, height_state)
        {'udp': True, 'serial': True}
    """

    def __init__(self):
        """Initialize output manager with empty output registry."""
        self._outputs: Dict[str, BaseOutput] = {}
        self._initialized = False
        self._total_broadcasts = 0
        self._start_time = time.time()
        logger.info("OutputManager created")

    def add_output(self, name: str, output: BaseOutput) -> None:
        """
        Register an output adapter.

        Args:
            name: Unique identifier for this output
            output: BaseOutput instance to register

        Raises:
            ValueError: If name is already registered
            TypeError: If output is not a BaseOutput instance
        """
        if not isinstance(output, BaseOutput):
            raise TypeError(f"Output must be a BaseOutput instance, got {type(output)}")

        if name in self._outputs:
            raise ValueError(f"Output with name '{name}' is already registered")

        self._outputs[name] = output
        logger.info(f"Registered output: {name} ({output.__class__.__name__})")

    def remove_output(self, name: str) -> bool:
        """
        Remove an output adapter.

        Closes the output before removing it.

        Args:
            name: Name of output to remove

        Returns:
            True if output was removed, False if not found
        """
        if name not in self._outputs:
            logger.warning(f"Cannot remove output '{name}': not found")
            return False

        output = self._outputs[name]
        try:
            output.close()
        except Exception as e:
            logger.error(f"Error closing output '{name}': {e}")

        del self._outputs[name]
        logger.info(f"Removed output: {name}")
        return True

    def get_output(self, name: str) -> Optional[BaseOutput]:
        """
        Get an output adapter by name.

        Args:
            name: Name of output to retrieve

        Returns:
            BaseOutput instance or None if not found
        """
        return self._outputs.get(name)

    def has_output(self, name: str) -> bool:
        """
        Check if an output is registered.

        Args:
            name: Name to check

        Returns:
            True if output exists, False otherwise
        """
        return name in self._outputs

    def get_active_outputs(self) -> List[str]:
        """
        Get list of all registered output names.

        Returns:
            List of output names
        """
        return list(self._outputs.keys())

    def get_connected_outputs(self) -> List[str]:
        """
        Get list of output names that are currently connected.

        Returns:
            List of connected output names
        """
        return [name for name, output in self._outputs.items() if output.is_connected()]

    def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered outputs.

        Returns:
            Dictionary mapping output names to initialization success status
        """
        results = {}
        for name, output in self._outputs.items():
            try:
                success = output.initialize()
                results[name] = success
                if success:
                    logger.info(f"Initialized output: {name}")
                else:
                    logger.warning(f"Failed to initialize output: {name}")
            except Exception as e:
                results[name] = False
                logger.error(f"Error initializing output '{name}': {e}")

        self._initialized = True
        successful = sum(1 for v in results.values() if v)
        logger.info(
            f"Initialized {successful}/{len(results)} outputs: {list(results.keys())}"
        )
        return results

    def close_all(self) -> None:
        """Close all registered outputs and cleanup resources."""
        for name, output in self._outputs.items():
            try:
                output.close()
                logger.info(f"Closed output: {name}")
            except Exception as e:
                logger.error(f"Error closing output '{name}': {e}")

        logger.info("All outputs closed")

    def send_state(
        self, touch_state: TouchState, height_state: HeightState
    ) -> Dict[str, bool]:
        """
        Send state to all registered outputs.

        Broadcasts to all outputs independently. Errors in one output don't
        affect others. Each output's send operation is isolated.

        Args:
            touch_state: Current touch zone states (32 zones)
            height_state: Current height level states (6 levels)

        Returns:
            Dictionary mapping output names to send success status
            True = sent successfully, False = send failed
        """
        if not self._outputs:
            logger.debug("No outputs registered, nothing to send")
            return {}

        results = {}
        self._total_broadcasts += 1

        for name, output in self._outputs.items():
            try:
                success = output.send_state(touch_state, height_state)
                results[name] = success
                if not success:
                    logger.debug(f"Output '{name}' returned False on send")
            except Exception as e:
                results[name] = False
                logger.error(f"Exception in output '{name}' during send: {e}")

        # Log if any sends failed
        failed = [name for name, success in results.items() if not success]
        if failed:
            logger.warning(
                f"Broadcast {self._total_broadcasts}: {len(failed)} output(s) failed: {failed}"
            )

        return results

    def send_state_to(
        self, name: str, touch_state: TouchState, height_state: HeightState
    ) -> bool:
        """
        Send state to a specific output.

        Args:
            name: Name of output to send to
            touch_state: Current touch zone states
            height_state: Current height level states

        Returns:
            True if send successful, False otherwise

        Raises:
            KeyError: If output name not found
        """
        if name not in self._outputs:
            raise KeyError(f"Output '{name}' not found")

        output = self._outputs[name]
        try:
            return output.send_state(touch_state, height_state)
        except Exception as e:
            logger.error(f"Exception in output '{name}' during send: {e}")
            return False

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics from all outputs.

        Returns:
            Dictionary mapping output names to their statistics dictionaries
        """
        return {name: output.get_stats() for name, output in self._outputs.items()}

    def get_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific output.

        Args:
            name: Name of output

        Returns:
            Statistics dictionary or None if output not found
        """
        output = self._outputs.get(name)
        return output.get_stats() if output else None

    def reset_all_stats(self) -> None:
        """Reset statistics for all outputs."""
        for name, output in self._outputs.items():
            output.reset_stats()
        self._total_broadcasts = 0
        logger.info("Reset statistics for all outputs")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of output manager state.

        Returns:
            Dictionary with manager statistics and output summary
        """
        uptime = time.time() - self._start_time
        connected_count = len(self.get_connected_outputs())
        total_count = len(self._outputs)

        return {
            "total_outputs": total_count,
            "connected_outputs": connected_count,
            "initialized": self._initialized,
            "total_broadcasts": self._total_broadcasts,
            "uptime_seconds": uptime,
            "outputs": {
                name: {
                    "type": output.__class__.__name__,
                    "connected": output.is_connected(),
                    "initialized": output.is_initialized(),
                    "packets_sent": output.stats.packets_sent,
                    "packets_failed": output.stats.packets_failed,
                }
                for name, output in self._outputs.items()
            },
        }

    def is_initialized(self) -> bool:
        """
        Check if manager has been initialized.

        Returns:
            True if initialize_all() has been called
        """
        return self._initialized

    def count_outputs(self) -> int:
        """
        Get number of registered outputs.

        Returns:
            Number of outputs
        """
        return len(self._outputs)

    def __repr__(self) -> str:
        """String representation of output manager."""
        connected = len(self.get_connected_outputs())
        total = len(self._outputs)
        return f"OutputManager({connected}/{total} connected, {self._total_broadcasts} broadcasts)"

    def __enter__(self):
        """Context manager entry - initialize all outputs."""
        self.initialize_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all outputs."""
        self.close_all()
        return False

    def __len__(self) -> int:
        """Number of registered outputs."""
        return len(self._outputs)

    def __contains__(self, name: str) -> bool:
        """Check if output name is registered."""
        return name in self._outputs

    def __iter__(self):
        """Iterate over output names."""
        return iter(self._outputs.keys())
