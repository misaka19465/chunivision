"""
Console Output Adapter for ChunIVision.

Displays controller state in a progress bar-like format at the bottom of the console.
Uses ANSI escape codes for terminal control.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict

from ..utils.logger import Logger
from ..vision.height_estimator import HeightState
from ..vision.touch_detector import TouchState
from .base_output import BaseOutput

logger = Logger.get_logger(__name__)


class ConsoleOutput(BaseOutput):
    """
    Console output displaying state as a progress bar at the bottom of the terminal.

    Shows touch zones as a visual grid and height levels as a bar chart.
    Updates in-place at the bottom of the console using ANSI escape codes.

    WARNING: This output adapter conflicts with console logging!
    When using ConsoleOutput, you should configure the logger to write to a file only:
        Logger.setup(level="INFO", log_file="chunivision.log")
    Or disable console logging to avoid interference with the display.

    Example:
        >>> # Setup logging to file only
        >>> Logger.setup(level="INFO", log_file="chunivision.log")
        >>>
        >>> output = ConsoleOutput({"update_interval": 0.1})
        >>> output.initialize()
        >>> output.send_state(touch_state, height_state)
    """

    # ANSI escape codes
    CLEAR_LINE = "\033[2K"
    CURSOR_UP = "\033[{}A"
    CURSOR_DOWN = "\033[{}B"
    CURSOR_TO_START = "\r"
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"

    # Display characters
    TOUCH_ACTIVE = "█"
    TOUCH_INACTIVE = "░"
    HEIGHT_ACTIVE = "█"
    HEIGHT_INACTIVE = "░"

    def __init__(self, config: Dict[str, Any], name: str = "Console"):
        """
        Initialize console output.

        Args:
            config: Configuration dictionary with optional keys:
                - update_interval: Minimum seconds between updates (default: 0.05)
                - show_stats: Whether to show statistics (default: True)
                - color_enabled: Whether to use colors (default: True)
            name: Output name
        """
        super().__init__(config, name)
        self._update_interval = config.get("update_interval", 0.05)
        self._show_stats = config.get("show_stats", True)
        self._color_enabled = config.get("color_enabled", True)
        self._last_update = 0.0
        self._display_lines = 0
        self._is_tty = sys.stdout.isatty()

    def _validate_config(self, config: Dict[str, Any]) -> list[str]:
        """Validate console output configuration."""
        errors = []
        if "update_interval" in config:
            interval = config["update_interval"]
            if not isinstance(interval, (int, float)) or interval < 0:
                errors.append("update_interval must be a non-negative number")
        return errors

    def initialize(self) -> bool:
        """
        Initialize console output.

        WARNING: Will interfere with console logging. Configure logger to use
        file output only before initializing console output.

        Returns:
            True if initialization successful
        """
        try:
            # Check if stdout is a terminal
            if not self._is_tty:
                logger.warning(f"{self.name}: Not a TTY, display may not work properly")

            # Warn if console logging is enabled
            root_logger = logging.getLogger()
            has_console_handler = any(
                isinstance(h, logging.StreamHandler)
                and h.stream in (sys.stdout, sys.stderr)
                for h in root_logger.handlers
            )
            if has_console_handler:
                logger.warning(
                    f"{self.name}: Console logging is enabled! This will interfere with the display. "
                    "Consider using Logger.setup(log_file='...') to log to file only."
                )

            # Hide cursor for cleaner display
            if self._is_tty and self._color_enabled:
                sys.stdout.write(self.HIDE_CURSOR)
                sys.stdout.flush()

            self._set_initialized(True)
            self._set_connected(True)
            logger.info(f"{self.name}: Console output initialized")
            return True

        except Exception as e:
            logger.error(f"{self.name}: Initialization failed: {e}")
            return False

    def send_state(self, touch_state: TouchState, height_state: HeightState) -> bool:
        """
        Display current state in console.

        Args:
            touch_state: Current touch zone states
            height_state: Current height level states

        Returns:
            True if display updated successfully
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self._last_update < self._update_interval:
                return True

            start_time = time.time()

            # Build display
            display = self._build_display(touch_state, height_state)

            # Clear previous display and show new one
            self._clear_previous_display()
            sys.stdout.write(display)
            sys.stdout.flush()

            # Update stats
            duration_ms = (time.time() - start_time) * 1000
            self._update_stats_success(len(display.encode()), duration_ms)
            self._last_update = current_time

            return True

        except Exception as e:
            self._update_stats_failure(str(e))
            logger.error(f"{self.name}: Display update failed: {e}")
            return False

    def close(self) -> None:
        """Clean up and restore terminal state."""
        try:
            # Clear display
            self._clear_previous_display()

            # Show cursor again
            if self._is_tty and self._color_enabled:
                sys.stdout.write(self.SHOW_CURSOR)
                sys.stdout.flush()

            self._set_connected(False)
            self._set_initialized(False)
            logger.info(f"{self.name}: Console output closed")

        except Exception as e:
            logger.error(f"{self.name}: Error during close: {e}")

    def _build_display(self, touch_state: TouchState, height_state: HeightState) -> str:
        """
        Build the console display string.

        Args:
            touch_state: Touch zone states
            height_state: Height level states

        Returns:
            Formatted display string
        """
        lines = []

        # Header
        header = self._build_header()
        lines.append(header)

        # Touch zones display (2 rows × 16 columns)
        touch_display = self._build_touch_display(touch_state)
        lines.extend(touch_display)

        # Height levels display
        height_display = self._build_height_display(height_state)
        lines.append(height_display)

        # Statistics (if enabled)
        if self._show_stats:
            stats_display = self._build_stats_display()
            lines.append(stats_display)

        # Footer separator
        footer = self._build_footer()
        lines.append(footer)

        self._display_lines = len(lines)
        return "\n".join(lines) + "\n"

    def _build_header(self) -> str:
        """Build header line."""
        if self._color_enabled:
            return f"{self.BOLD}{self.CYAN}╔══════════════════════════════════════════════════════════════╗{self.RESET}"
        return "╔══════════════════════════════════════════════════════════════╗"

    def _build_footer(self) -> str:
        """Build footer line."""
        if self._color_enabled:
            return f"{self.BOLD}{self.CYAN}╚══════════════════════════════════════════════════════════════╝{self.RESET}"
        return "╚══════════════════════════════════════════════════════════════╝"

    def _build_touch_display(self, touch_state: TouchState) -> list[str]:
        """
        Build touch zone display (2 rows).

        Args:
            touch_state: Touch zone states

        Returns:
            List of display lines
        """
        lines = []

        # Title
        title = "│ Touch Zones:                                               │"
        if self._color_enabled:
            title = f"{self.CYAN}{title}{self.RESET}"
        lines.append(title)

        # Top row (zones 2, 4, 6, ..., 32) - even numbers
        top_row = "│   "
        for col in range(16):
            zone_id = (col * 2) + 2  # Even zones: 2, 4, 6, ..., 32
            is_touched = touch_state.is_touched(zone_id)
            char = self.TOUCH_ACTIVE if is_touched else self.TOUCH_INACTIVE
            if self._color_enabled:
                color = self.GREEN if is_touched else self.DIM
                top_row += f"{color}{char}{char}{self.RESET} "
            else:
                top_row += f"{char}{char} "
        top_row += "│"
        lines.append(top_row)

        # Bottom row (zones 1, 3, 5, ..., 31) - odd numbers
        bottom_row = "│   "
        for col in range(16):
            zone_id = (col * 2) + 1  # Odd zones: 1, 3, 5, ..., 31
            is_touched = touch_state.is_touched(zone_id)
            char = self.TOUCH_ACTIVE if is_touched else self.TOUCH_INACTIVE
            if self._color_enabled:
                color = self.GREEN if is_touched else self.DIM
                bottom_row += f"{color}{char}{char}{self.RESET} "
            else:
                bottom_row += f"{char}{char} "
        bottom_row += "│"
        lines.append(bottom_row)

        return lines

    def _build_height_display(self, height_state: HeightState) -> str:
        """
        Build height levels display.

        Args:
            height_state: Height level states

        Returns:
            Display line
        """
        line = "│ Air Sensors: "

        # Display 6 height levels
        for level in range(6):
            is_active = height_state.is_active(level)
            char = self.HEIGHT_ACTIVE if is_active else self.HEIGHT_INACTIVE
            if self._color_enabled:
                if is_active:
                    # Color gradient based on height
                    colors = [
                        self.GREEN,
                        self.YELLOW,
                        self.YELLOW,
                        self.RED,
                        self.RED,
                        self.MAGENTA,
                    ]
                    color = colors[level]
                else:
                    color = self.DIM
                line += f"{color}{char}{char}{char}{self.RESET} "
            else:
                line += f"{char}{char}{char} "

        # Pad to width
        line += " " * (48 - len(line) + len("│ Air Sensors: ")) + "│"
        return line

    def _build_stats_display(self) -> str:
        """Build statistics display line."""
        stats = self.stats
        success_rate = (
            stats.packets_sent / (stats.packets_sent + stats.packets_failed)
            if (stats.packets_sent + stats.packets_failed) > 0
            else 0.0
        )

        line = f"│ Sent: {stats.packets_sent:6d} | Failed: {stats.packets_failed:4d} | Rate: {success_rate:5.1%} "

        # Pad to width
        padding = 62 - len(line) + line.count(self.RESET) * len(self.RESET)
        line += " " * max(0, padding - len(self.RESET) * line.count(self.RESET)) + "│"

        if self._color_enabled:
            # Color the stats
            if success_rate >= 0.95:
                color = self.GREEN
            elif success_rate >= 0.8:
                color = self.YELLOW
            else:
                color = self.RED
            line = f"{self.DIM}{line[:2]}{self.RESET}{color}{line[2:-1]}{self.RESET}{self.DIM}{line[-1]}{self.RESET}"

        return line

    def _clear_previous_display(self) -> None:
        """Clear the previous display from console."""
        if not self._is_tty or self._display_lines == 0:
            return

        # Move cursor up and clear lines
        for _ in range(self._display_lines):
            sys.stdout.write(self.CURSOR_UP.format(1))
            sys.stdout.write(self.CLEAR_LINE)

    def __del__(self):
        """Ensure cursor is shown on deletion."""
        try:
            if self._is_tty and self._color_enabled and self.is_initialized():
                sys.stdout.write(self.SHOW_CURSOR)
                sys.stdout.flush()
        except:
            pass
