"""
Output adapters for ChunIVision.

This module contains output adapters for different protocols:
- BaseOutput: Abstract base class for all outputs
- OutputManager: Manages multiple outputs simultaneously
- ConsoleOutput: Terminal display output
- Serial (COM port) - TODO: Phase 4
- HID (USB device) - TODO: Phase 4
- Keyboard (virtual keyboard) - TODO: Phase 4
- UDP (network) - TODO: Phase 4
"""

from .base_output import BaseOutput, OutputError, OutputStats
from .console_output import ConsoleOutput
from .output_manager import OutputManager

__all__ = [
    "BaseOutput",
    "OutputError",
    "OutputStats",
    "ConsoleOutput",
    "OutputManager",
]
