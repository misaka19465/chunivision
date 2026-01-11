"""
Centralized logging system for ChunIVision.

Provides consistent logging across all modules with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file output
- Log rotation
- Performance-optimized (no string formatting on disabled levels)
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Centralized logging system.

    Provides static methods for setting up and retrieving loggers
    across the application with consistent configuration.
    """

    _initialized: bool = False
    _log_level: str = "INFO"
    _log_file: Optional[Path] = None
    _formatter: Optional[logging.Formatter] = None

    @staticmethod
    def setup(
        level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: Optional[str] = None,
        enable_rotation: bool = True,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
    ) -> None:
        """
        Configure the logging system.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file. If None, only console logging is enabled
            log_format: Custom log format string. If None, uses default format
            enable_rotation: Whether to enable log rotation for file logging
            max_bytes: Maximum size of each log file before rotation (default 10MB)
            backup_count: Number of backup log files to keep (default 5)
        """
        Logger._initialized = True
        Logger._log_level = level.upper()
        Logger._log_file = Path(log_file) if log_file else None

        # Set default format if not provided
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        Logger._formatter = logging.Formatter(log_format)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, Logger._log_level))

        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, Logger._log_level))
        console_handler.setFormatter(Logger._formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if log file specified
        if Logger._log_file:
            # Create log directory if it doesn't exist
            Logger._log_file.parent.mkdir(parents=True, exist_ok=True)

            if enable_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    Logger._log_file, maxBytes=max_bytes, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(Logger._log_file)

            file_handler.setLevel(getattr(logging, Logger._log_level))
            file_handler.setFormatter(Logger._formatter)
            root_logger.addHandler(file_handler)

        # Log initialization
        logger = Logger.get_logger("Logger")
        logger.info(f"Logging system initialized (level={Logger._log_level})")
        if Logger._log_file:
            logger.info(f"Log file: {Logger._log_file}")

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance for a module.

        Args:
            name: Name of the logger (typically __name__ from the calling module)

        Returns:
            Configured logger instance
        """
        # Initialize with defaults if not already set up
        if not Logger._initialized:
            Logger.setup()

        return logging.getLogger(name)

    @staticmethod
    def set_level(level: str) -> None:
        """
        Change the logging level dynamically.

        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        Logger._log_level = level.upper()
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, Logger._log_level))

        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(getattr(logging, Logger._log_level))

        logger = Logger.get_logger("Logger")
        logger.info(f"Logging level changed to {Logger._log_level}")

    @staticmethod
    def get_level() -> str:
        """
        Get the current logging level.

        Returns:
            Current logging level as string
        """
        return Logger._log_level

    @staticmethod
    def shutdown() -> None:
        """
        Shutdown the logging system and flush all handlers.
        """
        logging.shutdown()
        Logger._initialized = False
