"""
Camera Manager for ChunIVision.

Manages dual Oculus Rift CV1 cameras for stereo vision, including initialization,
synchronized frame capture, and error recovery.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config.camera_config import CameraConfig
from ..oculus.exceptions import (
    DeviceInitializationError,
    DeviceNotFoundError,
    StreamingError,
)
from ..oculus.oculus_camera import OculusRiftCV1Camera
from ..oculus.triple_buffer import TripleBuffer
from ..utils.logger import Logger

logger = Logger.get_logger(__name__)


class CameraInitError(Exception):
    """Raised when camera initialization fails."""

    pass


class CameraManager:
    """
    Manages two infrared cameras for stereo vision.

    This class handles dual Oculus Rift CV1 camera initialization, synchronized
    frame capture using triple buffering, and automatic error recovery.

    Attributes:
        left_camera: OculusRiftCV1Camera object for left camera
        right_camera: OculusRiftCV1Camera object for right camera
        config: CameraConfig with camera parameters
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize camera manager with configuration.

        Args:
            config: CameraConfig object with camera indices and parameters

        Raises:
            CameraInitError: If configuration is invalid
        """
        logger.info("Initializing CameraManager")

        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            error_msg = f"Invalid camera configuration: {', '.join(validation_errors)}"
            logger.error(error_msg)
            raise CameraInitError(error_msg)

        self.config = config
        self.left_camera: Optional[OculusRiftCV1Camera] = None
        self.right_camera: Optional[OculusRiftCV1Camera] = None

        # Triple buffers for frame synchronization
        self._left_buffer: TripleBuffer[Tuple[np.ndarray, float]] = TripleBuffer()
        self._right_buffer: TripleBuffer[Tuple[np.ndarray, float]] = TripleBuffer()

        # Streaming state
        self._streaming = False
        self._streaming_lock = threading.Lock()

        # Statistics
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._fps = 0.0

        logger.info(
            f"CameraManager initialized with left_serial={config.left_camera_serial}, "
            f"right_serial={config.right_camera_serial}"
        )

    def initialize(self) -> bool:
        """
        Initialize both cameras and verify they are ready.

        Returns:
            True if both cameras initialized successfully, False otherwise

        Raises:
            CameraInitError: If cameras cannot be initialized
        """
        logger.info("Initializing cameras")

        try:
            # Initialize left camera
            logger.debug(
                f"Opening left camera (serial={self.config.left_camera_serial})"
            )
            try:
                self.left_camera = OculusRiftCV1Camera(
                    serial_number=self.config.left_camera_serial
                )
                logger.info("Left camera initialized successfully")
            except (DeviceNotFoundError, DeviceInitializationError) as e:
                logger.error(f"Failed to initialize left camera: {e}")
                raise CameraInitError(f"Left camera initialization failed: {e}") from e

            # Initialize right camera
            logger.debug(
                f"Opening right camera (serial={self.config.right_camera_serial})"
            )
            try:
                self.right_camera = OculusRiftCV1Camera(
                    serial_number=self.config.right_camera_serial
                )
                logger.info("Right camera initialized successfully")
            except (DeviceNotFoundError, DeviceInitializationError) as e:
                logger.error(f"Failed to initialize right camera: {e}")
                # Clean up left camera
                if self.left_camera:
                    self.left_camera.close()
                    self.left_camera = None
                raise CameraInitError(f"Right camera initialization failed: {e}") from e

            # Verify both cameras are ready
            if not self.is_ready():
                self.release()
                raise CameraInitError("Cameras initialized but not ready")

            logger.info("Both cameras initialized and ready")
            return True

        except CameraInitError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during camera initialization: {e}")
            self.release()
            raise CameraInitError(f"Camera initialization failed: {e}") from e

    def get_frame_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Capture synchronized frames from both cameras.

        This method retrieves the most recent frames from both cameras' triple buffers.
        Frames are synchronized by timestamp (within reasonable tolerance).

        Returns:
            Tuple of (left_frame, right_frame, timestamp) or None if capture failed
            - left_frame: numpy array of shape (H, W) for infrared image
            - right_frame: numpy array of shape (H, W) for infrared image
            - timestamp: average timestamp in seconds since epoch

        Note:
            Returns None if:
            - Cameras are not streaming
            - No frames available in buffers
            - Frame synchronization fails
        """
        if not self._streaming:
            logger.warning("Cannot get frame pair: cameras not streaming")
            return None

        # Get latest frames from triple buffers
        left_data = self._left_buffer.get_latest()
        right_data = self._right_buffer.get_latest()

        if left_data is None or right_data is None:
            return None

        left_frame, left_timestamp = left_data
        right_frame, right_timestamp = right_data

        # Check timestamp synchronization (allow up to 20ms difference)
        timestamp_diff = abs(left_timestamp - right_timestamp)
        if timestamp_diff > 0.020:  # 20ms
            logger.warning(
                f"Frame timestamp mismatch: {timestamp_diff*1000:.1f}ms "
                "(may indicate dropped frames)"
            )

        # Use average timestamp
        avg_timestamp = (left_timestamp + right_timestamp) / 2.0

        # Update statistics
        self._frame_count += 1
        current_time = time.time()
        if self._last_frame_time > 0:
            frame_interval = current_time - self._last_frame_time
            if frame_interval > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / frame_interval)
        self._last_frame_time = current_time

        return (left_frame, right_frame, avg_timestamp)

    def start_streaming(self) -> None:
        """
        Start streaming from both cameras.

        Raises:
            CameraInitError: If cameras are not initialized
            StreamingError: If streaming fails to start
        """
        with self._streaming_lock:
            if self._streaming:
                logger.warning("Cameras already streaming")
                return

            if not self.is_ready():
                raise CameraInitError("Cameras not initialized or ready")

            logger.info("Starting camera streaming")

            try:
                # Start left camera streaming
                if self.left_camera is None:
                    raise CameraInitError("Left camera not initialized")
                self.left_camera.start_streaming(self._left_frame_callback)
                logger.debug("Left camera streaming started")

                # Start right camera streaming
                try:
                    if self.right_camera is None:
                        raise CameraInitError("Right camera not initialized")
                    self.right_camera.start_streaming(self._right_frame_callback)
                    logger.debug("Right camera streaming started")
                except Exception as e:
                    # Stop left camera if right camera fails
                    logger.error(f"Failed to start right camera streaming: {e}")
                    self.left_camera.stop_streaming()
                    raise

                self._streaming = True
                self._frame_count = 0
                self._last_frame_time = 0.0
                self._fps = 0.0

                logger.info("Camera streaming started successfully")

            except Exception as e:
                logger.error(f"Failed to start streaming: {e}")
                raise StreamingError(f"Failed to start streaming: {e}") from e

    def stop_streaming(self) -> None:
        """Stop streaming from both cameras."""
        with self._streaming_lock:
            if not self._streaming:
                return

            logger.info("Stopping camera streaming")

            try:
                if self.left_camera:
                    self.left_camera.stop_streaming()
                    logger.debug("Left camera streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping left camera: {e}")

            try:
                if self.right_camera:
                    self.right_camera.stop_streaming()
                    logger.debug("Right camera streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping right camera: {e}")

            self._streaming = False

            # Clear buffers
            self._left_buffer.clear()
            self._right_buffer.clear()

            logger.info("Camera streaming stopped")

    def release(self) -> None:
        """Release camera resources and close connections."""
        logger.info("Releasing camera resources")

        # Stop streaming if active
        if self._streaming:
            self.stop_streaming()

        # Close cameras
        if self.left_camera:
            try:
                self.left_camera.close()
                logger.debug("Left camera closed")
            except Exception as e:
                logger.error(f"Error closing left camera: {e}")
            finally:
                self.left_camera = None

        if self.right_camera:
            try:
                self.right_camera.close()
                logger.debug("Right camera closed")
            except Exception as e:
                logger.error(f"Error closing right camera: {e}")
            finally:
                self.right_camera = None

        logger.info("Camera resources released")

    def is_ready(self) -> bool:
        """
        Check if both cameras are connected and ready.

        Returns:
            True if both cameras are initialized, False otherwise
        """
        return (
            self.left_camera is not None
            and self.right_camera is not None
            and hasattr(self.left_camera, "handle")
            and hasattr(self.right_camera, "handle")
            and self.left_camera.handle is not None
            and self.right_camera.handle is not None
        )

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get information about connected cameras.

        Returns:
            Dictionary with camera properties (resolution, FPS, etc.)
        """
        info = {
            "left_camera_ready": self.left_camera is not None,
            "right_camera_ready": self.right_camera is not None,
            "streaming": self._streaming,
            "frame_count": self._frame_count,
            "fps": self._fps,
        }

        if self.left_camera:
            info["left_camera"] = {
                "serial": self.config.left_camera_serial,
                "resolution": self.left_camera.get_frame_size(),
                "width": self.left_camera.get_frame_width(),
                "height": self.left_camera.get_frame_height(),
            }

        if self.right_camera:
            info["right_camera"] = {
                "serial": self.config.right_camera_serial,
                "resolution": self.right_camera.get_frame_size(),
                "width": self.right_camera.get_frame_width(),
                "height": self.right_camera.get_frame_height(),
            }

        return info

    def get_fps(self) -> float:
        """
        Get current frames per second.

        Returns:
            Current FPS estimate
        """
        return self._fps

    def _left_frame_callback(self, frame_data: bytes, metadata: Dict[str, Any]) -> None:
        """
        Callback for left camera frames.

        Args:
            frame_data: Raw frame bytes
            metadata: Frame metadata (frame_id, presentation_time, etc.)
        """
        try:
            # Convert raw bytes to numpy array
            # Oculus cameras produce 8-bit grayscale IR images
            if self.left_camera is None:
                logger.error("Left camera is None in callback")
                return
            width, height = self.left_camera.get_frame_size()
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width))

            # Store in triple buffer with timestamp
            timestamp = time.time()
            self._left_buffer.publish((frame, timestamp))

        except Exception as e:
            logger.error(f"Error processing left frame: {e}")

    def _right_frame_callback(
        self, frame_data: bytes, metadata: Dict[str, Any]
    ) -> None:
        """
        Callback for right camera frames.

        Args:
            frame_data: Raw frame bytes
            metadata: Frame metadata (frame_id, presentation_time, etc.)
        """
        try:
            # Convert raw bytes to numpy array
            if self.right_camera is None:
                logger.error("Right camera is None in callback")
                return
            width, height = self.right_camera.get_frame_size()
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width))

            # Store in triple buffer with timestamp
            timestamp = time.time()
            self._right_buffer.publish((frame, timestamp))

        except Exception as e:
            logger.error(f"Error processing right frame: {e}")

    def __enter__(self) -> "CameraManager":
        """
        Enter context manager.

        Returns:
            Self for context manager protocol
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager and cleanup resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.release()
