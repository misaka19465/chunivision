"""
Vision Pipeline for ChunIVision.

Orchestrates the complete vision processing workflow from camera capture
through hand detection to state output. Runs in a dedicated thread for
real-time processing with minimal latency.

Processing Flow:
1. Capture synchronized stereo frames from cameras
2. Generate depth map and 3D point cloud
3. Detect and track hands in 3D space
4. Map hands to touch zones and height levels
5. Update state and invoke callbacks
6. Monitor performance metrics
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..calibration.calibration_data import CalibrationData
from ..config.camera_config import CameraConfig
from ..config.zone_config import ZoneConfig
from ..utils.logger import Logger
from ..utils.performance import PerformanceMonitor, PerformanceStats
from .camera_manager import CameraManager
from .hand_detector import Hand, HandDetector, HandDetectorConfig
from .height_estimator import HeightEstimator, HeightEstimatorConfig, HeightState
from .stereo_processor import StereoConfig, StereoProcessor
from .touch_detector import TouchDetector, TouchDetectorConfig, TouchState

logger = Logger.get_logger(__name__)


@dataclass
class VisionPipelineConfig:
    """
    Configuration for the vision pipeline.

    Attributes:
        camera_config: Camera configuration
        calibration_data: Stereo and perspective calibration data
        zone_config: Touch zone layout configuration
        stereo_config: Stereo processing configuration
        hand_detector_config: Hand detection configuration
        touch_detector_config: Touch detection configuration
        height_estimator_config: Height estimation configuration
        target_fps: Target processing frame rate
        enable_performance_monitoring: Whether to track performance metrics
        max_processing_latency_ms: Maximum acceptable latency before warning
    """

    camera_config: CameraConfig
    calibration_data: CalibrationData
    zone_config: ZoneConfig
    stereo_config: StereoConfig = field(default_factory=StereoConfig)
    hand_detector_config: HandDetectorConfig = field(default_factory=HandDetectorConfig)
    touch_detector_config: TouchDetectorConfig = field(
        default_factory=TouchDetectorConfig
    )
    height_estimator_config: HeightEstimatorConfig = field(
        default_factory=HeightEstimatorConfig
    )
    target_fps: float = 60.0
    enable_performance_monitoring: bool = True
    max_processing_latency_ms: float = 10.0

    def validate(self) -> List[str]:
        """
        Validate pipeline configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate camera config
        errors.extend(self.camera_config.validate())

        # Validate calibration data
        errors.extend(self.calibration_data.validate())

        # Zone config doesn't have a validate method - it's a simple dataclass

        # Validate stereo config
        errors.extend(self.stereo_config.validate())

        # Validate hand detector config
        errors.extend(self.hand_detector_config.validate())

        # Validate touch detector config
        errors.extend(self.touch_detector_config.validate())

        # Validate height estimator config
        errors.extend(self.height_estimator_config.validate())

        # Validate target FPS
        if self.target_fps <= 0:
            errors.append(f"target_fps must be positive, got {self.target_fps}")

        if self.max_processing_latency_ms <= 0:
            errors.append(
                f"max_processing_latency_ms must be positive, got {self.max_processing_latency_ms}"
            )

        return errors


class VisionPipelineError(Exception):
    """Raised when pipeline encounters unrecoverable error."""

    pass


class VisionPipeline:
    """
    Main vision processing pipeline coordinator.

    Manages all vision components and runs the processing loop in a dedicated
    thread. Provides callbacks for state updates and performance monitoring.

    Example:
        >>> config = VisionPipelineConfig(...)
        >>> pipeline = VisionPipeline(config)
        >>> pipeline.set_callback(lambda touch, height: process_state(touch, height))
        >>> pipeline.start()
        >>> # ... pipeline runs in background ...
        >>> pipeline.stop()
    """

    def __init__(self, config: VisionPipelineConfig):
        """
        Initialize vision pipeline with configuration.

        Args:
            config: Pipeline configuration

        Raises:
            VisionPipelineError: If configuration is invalid
        """
        logger.info("Initializing VisionPipeline")

        # Validate configuration
        errors = config.validate()
        if errors:
            error_msg = f"Invalid pipeline configuration: {', '.join(errors)}"
            logger.error(error_msg)
            raise VisionPipelineError(error_msg)

        self.config = config

        # Initialize components
        try:
            logger.info("Initializing camera manager")
            self.camera_manager = CameraManager(config.camera_config)

            logger.info("Initializing stereo processor")
            self.stereo_processor = StereoProcessor(
                config.calibration_data, config.stereo_config
            )

            logger.info("Initializing hand detector")
            self.hand_detector = HandDetector(config.hand_detector_config)

            logger.info("Initializing touch detector")
            self.touch_detector = TouchDetector(
                config.zone_config, config.touch_detector_config
            )

            logger.info("Initializing height estimator")
            self.height_estimator = HeightEstimator(config.height_estimator_config)

        except Exception as e:
            error_msg = f"Failed to initialize pipeline components: {e}"
            logger.error(error_msg)
            raise VisionPipelineError(error_msg) from e

        # Performance monitoring
        self.performance_monitor: Optional[PerformanceMonitor] = None
        if config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(
                fps_window=60, latency_window=100
            )

        # Processing thread
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Callback for state updates
        self._callback: Optional[Callable[[TouchState, HeightState], None]] = None
        self._callback_lock = threading.Lock()

        # Current state
        self._current_touch_state: Optional[TouchState] = None
        self._current_height_state: Optional[HeightState] = None
        self._state_lock = threading.Lock()

        # Previous hands for tracking
        self._previous_hands: List[Hand] = []

        # Statistics
        self._frame_count = 0
        self._error_count = 0
        self._last_warning_time = 0.0

        logger.info("VisionPipeline initialized successfully")

    def initialize(self) -> bool:
        """
        Initialize cameras and verify all components are ready.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            VisionPipelineError: If initialization fails
        """
        logger.info("Initializing pipeline components")

        try:
            # Initialize cameras
            if not self.camera_manager.initialize():
                logger.error("Failed to initialize cameras")
                return False

            # Verify cameras are ready
            if not self.camera_manager.is_ready():
                logger.error("Cameras not ready after initialization")
                return False

            logger.info("Pipeline initialization complete")
            return True

        except Exception as e:
            error_msg = f"Pipeline initialization failed: {e}"
            logger.error(error_msg)
            raise VisionPipelineError(error_msg) from e

    def start(self) -> None:
        """
        Start the vision processing loop in a dedicated thread.

        Raises:
            VisionPipelineError: If pipeline is already running or start fails
        """
        if self._running:
            raise VisionPipelineError("Pipeline is already running")

        logger.info("Starting vision pipeline")

        # Ensure cameras are initialized
        if not self.camera_manager.is_ready():
            logger.info("Cameras not initialized, initializing now")
            if not self.initialize():
                raise VisionPipelineError("Failed to initialize cameras")

        # Reset stop event
        self._stop_event.clear()

        # Start processing thread
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop, name="VisionPipeline", daemon=True
        )
        self._processing_thread.start()

        logger.info("Vision pipeline started")

    def stop(self) -> None:
        """
        Stop the vision processing loop and cleanup resources.

        Blocks until processing thread terminates.
        """
        if not self._running:
            logger.warning("Pipeline is not running")
            return

        logger.info("Stopping vision pipeline")

        # Signal thread to stop
        self._running = False
        self._stop_event.set()

        # Wait for thread to terminate
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2.0)
            if self._processing_thread.is_alive():
                logger.warning("Processing thread did not terminate cleanly")

        # Release camera resources
        self.camera_manager.release()

        logger.info(
            f"Vision pipeline stopped (processed {self._frame_count} frames, "
            f"{self._error_count} errors)"
        )

    def set_callback(self, callback: Callable[[TouchState, HeightState], None]) -> None:
        """
        Set callback function to receive state updates.

        The callback is invoked in the processing thread whenever state changes
        are detected. The callback should be fast and non-blocking.

        Args:
            callback: Function called with (touch_state, height_state)
                     whenever state is updated
        """
        with self._callback_lock:
            self._callback = callback
            logger.info("State update callback registered")

    def get_stats(self) -> Optional[PerformanceStats]:
        """
        Get current performance statistics.

        Returns:
            PerformanceStats object with FPS, latency, etc., or None if
            performance monitoring is disabled
        """
        if self.performance_monitor is None:
            return None
        return self.performance_monitor.get_stats()

    def is_running(self) -> bool:
        """
        Check if pipeline is currently running.

        Returns:
            True if processing loop is active, False otherwise
        """
        return self._running

    def get_current_state(self) -> Tuple[Optional[TouchState], Optional[HeightState]]:
        """
        Get the most recent touch and height states.

        Returns:
            Tuple of (touch_state, height_state), or (None, None) if no state available

        Note:
            This is thread-safe and can be called while pipeline is running
        """
        with self._state_lock:
            return self._current_touch_state, self._current_height_state

    def get_frame_count(self) -> int:
        """
        Get total number of frames processed.

        Returns:
            Total frame count since pipeline started
        """
        return self._frame_count

    def get_error_count(self) -> int:
        """
        Get total number of processing errors.

        Returns:
            Total error count since pipeline started
        """
        return self._error_count

    def _processing_loop(self) -> None:
        """
        Main processing loop (runs in dedicated thread).

        This is the core of the vision pipeline, executing the following steps
        for each frame:
        1. Capture stereo frame pair
        2. Generate depth map and point cloud
        3. Detect hands
        4. Detect touches and estimate heights
        5. Update state and invoke callbacks
        """
        logger.info("Processing loop started")

        # Target frame time for FPS limiting
        target_frame_time = 1.0 / self.config.target_fps
        last_stats_log_time = time.time()
        stats_log_interval = 5.0  # Log stats every 5 seconds

        while self._running and not self._stop_event.is_set():
            frame_start_time = time.time()

            try:
                # Mark frame start for FPS tracking
                if self.performance_monitor:
                    self.performance_monitor.record_frame()

                # Step 1: Capture stereo frame pair
                stage_start = time.time()
                frame_pair = self.camera_manager.get_frame_pair()
                if frame_pair is None:
                    logger.warning("Failed to capture frame pair")
                    self._error_count += 1
                    continue

                left_frame, right_frame, timestamp = frame_pair
                if self.performance_monitor:
                    self.performance_monitor.record_latency(
                        "capture", (time.time() - stage_start) * 1000
                    )

                # Step 2: Generate depth map and point cloud
                stage_start = time.time()
                point_cloud = self.stereo_processor.process(left_frame, right_frame)
                if self.performance_monitor:
                    self.performance_monitor.record_latency(
                        "stereo", (time.time() - stage_start) * 1000
                    )

                # Step 3: Detect hands
                stage_start = time.time()
                detected_hands = self.hand_detector.detect(point_cloud)

                # Track hands across frames
                if self._previous_hands:
                    detected_hands = self.hand_detector.track(
                        detected_hands, self._previous_hands
                    )
                self._previous_hands = detected_hands

                if self.performance_monitor:
                    self.performance_monitor.record_latency(
                        "hands", (time.time() - stage_start) * 1000
                    )

                # Step 4: Detect touches
                stage_start = time.time()
                touch_state = self.touch_detector.detect_touches(detected_hands)
                touch_state.timestamp = timestamp
                if self.performance_monitor:
                    self.performance_monitor.record_latency(
                        "touch", (time.time() - stage_start) * 1000
                    )

                # Step 5: Estimate heights
                stage_start = time.time()
                height_state = self.height_estimator.estimate_heights(detected_hands)
                height_state.timestamp = timestamp
                if self.performance_monitor:
                    self.performance_monitor.record_latency(
                        "height", (time.time() - stage_start) * 1000
                    )

                # Update current state
                with self._state_lock:
                    self._current_touch_state = touch_state
                    self._current_height_state = height_state

                # Invoke callback if registered
                with self._callback_lock:
                    if self._callback is not None:
                        try:
                            self._callback(touch_state, height_state)
                        except Exception as e:
                            logger.error(f"Error in state update callback: {e}")
                            self._error_count += 1

                # Increment frame counter
                self._frame_count += 1

                # Check processing latency
                frame_latency_ms = (time.time() - frame_start_time) * 1000
                if (
                    frame_latency_ms > self.config.max_processing_latency_ms
                    and time.time() - self._last_warning_time > 5.0
                ):
                    logger.warning(
                        f"High processing latency: {frame_latency_ms:.2f}ms "
                        f"(target: {self.config.max_processing_latency_ms:.2f}ms)"
                    )
                    self._last_warning_time = time.time()

                # Log periodic statistics
                if (
                    self.performance_monitor
                    and time.time() - last_stats_log_time > stats_log_interval
                ):
                    stats = self.performance_monitor.get_stats()
                    logger.info(
                        f"Pipeline stats: {stats}, Hands: {len(detected_hands)}"
                    )
                    last_stats_log_time = time.time()

            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                self._error_count += 1
                # Continue processing despite errors

            # Sleep to maintain target FPS
            frame_elapsed = time.time() - frame_start_time
            if frame_elapsed < target_frame_time:
                time.sleep(target_frame_time - frame_elapsed)

        logger.info("Processing loop terminated")

    def __repr__(self) -> str:
        """String representation of pipeline."""
        return (
            f"VisionPipeline(running={self._running}, "
            f"frames={self._frame_count}, errors={self._error_count})"
        )

    def __enter__(self) -> "VisionPipeline":
        """Context manager entry - start pipeline."""
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop pipeline."""
        self.stop()
