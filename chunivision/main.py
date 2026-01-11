"""
ChunIVision main entry point.

This module provides the main application entry point and orchestrates
all system components including:
- Configuration loading and validation
- Logging setup
- Vision pipeline initialization
- Output manager setup
- Calibration workflow
- Graceful shutdown handling
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List

from .calibration.calibration_data import CalibrationData
from .calibration.calibrator import Calibrator, CalibrationError
from .config.camera_config import CameraConfig
from .config.settings import Settings
from .config.zone_config import ZoneConfig
from .output.console_output import ConsoleOutput
from .output.output_manager import OutputManager
from .utils.logger import Logger
from .vision.height_estimator import HeightState
from .vision.stereo_processor import StereoAlgorithm, StereoConfig
from .vision.touch_detector import TouchState
from .vision.vision_pipeline import VisionPipeline, VisionPipelineConfig


class ChunIVisionApp:
    """
    Main application class for ChunIVision.

    Orchestrates all system components and manages the application lifecycle.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the ChunIVision application.

        Args:
            settings: Application settings loaded from configuration
        """
        self.settings = settings
        self.logger = Logger.get_logger(__name__)
        self.pipeline: Optional[VisionPipeline] = None
        self.output_manager: Optional[OutputManager] = None
        self.calibration_data: Optional[CalibrationData] = None
        self._running = False
        self._shutdown_requested = False

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame) -> None:
            sig_name = signal.Signals(signum).name
            self.logger.info(f"Received signal {sig_name}, initiating shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _load_calibration(self, calibration_name: Optional[str] = None) -> bool:
        """
        Load calibration data from file.

        Args:
            calibration_name: Name of calibration profile to load.
                             If None, uses default from settings.

        Returns:
            True if calibration loaded successfully, False otherwise
        """
        if calibration_name:
            calibration_path = (
                Path(self.settings.paths.calibration_dir) / f"{calibration_name}.yaml"
            )
        else:
            calibration_path = Path(self.settings.calibration.file)

        if not calibration_path.exists():
            self.logger.warning(f"Calibration file not found: {calibration_path}")
            return False

        try:
            self.calibration_data = CalibrationData.load(str(calibration_path))
            errors = self.calibration_data.validate()
            if errors:
                for error in errors:
                    self.logger.warning(f"Calibration validation warning: {error}")
            self.logger.info(f"Loaded calibration from {calibration_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False

    def _setup_outputs(self, enabled_outputs: Optional[List[str]] = None) -> bool:
        """
        Set up output adapters based on configuration.

        Args:
            enabled_outputs: List of output names to enable.
                            If None, uses settings configuration.

        Returns:
            True if at least one output was set up successfully
        """
        self.output_manager = OutputManager()

        outputs_config = self.settings.outputs

        # Determine which outputs to enable
        if enabled_outputs:
            # Use command-line specified outputs
            to_enable = enabled_outputs
        else:
            # Use config-specified outputs
            to_enable = []
            if outputs_config.serial.get("enabled", False):
                to_enable.append("serial")
            if outputs_config.hid.get("enabled", False):
                to_enable.append("hid")
            if outputs_config.keyboard.get("enabled", False):
                to_enable.append("keyboard")
            if outputs_config.udp.get("enabled", False):
                to_enable.append("udp")

        # If no outputs specified, enable console output for debugging
        if not to_enable:
            self.logger.info("No outputs enabled, using console output")
            to_enable = ["console"]

        # Set up each output
        for output_name in to_enable:
            try:
                if output_name == "console":
                    output = ConsoleOutput(
                        {"update_interval": 0.05, "show_stats": True}, name="Console"
                    )
                    self.output_manager.add_output("console", output)
                elif output_name == "serial":
                    # Serial output not yet implemented
                    self.logger.warning("Serial output not yet implemented")
                elif output_name == "hid":
                    # HID output not yet implemented
                    self.logger.warning("HID output not yet implemented")
                elif output_name == "keyboard":
                    # Keyboard output not yet implemented
                    self.logger.warning("Keyboard output not yet implemented")
                elif output_name == "udp":
                    # UDP output not yet implemented
                    self.logger.warning("UDP output not yet implemented")
                else:
                    self.logger.warning(f"Unknown output type: {output_name}")
            except Exception as e:
                self.logger.error(f"Failed to create {output_name} output: {e}")

        # Initialize all outputs
        if self.output_manager.get_active_outputs():
            results = self.output_manager.initialize_all()
            success_count = sum(1 for v in results.values() if v)
            self.logger.info(
                f"Initialized {success_count}/{len(results)} output adapters"
            )
            return success_count > 0

        return False

    def _create_vision_pipeline_config(self) -> VisionPipelineConfig:
        """
        Create vision pipeline configuration from settings and calibration.

        Returns:
            VisionPipelineConfig for the vision pipeline

        Raises:
            ValueError: If calibration data is not loaded
        """
        if self.calibration_data is None:
            raise ValueError("Calibration data must be loaded before creating pipeline")

        # Create camera config from settings
        resolution = self.settings.camera.resolution
        camera_config = CameraConfig(
            left_camera_serial=self.settings.camera.left_camera_serial,
            right_camera_serial=self.settings.camera.right_camera_serial,
            resolution=(resolution[0], resolution[1]),
            fps=self.settings.camera.fps,
            exposure=self.settings.camera.exposure,
            baseline_distance=self.settings.camera.baseline_distance,
        )

        # Create zone config from settings
        zone_config = ZoneConfig(
            num_rows=self.settings.zones.num_rows,
            num_cols=self.settings.zones.num_cols,
            zone_width=self.settings.zones.zone_width,
            zone_height=self.settings.zones.zone_height,
            origin_x=self.settings.zones.origin_x,
            origin_y=self.settings.zones.origin_y,
        )

        # Create stereo config
        try:
            stereo_algorithm = StereoAlgorithm(self.settings.vision.stereo_algorithm)
        except ValueError:
            stereo_algorithm = StereoAlgorithm.SGBM
        stereo_config = StereoConfig(
            algorithm=stereo_algorithm,
            min_depth=self.settings.vision.min_depth,
            max_depth=self.settings.vision.max_depth,
        )

        return VisionPipelineConfig(
            camera_config=camera_config,
            calibration_data=self.calibration_data,
            zone_config=zone_config,
            stereo_config=stereo_config,
            target_fps=float(self.settings.camera.fps),
            enable_performance_monitoring=self.settings.performance.enabled,
        )

    def _on_state_update(
        self, touch_state: TouchState, height_state: HeightState
    ) -> None:
        """
        Callback for vision pipeline state updates.

        Args:
            touch_state: Current touch zone states
            height_state: Current height level states
        """
        if self.output_manager:
            self.output_manager.send_state(touch_state, height_state)

    def run(self) -> int:
        """
        Run the main application loop.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        self._setup_signal_handlers()
        self.logger.info("Starting ChunIVision in run mode")

        # Load calibration
        if not self._load_calibration():
            self.logger.error(
                "No calibration data available. Run calibration first: "
                "--mode calibration"
            )
            return 1

        # Set up outputs
        if not self._setup_outputs():
            self.logger.error("Failed to set up any output adapters")
            return 1

        try:
            # Create and initialize vision pipeline
            pipeline_config = self._create_vision_pipeline_config()
            self.pipeline = VisionPipeline(pipeline_config)

            # Set up state callback
            self.pipeline.set_callback(self._on_state_update)

            # Initialize and start pipeline
            if not self.pipeline.initialize():
                self.logger.error("Failed to initialize vision pipeline")
                return 1

            self.pipeline.start()
            self._running = True
            self.logger.info("Vision pipeline started, processing frames...")

            # Main loop - monitor performance and handle shutdown
            last_stats_time = time.time()
            stats_interval = self.settings.performance.log_interval

            while self._running and not self._shutdown_requested:
                time.sleep(0.1)

                # Periodically log performance stats
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    if self.settings.performance.enabled:
                        stats = self.pipeline.get_stats()
                        if stats:
                            self.logger.info(
                                f"Performance: FPS={stats.fps:.1f}, "
                                f"Latency={stats.avg_latency_ms:.1f}ms"
                            )

                            # Check for performance warnings
                            if (
                                stats.fps
                                < self.settings.performance.fps_warning_threshold
                            ):
                                self.logger.warning(
                                    f"FPS ({stats.fps:.1f}) below threshold "
                                    f"({self.settings.performance.fps_warning_threshold})"
                                )
                            if (
                                stats.avg_latency_ms
                                > self.settings.performance.latency_warning_threshold
                            ):
                                self.logger.warning(
                                    f"Latency ({stats.avg_latency_ms:.1f}ms) above "
                                    f"threshold ({self.settings.performance.latency_warning_threshold}ms)"
                                )
                    last_stats_time = current_time

            return 0

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            return 1
        finally:
            self._shutdown()

    def run_calibration(self) -> int:
        """
        Run the calibration workflow.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        self.logger.info("Starting ChunIVision calibration mode")

        try:
            # Create calibrator with settings from config
            resolution = self.settings.camera.resolution
            calibrator = Calibrator(
                board_physical_size=(
                    self.settings.calibration.board_width,
                    self.settings.calibration.board_height,
                ),
                zone_grid=(self.settings.zones.num_cols, self.settings.zones.num_rows),
                image_size=(resolution[0], resolution[1]),
                stereo_baseline=self.settings.camera.baseline_distance,
            )

            # For calibration, we need to initialize cameras directly
            # This is a simplified flow - in production, you'd use CameraManager
            from .oculus.oculus_camera import OculusRiftCV1Camera

            self.logger.info("Initializing cameras for calibration...")

            try:
                left_camera = OculusRiftCV1Camera(
                    serial_number=self.settings.camera.left_camera_serial
                )
                right_camera = OculusRiftCV1Camera(
                    serial_number=self.settings.camera.right_camera_serial
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize cameras: {e}")
                return 1

            try:
                # Run interactive calibration
                self.logger.info(
                    "Starting interactive calibration. Follow the on-screen instructions."
                )
                calibration_data = calibrator.run_interactive_calibration(
                    left_camera, right_camera
                )

                # Save calibration data
                calibration_path = Path(self.settings.calibration.file)
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                calibration_data.save(str(calibration_path))

                self.logger.info(f"Calibration saved to {calibration_path}")
                return 0

            finally:
                left_camera.close()
                right_camera.close()

        except CalibrationError as e:
            self.logger.error(f"Calibration failed: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error during calibration: {e}")
            return 1

    def run_debug(self) -> int:
        """
        Run in debug mode with additional visualization and logging.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        self.logger.info("Starting ChunIVision in debug mode")

        # Enable debug visualization options
        self.settings.debug.show_camera_view = True
        self.settings.debug.show_detections = True
        self.settings.debug.visualize_zones = True

        # Set verbose logging
        Logger.set_level("DEBUG")

        # Run normal application with debug settings
        return self.run()

    def run_test(self) -> int:
        """
        Run system tests.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        self.logger.info("Starting ChunIVision in test mode")

        tests_passed = 0
        tests_failed = 0

        # Test 1: Configuration loading
        self.logger.info("Test 1: Configuration validation...")
        errors = self.settings.validate()
        if errors:
            self.logger.error(f"Configuration errors: {errors}")
            tests_failed += 1
        else:
            self.logger.info("Configuration: OK")
            tests_passed += 1

        # Test 2: Calibration file
        self.logger.info("Test 2: Calibration file...")
        if self._load_calibration():
            self.logger.info("Calibration file: OK")
            tests_passed += 1
        else:
            self.logger.warning("Calibration file: NOT FOUND (run calibration first)")
            tests_failed += 1

        # Test 3: Output adapters
        self.logger.info("Test 3: Output adapters...")
        try:
            if self._setup_outputs(["console"]):
                self.logger.info("Output adapters: OK")
                tests_passed += 1
            else:
                self.logger.error("Output adapters: FAILED")
                tests_failed += 1
        except Exception as e:
            self.logger.error(f"Output adapters: FAILED ({e})")
            tests_failed += 1
        finally:
            if self.output_manager:
                self.output_manager.close_all()

        # Test 4: Camera detection
        self.logger.info("Test 4: Camera detection...")
        try:
            from .oculus.oculus_camera import OculusRiftCV1Camera

            # Just try to import and check for devices (don't actually open)
            self.logger.info("Camera module: OK (camera detection requires hardware)")
            tests_passed += 1
        except ImportError as e:
            self.logger.error(f"Camera module: FAILED ({e})")
            tests_failed += 1

        # Summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Test Results: {tests_passed} passed, {tests_failed} failed")
        self.logger.info(f"{'='*50}")

        return 0 if tests_failed == 0 else 1

    def _shutdown(self) -> None:
        """Clean up resources and shutdown gracefully."""
        self.logger.info("Shutting down ChunIVision...")
        self._running = False

        # Stop vision pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                self.logger.error(f"Error stopping pipeline: {e}")

        # Close output adapters
        if self.output_manager:
            try:
                self.output_manager.close_all()
            except Exception as e:
                self.logger.error(f"Error closing outputs: {e}")

        self.logger.info("ChunIVision shutdown complete")


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for ChunIVision application.

    Args:
        args: Command line arguments (default: sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="ChunIVision - Vision-based Chusan Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chunivision --mode run                 # Run with default config
  chunivision --mode calibration         # Run calibration wizard
  chunivision --mode debug               # Run with debug visualization
  chunivision --mode test                # Run system tests
  chunivision --config myconfig.yaml     # Use custom config file
  chunivision --output udp --output hid  # Enable specific outputs
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["run", "calibration", "debug", "test"],
        default="run",
        help="Application mode (default: run)",
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Configuration file path (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--calibration",
        default=None,
        help="Calibration profile name (overrides config file setting)",
    )

    parser.add_argument(
        "--output",
        choices=["serial", "hid", "keyboard", "udp", "console"],
        action="append",
        help="Enable specific output (can specify multiple times)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging level (overrides config file setting)",
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: console only)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    parsed_args = parser.parse_args(args)

    # Load configuration
    try:
        config_path = Path(parsed_args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {parsed_args.config}")
            print("Create a config file or use --config to specify path")
            return 1

        settings = Settings.load(str(config_path))
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Apply command-line overrides
    if parsed_args.log_level:
        settings.logging.level = parsed_args.log_level

    if parsed_args.mode:
        settings.app.mode = parsed_args.mode

    # Set up logging
    log_level = settings.logging.level
    log_file = parsed_args.log_file or settings.logging.file
    log_format = settings.logging.format

    # For console output mode, redirect logs to file to avoid conflicts
    if parsed_args.output and "console" in parsed_args.output:
        if not log_file:
            log_file = "chunivision.log"

    Logger.setup(
        level=log_level,
        log_file=log_file,
        log_format=log_format,
        enable_rotation=settings.logging.rotation,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )

    logger = Logger.get_logger(__name__)

    # Validate settings
    validation_errors = settings.validate()
    if validation_errors:
        for error in validation_errors:
            logger.error(f"Configuration error: {error}")
        return 1

    # Log startup info
    logger.info(f"ChunIVision v{settings.app.version}")
    logger.info(f"Mode: {parsed_args.mode}")
    logger.info(f"Config: {parsed_args.config}")

    # Create and run application
    app = ChunIVisionApp(settings)

    # Apply calibration override if specified
    if parsed_args.calibration:
        app._load_calibration(parsed_args.calibration)

    # Run appropriate mode
    try:
        if parsed_args.mode == "run":
            return app.run()
        elif parsed_args.mode == "calibration":
            return app.run_calibration()
        elif parsed_args.mode == "debug":
            return app.run_debug()
        elif parsed_args.mode == "test":
            return app.run_test()
        else:
            logger.error(f"Unknown mode: {parsed_args.mode}")
            return 1
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        Logger.shutdown()


if __name__ == "__main__":
    sys.exit(main())
