"""
Example usage of the VisionPipeline module.

This demonstrates how to set up and use the complete vision processing
pipeline for ChunIVision.
"""

from chunivision.calibration.calibration_data import CalibrationData
from chunivision.config.camera_config import CameraConfig
from chunivision.config.zone_config import ZoneConfig
from chunivision.vision import (
    VisionPipeline,
    VisionPipelineConfig,
    TouchState,
    HeightState,
)


def state_callback(touch: TouchState, height: HeightState):
    """
    Callback function that receives state updates from the pipeline.

    This is called in the processing thread whenever new state is available.
    """
    # Check which zones are touched
    touched_zones = [i + 1 for i in range(32) if touch.zones[i]]
    if touched_zones:
        print(f"Touched zones: {touched_zones}")

    # Check which height levels are active
    active_levels = [i for i in range(6) if height.levels[i]]
    if active_levels:
        print(f"Active height levels: {active_levels}")


def main():
    """Main example showing VisionPipeline usage."""

    # 1. Load or create configuration
    print("Loading configuration...")

    # Camera configuration (use actual serial numbers from your cameras)
    camera_config = CameraConfig(
        left_camera_serial="LEFT_CAMERA_SERIAL",
        right_camera_serial="RIGHT_CAMERA_SERIAL",
        resolution=(640, 480),
        fps=60,
    )

    # Load calibration data (or use default for testing)
    try:
        calibration_data = CalibrationData.load("configs/calibrations/default.yaml")
    except FileNotFoundError:
        print("Using default calibration (you should calibrate for production use)")
        calibration_data = CalibrationData.create_default()

    # Zone configuration
    zone_config = ZoneConfig()

    # 2. Create pipeline configuration
    print("Creating pipeline configuration...")
    pipeline_config = VisionPipelineConfig(
        camera_config=camera_config,
        calibration_data=calibration_data,
        zone_config=zone_config,
        target_fps=60.0,
        enable_performance_monitoring=True,
        max_processing_latency_ms=10.0,
    )

    # 3. Create and initialize pipeline
    print("Initializing vision pipeline...")
    pipeline = VisionPipeline(pipeline_config)

    if not pipeline.initialize():
        print("Failed to initialize pipeline!")
        return

    # 4. Set callback for state updates
    pipeline.set_callback(state_callback)

    # 5. Start pipeline
    print("Starting vision pipeline...")
    pipeline.start()

    try:
        # Pipeline is now running in background thread
        print("Pipeline running. Press Ctrl+C to stop...")

        # Main loop - could do other work here
        import time

        while True:
            time.sleep(1)

            # Optionally print performance stats
            stats = pipeline.get_stats()
            if stats:
                print(f"Performance: {stats}")

            # Get current state
            touch, height = pipeline.get_current_state()
            if touch and height:
                print(f"Frames processed: {pipeline.get_frame_count()}")

    except KeyboardInterrupt:
        print("\nStopping pipeline...")

    finally:
        # 6. Stop pipeline and cleanup
        pipeline.stop()
        print("Pipeline stopped.")


def example_with_context_manager():
    """Example using VisionPipeline as a context manager."""

    # Create configuration (same as above)
    camera_config = CameraConfig(
        left_camera_serial="LEFT",
        right_camera_serial="RIGHT",
    )
    calibration_data = CalibrationData.create_default()
    zone_config = ZoneConfig()

    pipeline_config = VisionPipelineConfig(
        camera_config=camera_config,
        calibration_data=calibration_data,
        zone_config=zone_config,
    )

    # Use context manager - automatically starts and stops
    with VisionPipeline(pipeline_config) as pipeline:
        pipeline.set_callback(state_callback)

        # Pipeline is running here
        import time

        time.sleep(5)  # Run for 5 seconds

    # Pipeline automatically stopped when exiting context


if __name__ == "__main__":
    # Run the main example
    main()

    # Or use the context manager version:
    # example_with_context_manager()
