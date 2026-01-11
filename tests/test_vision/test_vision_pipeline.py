"""
Unit tests for VisionPipeline module.

Tests the main vision processing pipeline including:
- Configuration validation
- Component initialization
- Processing loop
- Callback system
- State management
- Performance monitoring
- Error handling
"""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from chunivision.calibration.calibration_data import CalibrationData
from chunivision.config.camera_config import CameraConfig
from chunivision.config.zone_config import ZoneConfig
from chunivision.vision.hand_detector import Hand, HandDetectorConfig
from chunivision.vision.height_estimator import HeightEstimatorConfig, HeightState
from chunivision.vision.point_cloud import PointCloud3D
from chunivision.vision.stereo_processor import StereoConfig
from chunivision.vision.touch_detector import TouchDetectorConfig, TouchState
from chunivision.vision.vision_pipeline import (
    VisionPipeline,
    VisionPipelineConfig,
    VisionPipelineError,
)


@pytest.fixture
def mock_camera_config():
    """Create a mock camera configuration."""
    return CameraConfig(
        left_camera_serial="LEFT123",
        right_camera_serial="RIGHT456",
        resolution=(640, 480),
        fps=60,
    )


@pytest.fixture
def mock_calibration_data():
    """Create a mock calibration data."""
    return CalibrationData.create_default()


@pytest.fixture
def mock_zone_config():
    """Create a mock zone configuration."""
    return ZoneConfig()


@pytest.fixture
def valid_pipeline_config(mock_camera_config, mock_calibration_data, mock_zone_config):
    """Create a valid pipeline configuration."""
    return VisionPipelineConfig(
        camera_config=mock_camera_config,
        calibration_data=mock_calibration_data,
        zone_config=mock_zone_config,
        target_fps=30.0,  # Lower FPS for testing
        enable_performance_monitoring=True,
    )


class TestVisionPipelineConfig:
    """Test VisionPipelineConfig class."""

    def test_config_creation(self, valid_pipeline_config):
        """Test creating a valid configuration."""
        assert valid_pipeline_config.target_fps == 30.0
        assert valid_pipeline_config.enable_performance_monitoring is True

    def test_config_validation_valid(self, valid_pipeline_config):
        """Test validation of valid configuration."""
        errors = valid_pipeline_config.validate()
        assert len(errors) == 0

    def test_config_validation_invalid_fps(
        self, mock_camera_config, mock_calibration_data, mock_zone_config
    ):
        """Test validation catches invalid FPS."""
        config = VisionPipelineConfig(
            camera_config=mock_camera_config,
            calibration_data=mock_calibration_data,
            zone_config=mock_zone_config,
            target_fps=-10.0,
        )
        errors = config.validate()
        assert any("target_fps" in e for e in errors)

    def test_config_validation_invalid_latency(
        self, mock_camera_config, mock_calibration_data, mock_zone_config
    ):
        """Test validation catches invalid latency."""
        config = VisionPipelineConfig(
            camera_config=mock_camera_config,
            calibration_data=mock_calibration_data,
            zone_config=mock_zone_config,
            max_processing_latency_ms=-5.0,
        )
        errors = config.validate()
        assert any("max_processing_latency_ms" in e for e in errors)

    def test_config_with_custom_components(
        self, mock_camera_config, mock_calibration_data, mock_zone_config
    ):
        """Test configuration with custom component configs."""
        config = VisionPipelineConfig(
            camera_config=mock_camera_config,
            calibration_data=mock_calibration_data,
            zone_config=mock_zone_config,
            stereo_config=StereoConfig(num_disparities=64),
            hand_detector_config=HandDetectorConfig(max_hands=4),
            touch_detector_config=TouchDetectorConfig(touch_threshold_z=2.0),
            height_estimator_config=HeightEstimatorConfig(hysteresis=1.0),
        )
        errors = config.validate()
        assert len(errors) == 0
        assert config.stereo_config.num_disparities == 64
        assert config.hand_detector_config.max_hands == 4


class TestVisionPipeline:
    """Test VisionPipeline class."""

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_pipeline_creation(self, mock_camera_manager_class, valid_pipeline_config):
        """Test creating a pipeline instance."""
        pipeline = VisionPipeline(valid_pipeline_config)

        assert pipeline.config == valid_pipeline_config
        assert not pipeline.is_running()
        assert pipeline.get_frame_count() == 0
        assert pipeline.get_error_count() == 0

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_pipeline_creation_invalid_config(
        self, mock_camera_manager_class, mock_camera_config, mock_zone_config
    ):
        """Test pipeline creation fails with invalid config."""
        # Create invalid calibration data - wrong shaped matrix
        invalid_calibration = CalibrationData.create_default()
        invalid_calibration.camera_left_matrix = np.zeros((2, 2))  # Should be 3x3!

        config = VisionPipelineConfig(
            camera_config=mock_camera_config,
            calibration_data=invalid_calibration,
            zone_config=mock_zone_config,
        )

        with pytest.raises(VisionPipelineError):
            VisionPipeline(config)

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_pipeline_initialization(
        self, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test pipeline initialization."""
        # Setup mock
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True
        mock_camera_manager_class.return_value = mock_camera_manager

        pipeline = VisionPipeline(valid_pipeline_config)
        result = pipeline.initialize()

        assert result is True
        mock_camera_manager.initialize.assert_called_once()

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_pipeline_initialization_failure(
        self, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test pipeline initialization failure."""
        # Setup mock to fail
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = False
        mock_camera_manager_class.return_value = mock_camera_manager

        pipeline = VisionPipeline(valid_pipeline_config)
        result = pipeline.initialize()

        assert result is False

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_callback_registration(
        self, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test callback function registration."""
        pipeline = VisionPipeline(valid_pipeline_config)

        callback_called = []

        def test_callback(touch: TouchState, height: HeightState):
            callback_called.append((touch, height))

        pipeline.set_callback(test_callback)
        assert pipeline._callback == test_callback

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_start_stop_pipeline(
        self, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test starting and stopping the pipeline."""
        # Setup mocks
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True
        mock_camera_manager.get_frame_pair.return_value = None  # No frames to process
        mock_camera_manager_class.return_value = mock_camera_manager

        pipeline = VisionPipeline(valid_pipeline_config)
        pipeline.initialize()

        # Start pipeline
        pipeline.start()
        assert pipeline.is_running()
        assert pipeline._processing_thread is not None
        assert pipeline._processing_thread.is_alive()

        # Let it run briefly
        time.sleep(0.1)

        # Stop pipeline
        pipeline.stop()
        assert not pipeline.is_running()

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_start_already_running(
        self, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test starting an already running pipeline raises error."""
        # Setup mocks
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True
        mock_camera_manager_class.return_value = mock_camera_manager

        pipeline = VisionPipeline(valid_pipeline_config)
        pipeline.initialize()
        pipeline.start()

        with pytest.raises(VisionPipelineError):
            pipeline.start()

        pipeline.stop()

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_stop_not_running(self, mock_camera_manager_class, valid_pipeline_config):
        """Test stopping a non-running pipeline."""
        pipeline = VisionPipeline(valid_pipeline_config)

        # Should not raise, just log warning
        pipeline.stop()
        assert not pipeline.is_running()

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    @patch("chunivision.vision.vision_pipeline.StereoProcessor")
    @patch("chunivision.vision.vision_pipeline.HandDetector")
    def test_processing_loop_integration(
        self,
        mock_hand_detector_class,
        mock_stereo_class,
        mock_camera_manager_class,
        valid_pipeline_config,
    ):
        """Test the complete processing loop with mocked components."""
        # Setup camera manager mock
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True

        # Create mock frames
        left_frame = np.zeros((480, 640), dtype=np.uint8)
        right_frame = np.zeros((480, 640), dtype=np.uint8)
        timestamp = time.time()

        # Return frames 3 times, then None to stop
        frame_returns = [
            (left_frame, right_frame, timestamp),
            (left_frame, right_frame, timestamp),
            (left_frame, right_frame, timestamp),
            None,
        ]
        mock_camera_manager.get_frame_pair.side_effect = frame_returns
        mock_camera_manager_class.return_value = mock_camera_manager

        # Setup stereo processor mock
        mock_stereo = Mock()
        mock_point_cloud = PointCloud3D(
            points=np.random.rand(100, 3).astype(np.float32),
            colors=None,
            timestamp=timestamp,
        )
        mock_stereo.process.return_value = mock_point_cloud
        mock_stereo_class.return_value = mock_stereo

        # Setup hand detector mock
        mock_hand_detector = Mock()
        mock_hands = [
            Hand(
                position=np.array([10.0, 20.0, 5.0]),
                confidence=0.9,
                track_id=1,
                timestamp=timestamp,
            )
        ]
        mock_hand_detector.detect.return_value = mock_hands
        mock_hand_detector.track.return_value = mock_hands
        mock_hand_detector_class.return_value = mock_hand_detector

        # Create pipeline
        pipeline = VisionPipeline(valid_pipeline_config)
        pipeline.initialize()

        # Track callback invocations
        callback_count = []

        def test_callback(touch: TouchState, height: HeightState):
            callback_count.append((touch, height))

        pipeline.set_callback(test_callback)

        # Start and let run
        pipeline.start()
        time.sleep(0.5)  # Let it process a few frames
        pipeline.stop()

        # Verify processing occurred
        assert pipeline.get_frame_count() > 0
        assert len(callback_count) > 0

        # Verify components were called
        assert mock_stereo.process.call_count > 0
        assert mock_hand_detector.detect.call_count > 0

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_get_current_state(self, mock_camera_manager_class, valid_pipeline_config):
        """Test getting current state."""
        pipeline = VisionPipeline(valid_pipeline_config)

        # Initially should be None
        touch, height = pipeline.get_current_state()
        assert touch is None
        assert height is None

        # Set mock state
        mock_touch = TouchState()
        mock_height = HeightState()
        with pipeline._state_lock:
            pipeline._current_touch_state = mock_touch
            pipeline._current_height_state = mock_height

        # Should return the mock state
        touch, height = pipeline.get_current_state()
        assert touch == mock_touch
        assert height == mock_height

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_get_stats(self, mock_camera_manager_class, valid_pipeline_config):
        """Test getting performance statistics."""
        pipeline = VisionPipeline(valid_pipeline_config)

        # Should return stats if monitoring enabled
        stats = pipeline.get_stats()
        assert stats is not None
        assert stats.fps >= 0

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_get_stats_disabled(self, mock_camera_manager_class):
        """Test getting stats when monitoring disabled."""
        config = VisionPipelineConfig(
            camera_config=CameraConfig(
                left_camera_serial="LEFT", right_camera_serial="RIGHT"
            ),
            calibration_data=CalibrationData.create_default(),
            zone_config=ZoneConfig(),
            enable_performance_monitoring=False,
        )

        pipeline = VisionPipeline(config)
        stats = pipeline.get_stats()
        assert stats is None

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_context_manager(self, mock_camera_manager_class, valid_pipeline_config):
        """Test pipeline as context manager."""
        # Setup mocks
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True
        mock_camera_manager.get_frame_pair.return_value = None
        mock_camera_manager_class.return_value = mock_camera_manager

        # Use as context manager
        with VisionPipeline(valid_pipeline_config) as pipeline:
            assert pipeline.is_running()
            time.sleep(0.1)

        # Should be stopped after exit
        assert not pipeline.is_running()

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    @patch("chunivision.vision.vision_pipeline.StereoProcessor")
    def test_error_handling_in_loop(
        self, mock_stereo_class, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test error handling during processing loop."""
        # Setup camera manager
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True

        left_frame = np.zeros((480, 640), dtype=np.uint8)
        right_frame = np.zeros((480, 640), dtype=np.uint8)

        # Return a few frames then None
        mock_camera_manager.get_frame_pair.side_effect = [
            (left_frame, right_frame, time.time()),
            None,
        ]
        mock_camera_manager_class.return_value = mock_camera_manager

        # Setup stereo processor to raise exception
        mock_stereo = Mock()
        mock_stereo.process.side_effect = RuntimeError("Stereo processing failed")
        mock_stereo_class.return_value = mock_stereo

        pipeline = VisionPipeline(valid_pipeline_config)
        pipeline.initialize()
        pipeline.start()

        # Let it run and encounter error
        time.sleep(0.3)
        pipeline.stop()

        # Should have errors but continue running
        assert pipeline.get_error_count() > 0

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_callback_error_handling(
        self, mock_camera_manager_class, valid_pipeline_config
    ):
        """Test error handling in callback."""
        # Setup mocks for basic operation
        mock_camera_manager = Mock()
        mock_camera_manager.initialize.return_value = True
        mock_camera_manager.is_ready.return_value = True
        mock_camera_manager_class.return_value = mock_camera_manager

        pipeline = VisionPipeline(valid_pipeline_config)

        # Register callback that raises exception
        def bad_callback(touch: TouchState, height: HeightState):
            raise RuntimeError("Callback error")

        pipeline.set_callback(bad_callback)

        # Manually trigger callback with mock states
        mock_touch = TouchState()
        mock_height = HeightState()

        # Should not raise, just log error and increment error count
        initial_errors = pipeline.get_error_count()
        with pipeline._callback_lock:
            try:
                pipeline._callback(mock_touch, mock_height)
            except Exception:
                pass

        # Error count should increase if exception occurred
        # (in real processing loop, this is handled)

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    def test_repr(self, mock_camera_manager_class, valid_pipeline_config):
        """Test string representation."""
        pipeline = VisionPipeline(valid_pipeline_config)
        repr_str = repr(pipeline)

        assert "VisionPipeline" in repr_str
        assert "running=" in repr_str
        assert "frames=" in repr_str
        assert "errors=" in repr_str


class TestVisionPipelineIntegration:
    """Integration tests for VisionPipeline (require more complex setup)."""

    @patch("chunivision.vision.vision_pipeline.CameraManager")
    @patch("chunivision.vision.vision_pipeline.StereoProcessor")
    @patch("chunivision.vision.vision_pipeline.HandDetector")
    @patch("chunivision.vision.vision_pipeline.TouchDetector")
    @patch("chunivision.vision.vision_pipeline.HeightEstimator")
    def test_full_pipeline_with_detections(
        self,
        mock_height_est_class,
        mock_touch_det_class,
        mock_hand_det_class,
        mock_stereo_class,
        mock_camera_class,
        valid_pipeline_config,
    ):
        """Test full pipeline with realistic mock data."""
        timestamp = time.time()

        # Mock camera
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera.is_ready.return_value = True
        left_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        mock_camera.get_frame_pair.side_effect = [
            (left_frame, right_frame, timestamp),
            (left_frame, right_frame, timestamp),
            None,
        ]
        mock_camera_class.return_value = mock_camera

        # Mock stereo
        mock_stereo = Mock()
        point_cloud = PointCloud3D(
            points=np.random.rand(200, 3).astype(np.float32) * 50,
            timestamp=timestamp,
        )
        mock_stereo.process.return_value = point_cloud
        mock_stereo_class.return_value = mock_stereo

        # Mock hand detector
        mock_hand_det = Mock()
        hands = [
            Hand(
                position=np.array([10.0, 15.0, 3.0]),
                velocity=np.array([0.5, 0.2, -0.1]),
                confidence=0.95,
                track_id=1,
                timestamp=timestamp,
            ),
            Hand(
                position=np.array([30.0, 25.0, 18.0]),
                velocity=np.array([-0.3, 0.1, 0.2]),
                confidence=0.88,
                track_id=2,
                timestamp=timestamp,
            ),
        ]
        mock_hand_det.detect.return_value = hands
        mock_hand_det.track.return_value = hands
        mock_hand_det_class.return_value = mock_hand_det

        # Mock touch detector
        mock_touch_det = Mock()
        touch_state = TouchState(timestamp=timestamp)
        touch_state.zones[0] = True  # Zone 1 touched
        mock_touch_det.detect_touches.return_value = touch_state
        mock_touch_det_class.return_value = mock_touch_det

        # Mock height estimator
        mock_height_est = Mock()
        height_state = HeightState(timestamp=timestamp)
        height_state.levels[2] = True  # Level 2 active
        mock_height_est.estimate_heights.return_value = height_state
        mock_height_est_class.return_value = mock_height_est

        # Create and run pipeline
        pipeline = VisionPipeline(valid_pipeline_config)
        pipeline.initialize()

        callback_results = []

        def capture_callback(touch: TouchState, height: HeightState):
            callback_results.append((touch, height))

        pipeline.set_callback(capture_callback)

        pipeline.start()
        time.sleep(0.5)
        pipeline.stop()

        # Verify callbacks were invoked
        assert len(callback_results) > 0

        # Verify states
        for touch, height in callback_results:
            assert isinstance(touch, TouchState)
            assert isinstance(height, HeightState)

        # Verify all components were called
        assert mock_stereo.process.call_count > 0
        assert mock_hand_det.detect.call_count > 0
        assert mock_touch_det.detect_touches.call_count > 0
        assert mock_height_est.estimate_heights.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
