# Development Guidelines

This document provides guidelines for developing and extending the ChunIVision system.

## Code Style and Conventions

### Python Style

Follow PEP 8 with these specific conventions:

- **Indentation**: 4 spaces (no tabs)
- **Line Length**: 100 characters maximum
- **Imports**: Group in order: standard library, third-party, local modules
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Google style docstrings for all public classes and functions

**Example:**

```python
from typing import Optional, Tuple, List
import numpy as np
from .camera_manager import CameraManager


class VisionPipeline:
    """
    Main vision processing pipeline coordinator.
    
    This class orchestrates all vision processing components including
    camera capture, stereo processing, hand detection, and state management.
    
    Attributes:
        camera_manager: Manages dual camera system
        is_running: Whether pipeline is currently active
    """
    
    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize vision pipeline with configuration.
        
        Args:
            config: Pipeline configuration parameters
            
        Raises:
            InitializationError: If pipeline cannot be initialized
        """
        self.camera_manager = CameraManager(config.camera_config)
        self.is_running = False
        
    def process_frame(self, 
                      left_frame: np.ndarray, 
                      right_frame: np.ndarray) -> Tuple[TouchState, HeightState]:
        """
        Process a stereo frame pair.
        
        Args:
            left_frame: Left camera frame (H, W) numpy array
            right_frame: Right camera frame (H, W) numpy array
            
        Returns:
            Tuple of (touch_state, height_state) representing detected state
            
        Raises:
            ProcessingError: If frame processing fails
        """
        # Implementation...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `CameraManager`, `TouchDetector`)
- **Functions/Methods**: `snake_case` (e.g., `get_frame_pair`, `detect_hands`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_FPS`, `DEFAULT_THRESHOLD`)
- **Private Members**: Prefix with `_` (e.g., `_internal_buffer`)
- **Type Aliases**: `PascalCase` (e.g., `FramePair = Tuple[np.ndarray, np.ndarray]`)

### File Organization

Each module file should follow this structure:

```python
"""
Module docstring explaining purpose and usage.
"""

# Standard library imports
import os
import sys
from typing import Optional

# Third-party imports
import numpy as np
import cv2

# Local imports
from ..config import Settings
from .camera_manager import CameraManager

# Constants
DEFAULT_FPS = 60
MAX_RETRIES = 3

# Type aliases
FramePair = Tuple[np.ndarray, np.ndarray]

# Classes and functions
class MyClass:
    """Class implementation."""
    pass


def my_function() -> None:
    """Function implementation."""
    pass
```

---

## Testing

### Unit Tests

All modules must have corresponding unit tests in `tests/` directory.

**Test File Structure:**

```python
import unittest
import numpy as np
from unittest.mock import Mock, patch
from chunivision.vision.hand_detector import HandDetector, Hand


class TestHandDetector(unittest.TestCase):
    """Unit tests for HandDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HandDetector(config=Mock())
        
    def test_detect_single_hand(self):
        """Test detection of a single hand in point cloud."""
        # Create mock point cloud with one hand-sized cluster
        point_cloud = self._create_mock_point_cloud(num_points=500)
        
        hands = self.detector.detect(point_cloud)
        
        self.assertEqual(len(hands), 1)
        self.assertIsInstance(hands[0], Hand)
        self.assertGreater(hands[0].confidence, 0.8)
        
    def test_detect_no_hands(self):
        """Test detection with empty point cloud."""
        point_cloud = self._create_mock_point_cloud(num_points=0)
        
        hands = self.detector.detect(point_cloud)
        
        self.assertEqual(len(hands), 0)
        
    def _create_mock_point_cloud(self, num_points: int):
        """Helper to create mock point cloud for testing."""
        # Implementation...
```

**Running Tests:**

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_vision/test_hand_detector.py

# Run with coverage
python -m pytest --cov=chunivision tests/
```

### Integration Tests

Integration tests verify module interactions:

```python
class TestVisionPipeline(unittest.TestCase):
    """Integration tests for complete vision pipeline."""
    
    @patch('chunivision.vision.camera_manager.Camera')
    def test_full_pipeline(self, mock_camera):
        """Test end-to-end pipeline processing."""
        # Set up mock cameras
        mock_camera.return_value.capture.return_value = self._create_test_frame()
        
        # Initialize pipeline
        pipeline = VisionPipeline(config)
        pipeline.start()
        
        # Capture state update
        states = []
        pipeline.set_callback(lambda t, h: states.append((t, h)))
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify state updates
        self.assertGreater(len(states), 0)
        
        pipeline.stop()
```

### Performance Tests

Benchmark critical paths:

```python
import time

def test_hand_detection_performance():
    """Verify hand detection meets performance target (<5ms)."""
    detector = HandDetector(config)
    point_cloud = create_realistic_point_cloud()
    
    # Warmup
    for _ in range(10):
        detector.detect(point_cloud)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        detector.detect(point_cloud)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / 100) * 1000
    assert avg_time_ms < 5.0, f"Detection too slow: {avg_time_ms:.2f}ms"
```

---

## Adding New Features

### Adding a New Output Adapter

1. **Create new file**: `chunivision/output/my_output.py`

2. **Implement BaseOutput interface**:

```python
from .base_output import BaseOutput
from ..utils.state_manager import TouchState, HeightState


class MyOutput(BaseOutput):
    """
    Output adapter for my custom protocol.
    
    Sends state updates via [describe your protocol].
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.connection = None
        
    def initialize(self) -> bool:
        """Initialize connection to output destination."""
        try:
            self.connection = establish_connection(self.config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
            
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> bool:
        """Send state update via my protocol."""
        try:
            packet = self._create_packet(touch_state, height_state)
            self.connection.send(packet)
            return True
        except Exception as e:
            logger.error(f"Failed to send: {e}")
            return False
            
    def close(self) -> None:
        """Clean up connection."""
        if self.connection:
            self.connection.close()
            
    def _create_packet(self, touch_state, height_state):
        """Create protocol-specific packet."""
        # Implementation...
```

1. **Add configuration schema** in `configs/default.yaml`:

```yaml
outputs:
  my_output:
    enabled: true
    parameter1: value1
    parameter2: value2
```

1. **Register in output manager** (`output/output_manager.py`):

```python
from .my_output import MyOutput

# In OutputManager.load_from_config():
if config.get('my_output', {}).get('enabled'):
    output = MyOutput(config['my_output'])
    self.add_output('my_output', output)
```

1. **Write tests** in `tests/test_output/test_my_output.py`

2. **Document protocol** in `docs/OUTPUT_PROTOCOLS.md`

### Adding a New Vision Processor

To add a new component to the vision pipeline (e.g., gesture recognizer):

1. **Create module**: `chunivision/vision/gesture_recognizer.py`

```python
from typing import List
from .hand_detector import Hand


class Gesture:
    """Represents a detected gesture."""
    gesture_type: str  # "swipe_left", "swipe_right", etc.
    confidence: float
    timestamp: float


class GestureRecognizer:
    """
    Recognizes gestures from hand movements.
    """
    
    def __init__(self, config):
        self.config = config
        self.hand_history = []
        
    def recognize(self, hands: List[Hand]) -> List[Gesture]:
        """
        Recognize gestures from current and historical hand positions.
        
        Args:
            hands: Currently detected hands
            
        Returns:
            List of recognized gestures
        """
        self.hand_history.append(hands)
        
        # Keep limited history
        if len(self.hand_history) > 30:  # 0.5s at 60fps
            self.hand_history.pop(0)
            
        gestures = []
        gestures.extend(self._detect_swipes())
        # Add more gesture types...
        
        return gestures
```

1. **Integrate into pipeline** (`vision/vision_pipeline.py`):

```python
from .gesture_recognizer import GestureRecognizer

class VisionPipeline:
    def __init__(self, ...):
        # ... existing initialization
        self.gesture_recognizer = GestureRecognizer(config.gesture_config)
        
    def _process_loop(self):
        while self.is_running:
            # ... existing processing
            hands = self.hand_detector.detect(point_cloud)
            touch_state = self.touch_detector.detect_touches(hands)
            height_state = self.height_estimator.estimate_heights(hands)
            
            # Add gesture recognition
            gestures = self.gesture_recognizer.recognize(hands)
            
            # Notify callbacks
            self._notify_state_callback(touch_state, height_state, gestures)
```

1. **Update configuration schema**

2. **Add tests and documentation**

---

## Debugging

### Logging

Use the centralized logger:

```python
from chunivision.utils.logger import Logger

logger = Logger.get_logger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)  # Include traceback
```

### Visualization Tools

For debugging vision processing:

```python
from chunivision.utils.visualization import Visualizer

viz = Visualizer()

# Display depth map
viz.show_depth_map(depth_map, title="Stereo Depth")

# Overlay detections on frame
viz.overlay_hands(frame, hands)
viz.overlay_zones(frame, touch_state, zone_config)

# Show point cloud
viz.show_point_cloud(point_cloud)
```

### Performance Profiling

```python
from chunivision.utils.performance import PerformanceMonitor, profile

monitor = PerformanceMonitor()

@profile(monitor, "hand_detection")
def detect_hands(point_cloud):
    # Function is automatically timed
    pass

# Later, get statistics
stats = monitor.get_stats()
print(f"Hand detection avg: {stats['hand_detection']['avg_ms']:.2f}ms")
```

---

## Configuration Management

### YAML Configuration Files

All configuration in `configs/` directory:

```yaml
# configs/default.yaml
version: "1.0"

camera:
  left_index: 0
  right_index: 1
  resolution: [640, 480]
  fps: 60

vision:
  touch_threshold_z: 2.0  # cm
  min_hand_size: 100  # points
  max_hands: 10

output:
  enabled_outputs:
    - udp
    - serial
```

### Loading Configuration

```python
from chunivision.config.settings import Settings

# Load configuration
settings = Settings.load('configs/default.yaml')

# Access values
camera_config = settings.camera
fps = settings.camera.fps
```

### Environment Variables

Override config via environment variables:

```bash
export CHUNIVISION_CAMERA_FPS=120
export CHUNIVISION_LOG_LEVEL=DEBUG
python -m chunivision.main
```

---

## Continuous Integration

### Pre-commit Checks

Run before committing:

```bash
# Format code
black chunivision/

# Check style
flake8 chunivision/

# Type checking
mypy chunivision/

# Run tests
pytest tests/
```

### GitHub Actions

Automated CI pipeline (`.github/workflows/ci.yml`):

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -r requirements.txt
      - run: pytest --cov=chunivision tests/
      - run: flake8 chunivision/
      - run: mypy chunivision/
```

---

## Best Practices

### Error Handling

- **Fail gracefully**: Log errors and continue when possible
- **Validate inputs**: Check types and ranges at function boundaries
- **Use specific exceptions**: Define custom exception classes for different error types
- **Clean up resources**: Use context managers (`with` statements) where possible

```python
class CameraInitError(Exception):
    """Raised when camera initialization fails."""
    pass


class CameraManager:
    def initialize(self) -> bool:
        try:
            self.camera = Camera(self.config.index)
        except CameraNotFoundError as e:
            logger.error(f"Camera not found: {e}")
            raise CameraInitError(f"Failed to initialize camera {self.config.index}") from e
        
        return True
```

### Thread Safety

- **Minimize shared state**: Use message passing instead of shared memory
- **Lock critical sections**: Use `threading.Lock` for shared data
- **Use thread-safe queues**: `queue.Queue` for inter-thread communication

```python
import queue
import threading

class VisionPipeline:
    def __init__(self):
        self.output_queue = queue.Queue(maxsize=10)
        self.lock = threading.Lock()
        
    def _process_loop(self):
        while self.is_running:
            state = self._process_frame()
            
            # Thread-safe queue
            try:
                self.output_queue.put(state, timeout=0.01)
            except queue.Full:
                logger.warning("Output queue full, dropping frame")
```

### Resource Management

- **Close resources**: Always close cameras, files, sockets
- **Use context managers**: Implement `__enter__` and `__exit__`

```python
class CameraManager:
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
# Usage:
with CameraManager(config) as camera_manager:
    frame_pair = camera_manager.get_frame_pair()
    # Camera automatically released on exit
```

---

## Documentation

### Module Docstrings

Every module should have a comprehensive docstring:

```python
"""
Camera management module for ChunIVision.

This module provides the CameraManager class which handles initialization,
frame capture, and synchronization of dual infrared cameras using the oculus
library. It supports automatic reconnection and graceful degradation to
single-camera mode if one camera fails.

Example:
    >>> config = CameraConfig(left_index=0, right_index=1)
    >>> manager = CameraManager(config)
    >>> manager.initialize()
    >>> left, right, timestamp = manager.get_frame_pair()

Attributes:
    DEFAULT_FPS (int): Default camera frame rate
    MAX_RECONNECT_ATTEMPTS (int): Maximum reconnection retries
"""
```

### API Documentation

Generate API docs using Sphinx:

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Generate docs
cd docs/
sphinx-build -b html . _build/html
```

---

## Version Control

### Commit Messages

Follow conventional commits:

```
feat: Add gesture recognition module
fix: Resolve camera synchronization issue
docs: Update calibration guide
refactor: Simplify hand detection algorithm
test: Add unit tests for touch detector
perf: Optimize stereo processing loop
```

### Branching Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/xyz`: New feature development
- `bugfix/xyz`: Bug fixes
- `hotfix/xyz`: Critical production fixes

---

## Getting Help

- **Documentation**: Start with README and docs/
- **Examples**: Check tests/ for usage examples
- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions

This framework is designed to be AI-friendly—clear structure, comprehensive documentation, and modular design make it easy for AI assistants to understand and help with development.
