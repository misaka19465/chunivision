# ChunIVision Architecture Documentation

**Author**: Misaka 19465
**Platform**: Windows Only
**Version**: 1.0

## System Overview

ChunIVision is a vision-based Chusan controller that uses dual infrared cameras to detect hand positions and gestures, replacing traditional touch sensors and infrared height detection arrays.

**Key Specifications:**

- **Touch Zones**: 32 zones (2 rows × 16 columns), each 27.5mm × 45mm
  - Zone numbering: bottom-right is 1, odd numbers on bottom row, even on top row
- **Air Sensors**: 6 height levels at 17.9cm, 21.3cm, 24.7cm, 28.1cm, 31.5cm, 34.9cm
- **Output Methods**: Serial (COM port), HID, UDP (no RGB/LED control)
- **Platform**: Windows exclusive

## Architectural Goals

1. **Modularity**: Each component is independently testable and replaceable
2. **Low Latency**: Optimize for <10ms end-to-end latency
3. **Extensibility**: Easy to add new features, outputs, or vision algorithms
4. **Robustness**: Graceful handling of errors and edge cases
5. **AI-Maintainability**: Clear documentation and interfaces for AI-assisted development

## High-Level Data Flow

```
┌──────────────┐
│   Camera 1   │ (Left-Front, ~45° to origin)
│  (Infrared)  │
└──────┬───────┘
       │
       │ Raw Frames (distortion-corrected via oculus)
       │
       ▼
┌─────────────────────────────────────────┐
│       Camera Manager                    │
│  - Synchronizes dual camera frames      │
│  - Frame buffering (triple buffer)      │
│  - Timestamp alignment                  │
└─────────────────┬───────────────────────┘
                  │
       ┌──────────┘
       │
       ▼
┌──────────────┐
│   Camera 2   │ (Right-Front, ~45° to origin)
│  (Infrared)  │
└──────┬───────┘
       │
       │ Synchronized Frame Pair
       │
       ▼
┌─────────────────────────────────────────┐
│       Stereo Processor                  │
│  - Depth map generation                 │
│  - 3D point cloud reconstruction        │
│  - Camera calibration application       │
└─────────────────┬───────────────────────┘
                  │
                  │ 3D Point Cloud + Depth Map
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Hand Detector                     │
│  - Detects hand regions                 │
│  - Tracks hand positions                │
│  - Filters noise/false positives        │
└─────────────────┬───────────────────────┘
                  │
                  │ Hand Positions (x, y, z)
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────────┐
│Touch Detector │   │ Height Estimator  │
│- Maps hands   │   │ - Assigns hands   │
│  to 32 zones  │   │   to 6 levels     │
│- Detects      │   │ - Height tracking │
│  touch state  │   │                   │
└───────┬───────┘   └─────────┬─────────┘
        │                     │
        │ Zone States         │ Height States
        │ (32 bools)          │ (6 bools)
        │                     │
        └─────────┬───────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       State Manager                     │
│  - Maintains current state              │
│  - Detects state changes (events)       │
│  - Debouncing & filtering               │
│  - State history for debugging          │
└─────────────────┬───────────────────────┘
                  │
                  │ State Change Events
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Output Manager                    │
│  - Routes events to active outputs      │
│  - Handles multiple simultaneous outs   │
│  - Output-specific formatting           │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┼────────┬──────────┐
         │        │        │          │
         ▼        ▼        ▼          ▼
    ┌────────┐┌──────┐┌─────────┐┌──────┐
    │ Serial ││ HID  ││Keyboard ││ UDP  │
    │ Output ││Output││ Output  ││Output│
    └────────┘└──────┘└─────────┘└──────┘
         │        │        │          │
         └────────┴────────┴──────────┘
                  │
                  ▼
         Game receives input
```

## Component Details

### 1. Vision Processing Layer

#### Camera Manager (`vision/camera_manager.py`)

**Responsibilities:**

- Initialize cameras using oculus library
- Capture frames from both cameras simultaneously
- Apply distortion correction (via oculus)
- Synchronize timestamps between cameras
- Manage triple buffering for frame access

**Key Interfaces:**

```python
class CameraManager:
    def initialize(config: CameraConfig) -> bool
    def get_frame_pair() -> Tuple[Frame, Frame, timestamp]
    def release() -> None
    def is_ready() -> bool
```

**Design Considerations:**

- Must handle camera disconnection gracefully
- Should support hot-plugging (detect camera reconnection)
- Configurable frame rate and resolution
- Thread-safe frame access

#### Stereo Processor (`vision/stereo_processor.py`)

**Responsibilities:**

- Convert stereo image pair to depth map
- Generate 3D point cloud
- Apply perspective transformation from calibration
- Coordinate system transformation (camera → world)

**Key Interfaces:**

```python
class StereoProcessor:
    def __init__(calibration_data: CalibrationData)
    def process(left_frame, right_frame) -> PointCloud3D
    def get_depth_map(left_frame, right_frame) -> DepthMap
    def update_calibration(calibration_data: CalibrationData)
```

**Design Considerations:**

- Optimized stereo matching algorithm (e.g., Semi-Global Block Matching)
- Support for GPU acceleration (optional)
- Configurable depth range based on touch zone height

#### Hand Detector (`vision/hand_detector.py`)

**Responsibilities:**

- Detect hand regions in point cloud
- Track multiple hands across frames
- Filter out non-hand objects
- Estimate hand orientation and pose (optional, for future)

**Key Interfaces:**

```python
class HandDetector:
    def detect(point_cloud: PointCloud3D) -> List[Hand]
    def track(hands: List[Hand], previous_hands: List[Hand]) -> List[Hand]

class Hand:
    position: Point3D
    velocity: Vector3D
    confidence: float
    track_id: int
```

**Design Considerations:**

- Fast detection algorithm (target: <5ms per frame)
- Robust to varying hand sizes and skin tones (IR advantage)
- Tracking persistence to avoid flickering

#### Touch Detector (`vision/touch_detector.py`)

**Responsibilities:**

- Map hand positions to 32 touch zones
- Determine if hand is "touching" (z-coordinate threshold)
- Handle multi-touch (multiple hands/fingers)

**Key Interfaces:**

```python
class TouchDetector:
    def __init__(zone_config: ZoneConfig)
    def detect_touches(hands: List[Hand]) -> TouchState

class TouchState:
    zones: np.ndarray  # 32 booleans
    timestamp: float
```

**Design Considerations:**

- Configurable touch threshold (Z-height)
- Zone boundary handling (smooth transitions)
- Support for palm rejection (future feature)

#### Height Estimator (`vision/height_estimator.py`)

**Responsibilities:**

- Assign hands to one of 6 height levels
- Emulate traditional IR sensor array behavior
- Smooth height transitions to avoid jitter

**Key Interfaces:**

```python
class HeightEstimator:
    def estimate_heights(hands: List[Hand]) -> HeightState

class HeightState:
    levels: np.ndarray  # 6 booleans
    timestamp: float
```

**Design Considerations:**

- Configurable height thresholds (0-5cm, 5-10cm, etc.)
- Hysteresis to prevent rapid level switching
- Consider hand size in height estimation

#### Vision Pipeline (`vision/vision_pipeline.py`)

**Responsibilities:**

- Orchestrate all vision processing steps
- Manage processing thread/loop
- Performance monitoring and logging
- Error recovery

**Key Interfaces:**

```python
class VisionPipeline:
    def start() -> None
    def stop() -> None
    def set_callback(callback: Callable[[TouchState, HeightState], None])
    def get_stats() -> PerformanceStats
```

**Design Considerations:**

- Runs in dedicated thread for real-time processing
- Callback-based output (publish-subscribe pattern)
- Graceful degradation if components fail

### 2. Calibration System

#### Calibrator (`calibration/calibrator.py`)

**Responsibilities:**

- Coordinate calibration workflow
- Guide user through calibration steps
- Validate calibration results
- Save/load calibration data

**Key Interfaces:**

```python
class Calibrator:
    def run_calibration() -> CalibrationData
    def load_calibration(path: str) -> CalibrationData
    def save_calibration(data: CalibrationData, path: str) -> None
    def validate_calibration(data: CalibrationData) -> bool
```

**Calibration Workflow:**

1. Initialize cameras
2. Prompt user to place reference board
3. User selects 4 corners via GUI
4. Calculate perspective transformation
5. Verify transformation accuracy
6. Save calibration data

#### Zone Selector (`calibration/zone_selector.py`)

**Responsibilities:**

- Display camera view to user
- Interactive point selection UI
- Visual feedback during selection

**Key Interfaces:**

```python
class ZoneSelector:
    def select_points(frame: Frame) -> List[Point2D]  # Returns 4 points
    def show_preview(frame: Frame, transform: Transform) -> None
```

**Design Considerations:**

- Clear UI with instructions
- Zoom capability for precise selection
- Ability to retry if selection is poor

#### Transform Calculator (`calibration/transform_calculator.py`)

**Responsibilities:**

- Compute perspective transformation matrix
- Calculate inverse transformation for 2D→3D mapping
- Validate transformation quality

**Key Interfaces:**

```python
class TransformCalculator:
    def calculate_transform(
        image_points: List[Point2D],
        world_points: List[Point2D]
    ) -> Transform

    def validate_transform(transform: Transform) -> float  # Quality score
```

#### Calibration Data (`calibration/calibration_data.py`)

**Responsibilities:**

- Store calibration parameters
- Serialize/deserialize to YAML
- Versioning for backward compatibility

**Data Structure:**

```python
class CalibrationData:
    version: str
    timestamp: datetime
    camera_left_transform: Transform
    camera_right_transform: Transform
    zone_boundaries: ZoneBoundaries
    height_thresholds: List[float]  # 6 thresholds
```

### 3. Output Adapters

All output adapters inherit from `BaseOutput` and implement a common interface.

#### Base Output (`output/base_output.py`)

**Responsibilities:**

- Define output adapter interface
- Common error handling
- State tracking

**Key Interfaces:**

```python
class BaseOutput(ABC):
    @abstractmethod
    def initialize() -> bool

    @abstractmethod
    def send_state(touch_state: TouchState, height_state: HeightState) -> None

    @abstractmethod
    def close() -> None

    def is_connected() -> bool
```

#### Serial Output (`output/serial_output.py`)

**Protocol:**

- Creates virtual serial port pair (e.g., using `pty` or `socat`)
- Binary protocol: 38 bytes per update
  - 32 bytes: touch zones (1 byte per zone, 0/1)
  - 6 bytes: height levels (1 byte per level, 0/1)
- Configurable baud rate (default: 115200)

#### HID Output (`output/hid_output.py`)

**Protocol:**

- Uses Linux uhid or Windows vjoy
- Custom HID descriptor defining 38 buttons
- Low-level USB HID report format

#### Keyboard Output (`output/keyboard_output.py`)

**Protocol:**

- Uses uinput (Linux) or pynput (cross-platform)
- Maps each zone/height to a keyboard key
- Configurable key mappings

#### UDP Output (`output/udp_output.py`)

**Protocol:**

- JSON or binary packet format
- Configurable target IP and port
- Low overhead for local communication

```json
{
  "timestamp": 1234567890.123,
  "touch_zones": [0, 1, 0, ...],  // 32 values
  "height_levels": [0, 0, 1, 0, 0, 0]  // 6 values
}
```

#### Output Manager (`output/output_manager.py`)

**Responsibilities:**

- Manage multiple active outputs simultaneously
- Broadcast state updates to all outputs
- Handle output-specific errors without affecting others

### 4. Configuration System

#### Settings (`config/settings.py`)

**Global Settings:**

- Application mode (calibration, run, debug)
- Logging level and output
- Performance monitoring settings
- Default paths

#### Zone Config (`config/zone_config.py`)

**Zone Layout:**

- Physical dimensions of touch zones
- Zone grid configuration (2×16)
- Origin position
- Touch threshold Z-height

#### Camera Config (`config/camera_config.py`)

**Camera Parameters:**

- Camera indices
- Resolution and frame rate
- Camera positions and angles
- Stereo baseline distance

### 5. Utilities

#### Logger (`utils/logger.py`)

- Centralized logging with configurable outputs
- Performance-optimized (no string formatting on disabled levels)
- Support for log rotation

#### Performance Monitor (`utils/performance.py`)

- FPS tracking
- Latency measurement at each pipeline stage
- Memory usage monitoring
- Export performance reports

#### Geometry (`utils/geometry.py`)

- Coordinate transformations (2D↔3D, camera↔world)
- Zone boundary calculations
- Point-in-polygon tests
- Distance calculations

#### State Manager (`utils/state_manager.py`)

- Maintain current game state
- Detect state changes (edge detection)
- Debouncing for noisy inputs
- State history buffer

## Threading Model

```
Main Thread
├── Configuration loading
├── Calibration UI (if in calibration mode)
└── Output Manager

Vision Thread (high priority)
├── Camera Manager (frame capture)
├── Stereo Processor
├── Hand Detector
├── Touch/Height Detection
└── Callback to State Manager

Output Thread
└── Send updates to outputs (triggered by State Manager)
```

**Synchronization:**

- Thread-safe queues for frame passing
- Lock-free state updates where possible
- Callbacks to minimize latency

## Error Handling Strategy

1. **Camera Failures:**
   - Retry connection with exponential backoff
   - Fall back to single camera if one fails (degraded mode)
   - Alert user via logging/UI

2. **Calibration Errors:**
   - Validate user input in real-time
   - Provide clear error messages
   - Allow calibration retry

3. **Output Failures:**
   - Isolate failures (one output failure doesn't affect others)
   - Log errors and continue operation
   - Reconnection attempts for network outputs

4. **Processing Errors:**
   - Skip corrupt frames
   - Fall back to previous valid state
   - Log anomalies for debugging

## Performance Optimization

1. **Vision Processing:**
   - Use NumPy vectorized operations
   - Optimize stereo matching (GPU if available)
   - Reduce frame resolution if needed for speed

2. **Memory Management:**
   - Reuse frame buffers (triple buffering)
   - Avoid unnecessary copies
   - Pool allocation for point clouds

3. **I/O:**
   - Asynchronous output operations
   - Batch state updates if possible
   - Minimize serialization overhead

## Extensibility Points

1. **New Vision Algorithms:**
   - Swap `HandDetector` implementation
   - Add new processors to pipeline
   - Custom calibration methods

2. **New Output Protocols:**
   - Inherit from `BaseOutput`
   - Implement required methods
   - Register with `OutputManager`

3. **Advanced Features (Future):**
   - Gesture recognition (swipes, etc.)
   - Finger tracking (individual fingers)
   - Force/pressure estimation
   - Air motion tracking

## Testing Strategy

1. **Unit Tests:**
   - Each module tested independently
   - Mock dependencies
   - Test edge cases
   - No hardware required

2. **Integration Tests:**
   - Pipeline end-to-end tests
   - Calibration workflow tests
   - Output protocol tests
   - Use mocked hardware components

3. **Performance Tests:**
   - Latency benchmarks
   - FPS stress tests
   - Memory leak detection

4. **Hardware Tests:**
   - **Marked with `@pytest.mark.hardware` decorator**
   - Require physical hardware connections (cameras, USB devices)
   - Test actual camera initialization and communication
   - Verify USB device enumeration and serial numbers
   - Real-world calibration scenarios
   - Game integration testing
   - **Run separately from unit tests using:** `pytest -m hardware`
   - **Skip in CI/automated environments without hardware**

### Hardware Test Requirements

Hardware tests validate the system's interaction with physical devices:

- **Camera Connection Tests**: Verify Oculus Rift CV1 cameras can be detected, opened, and configured
- **USB Communication Tests**: Test USB device enumeration, serial number reading, and device matching
- **Streaming Tests**: Validate frame capture, callback mechanisms, and data transfer
- **Multi-Device Tests**: Ensure dual cameras can operate simultaneously without conflicts
- **Calibration Verification**: Test real camera distortion correction and calibration data

**Running Hardware Tests:**

```bash
# Run only hardware tests
pytest -m hardware -v

# Run all tests except hardware tests
pytest -m "not hardware" -v

# Run all tests including hardware tests
pytest -v
```

**Marking Hardware Tests:**

```python
import pytest
from chunivision.oculus import list_oculus_cameras

def is_hardware_available() -> bool:
    """Check if Oculus cameras are connected and available."""
    try:
        cameras = list_oculus_cameras()
        return len(cameras) > 0
    except Exception:
        return False

@pytest.mark.hardware
@pytest.mark.skipif(
    not is_hardware_available(),
    reason="hardware needed."
)
def test_actual_camera_connection():
    """Test requires physical Oculus cameras connected.

    Automatically skips if cameras not available.
    """
    cameras = list_oculus_cameras()
    assert len(cameras) >= 2
```
