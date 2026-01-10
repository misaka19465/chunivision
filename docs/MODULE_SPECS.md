# Module Specifications and API Reference

This document provides detailed specifications for each module in the ChunIVision system, including class interfaces, data structures, and interaction patterns.

## Table of Contents

1. [Vision Processing Modules](#vision-processing-modules)
2. [Calibration Modules](#calibration-modules)
3. [Output Adapters](#output-adapters)
4. [Configuration Modules](#configuration-modules)
5. [Utility Modules](#utility-modules)
6. [Data Structures](#data-structures)

---

## Vision Processing Modules

### CameraManager (`vision/camera_manager.py`)

**Purpose:** Manages dual camera initialization, frame capture, and synchronization.

**Class Interface:**

```python
class CameraManager:
    """
    Manages two infrared cameras for stereo vision.
    
    Attributes:
        left_camera: Camera object for left camera
        right_camera: Camera object for right camera
        config: CameraConfig with camera parameters
    """
    
    def __init__(self, config: CameraConfig):
        """
        Initialize camera manager with configuration.
        
        Args:
            config: CameraConfig object with camera indices and parameters
        """
        
    def initialize(self) -> bool:
        """
        Initialize both cameras and verify they are ready.
        
        Returns:
            True if both cameras initialized successfully, False otherwise
            
        Raises:
            CameraInitError: If cameras cannot be initialized
        """
        
    def get_frame_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Capture synchronized frames from both cameras.
        
        Returns:
            Tuple of (left_frame, right_frame, timestamp) or None if capture failed
            Frames are numpy arrays of shape (H, W) for infrared images
            Timestamp is in seconds since epoch
        """
        
    def release(self) -> None:
        """Release camera resources and close connections."""
        
    def is_ready(self) -> bool:
        """Check if both cameras are connected and ready."""
        
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get information about connected cameras.
        
        Returns:
            Dictionary with camera properties (resolution, FPS, etc.)
        """
```

**Dependencies:**

- `oculus.camera` for camera interface
- `oculus.triple_buffer` for efficient frame buffering

**Configuration Parameters:**

- `left_camera_index`: Index of left camera device
- `right_camera_index`: Index of right camera device
- `resolution`: Tuple of (width, height)
- `fps`: Target frame rate
- `exposure`: Camera exposure setting

**Error Handling:**

- Retry connection on transient failures
- Log detailed error messages
- Graceful degradation to single camera if one fails

---

### StereoProcessor (`vision/stereo_processor.py`)

**Purpose:** Processes stereo image pairs to generate depth maps and 3D point clouds.

**Class Interface:**

```python
class StereoProcessor:
    """
    Converts stereo images to 3D point clouds.
    
    Uses calibration data to perform stereo matching and 3D reconstruction.
    """
    
    def __init__(self, calibration_data: CalibrationData):
        """
        Initialize processor with calibration data.
        
        Args:
            calibration_data: Calibration matrices and parameters
        """
        
    def process(self, 
                left_frame: np.ndarray, 
                right_frame: np.ndarray) -> PointCloud3D:
        """
        Generate 3D point cloud from stereo image pair.
        
        Args:
            left_frame: Left camera frame (H, W) numpy array
            right_frame: Right camera frame (H, W) numpy array
            
        Returns:
            PointCloud3D object with 3D coordinates
        """
        
    def get_depth_map(self, 
                      left_frame: np.ndarray, 
                      right_frame: np.ndarray) -> np.ndarray:
        """
        Compute depth map from stereo pair.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            
        Returns:
            Depth map as (H, W) numpy array with depth in cm
        """
        
    def update_calibration(self, calibration_data: CalibrationData) -> None:
        """Update calibration parameters (e.g., after recalibration)."""
        
    def set_depth_range(self, min_depth: float, max_depth: float) -> None:
        """
        Set valid depth range for filtering.
        
        Args:
            min_depth: Minimum depth in cm (e.g., 0 for touch surface)
            max_depth: Maximum depth in cm (e.g., 30 for max height)
        """
```

**Algorithm Considerations:**

- Use Semi-Global Block Matching (SGBM) or similar
- Optimize for speed (GPU acceleration optional)
- Filter outliers and noise in depth map

---

### HandDetector (`vision/hand_detector.py`)

**Purpose:** Detects and tracks hands in 3D point cloud.

**Class Interface:**

```python
class Hand:
    """
    Represents a detected hand.
    
    Attributes:
        position: 3D position (x, y, z) in cm from origin
        velocity: 3D velocity vector in cm/s
        confidence: Detection confidence [0.0, 1.0]
        track_id: Unique identifier for tracking across frames
        timestamp: Detection timestamp
    """
    position: np.ndarray  # shape (3,)
    velocity: np.ndarray  # shape (3,)
    confidence: float
    track_id: int
    timestamp: float


class HandDetector:
    """
    Detects hands in point cloud and tracks them across frames.
    """
    
    def __init__(self, config: HandDetectorConfig):
        """
        Initialize hand detector.
        
        Args:
            config: Detection parameters (thresholds, etc.)
        """
        
    def detect(self, point_cloud: PointCloud3D) -> List[Hand]:
        """
        Detect all hands in current point cloud.
        
        Args:
            point_cloud: 3D point cloud from stereo processing
            
        Returns:
            List of detected Hand objects
        """
        
    def track(self, 
              current_hands: List[Hand], 
              previous_hands: List[Hand]) -> List[Hand]:
        """
        Associate current detections with previous frame for tracking.
        
        Args:
            current_hands: Hands detected in current frame
            previous_hands: Hands from previous frame
            
        Returns:
            List of hands with updated track_ids and velocities
        """
        
    def reset_tracking(self) -> None:
        """Reset all tracking state (e.g., after occlusion or error)."""
```

**Detection Strategy:**

- Cluster 3D points by proximity
- Filter clusters by size (hand-sized regions)
- Use IR properties to distinguish hands from background
- Track using Kalman filter or similar

---

### TouchDetector (`vision/touch_detector.py`)

**Purpose:** Maps hand positions to touch zone states.

**Class Interface:**

```python
class TouchState:
    """
    Represents state of all 32 touch zones.
    
    Attributes:
        zones: Boolean array of size 32 (True = touched)
        timestamp: State timestamp
        touch_positions: Optional list of exact touch positions
    """
    zones: np.ndarray  # shape (32,), dtype=bool
    timestamp: float
    touch_positions: Optional[List[np.ndarray]] = None


class TouchDetector:
    """
    Determines which touch zones are being touched based on hand positions.
    """
    
    def __init__(self, zone_config: ZoneConfig):
        """
        Initialize with zone layout configuration.
        
        Args:
            zone_config: Defines zone boundaries and touch thresholds
        """
        
    def detect_touches(self, hands: List[Hand]) -> TouchState:
        """
        Determine touch state from hand positions.
        
        Args:
            hands: List of detected hands
            
        Returns:
            TouchState indicating which zones are touched
        """
        
    def set_touch_threshold(self, z_threshold: float) -> None:
        """
        Set Z-coordinate threshold for touch detection.
        
        Args:
            z_threshold: Height in cm below which hand is considered touching
        """
        
    def get_zone_boundary(self, zone_id: int) -> np.ndarray:
        """
        Get boundary polygon for a specific zone.
        
        Args:
            zone_id: Zone number (0-31)
            
        Returns:
            Array of 2D boundary points
        """
```

**Zone Mapping:**

- Zones numbered 0-31 (rows 0-1, columns 0-15)
- Use point-in-polygon test for zone assignment
- Handle overlapping hands (priority to lowest Z)

---

### HeightEstimator (`vision/height_estimator.py`)

**Purpose:** Estimates which height levels are occupied by hands.

**Class Interface:**

```python
class HeightState:
    """
    Represents state of 6 height levels.
    
    Attributes:
        levels: Boolean array of size 6 (True = hand present at level)
        timestamp: State timestamp
        exact_heights: Optional list of exact hand heights
    """
    levels: np.ndarray  # shape (6,), dtype=bool
    timestamp: float
    exact_heights: Optional[List[float]] = None


class HeightEstimator:
    """
    Assigns hands to discrete height levels (emulates IR sensor array).
    """
    
    def __init__(self, height_config: HeightConfig):
        """
        Initialize with height level thresholds.
        
        Args:
            height_config: Defines 6 height level boundaries
        """
        
    def estimate_heights(self, hands: List[Hand]) -> HeightState:
        """
        Determine which height levels are occupied.
        
        Args:
            hands: List of detected hands
            
        Returns:
            HeightState indicating occupied levels
        """
        
    def set_thresholds(self, thresholds: List[float]) -> None:
        """
        Update height level thresholds.
        
        Args:
            thresholds: List of 6 threshold values in cm
        """
```

**Height Levels:**

- Level 0: 0-5cm
- Level 1: 5-10cm
- Level 2: 10-15cm
- Level 3: 15-20cm
- Level 4: 20-25cm
- Level 5: 25cm+

**Hysteresis:**

- Use hysteresis to prevent rapid level switching
- Smooth transitions over multiple frames

---

### VisionPipeline (`vision/vision_pipeline.py`)

**Purpose:** Orchestrates the entire vision processing workflow.

**Class Interface:**

```python
class VisionPipeline:
    """
    Main vision processing pipeline coordinator.
    
    Manages all vision components and runs processing loop.
    """
    
    def __init__(self, 
                 camera_config: CameraConfig,
                 calibration_data: CalibrationData,
                 zone_config: ZoneConfig,
                 height_config: HeightConfig):
        """Initialize pipeline with all required configurations."""
        
    def start(self) -> None:
        """Start the vision processing loop in a dedicated thread."""
        
    def stop(self) -> None:
        """Stop the vision processing loop and cleanup resources."""
        
    def set_callback(self, 
                     callback: Callable[[TouchState, HeightState], None]) -> None:
        """
        Set callback function to receive state updates.
        
        Args:
            callback: Function called with (touch_state, height_state)
                     whenever state changes
        """
        
    def get_stats(self) -> PerformanceStats:
        """
        Get current performance statistics.
        
        Returns:
            PerformanceStats object with FPS, latency, etc.
        """
        
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get latest camera frame for debugging/visualization."""
```

**Processing Loop:**

1. Capture frame pair from cameras
2. Generate depth map and point cloud
3. Detect hands
4. Detect touches and estimate heights
5. Update state manager
6. Invoke callbacks with state changes

---

## Calibration Modules

### Calibrator (`calibration/calibrator.py`)

**Purpose:** Manages the calibration workflow.

**Class Interface:**

```python
class Calibrator:
    """
    Coordinates camera and zone calibration process.
    """
    
    def __init__(self, camera_manager: CameraManager):
        """
        Initialize calibrator with camera manager.
        
        Args:
            camera_manager: Active camera manager instance
        """
        
    def run_calibration(self) -> CalibrationData:
        """
        Execute full calibration workflow interactively.
        
        Returns:
            CalibrationData with all calibration parameters
            
        Raises:
            CalibrationError: If calibration fails or is cancelled
        """
        
    def load_calibration(self, path: str) -> CalibrationData:
        """
        Load calibration from file.
        
        Args:
            path: Path to calibration YAML file
            
        Returns:
            Loaded CalibrationData object
        """
        
    def save_calibration(self, 
                         data: CalibrationData, 
                         path: str) -> None:
        """
        Save calibration to file.
        
        Args:
            data: CalibrationData to save
            path: Output file path
        """
        
    def validate_calibration(self, data: CalibrationData) -> Tuple[bool, float]:
        """
        Validate calibration quality.
        
        Args:
            data: CalibrationData to validate
            
        Returns:
            Tuple of (is_valid, quality_score)
            quality_score is in range [0.0, 1.0]
        """
```

**Calibration Steps:**

1. Initialize cameras and display live view
2. Prompt user to place reference board (correct size rectangle)
3. User selects 4 corners of board in each camera view
4. Calculate perspective transforms for both cameras
5. Verify calibration by checking reprojection error
6. Save calibration data

---

### ZoneSelector (`calibration/zone_selector.py`)

**Purpose:** Interactive UI for selecting calibration points.

**Class Interface:**

```python
class ZoneSelector:
    """
    GUI for user to select calibration points on camera view.
    """
    
    def __init__(self, window_title: str = "Zone Calibration"):
        """Initialize selector with window configuration."""
        
    def select_points(self, 
                      frame: np.ndarray, 
                      num_points: int = 4,
                      instructions: str = "") -> List[np.ndarray]:
        """
        Display frame and let user select points by clicking.
        
        Args:
            frame: Camera frame to display
            num_points: Number of points to select
            instructions: Text to display to user
            
        Returns:
            List of selected points as (x, y) coordinates
        """
        
    def show_preview(self, 
                     frame: np.ndarray, 
                     transform: np.ndarray,
                     zone_grid: Optional[np.ndarray] = None) -> None:
        """
        Show preview of calibrated view with zone overlay.
        
        Args:
            frame: Camera frame
            transform: Perspective transform matrix
            zone_grid: Optional zone boundary overlay
        """
        
    def close(self) -> None:
        """Close the selector window."""
```

**UI Features:**

- Click to select points
- Display selected points with markers
- Zoom capability for precision
- Undo last point
- Visual feedback and instructions

---

### TransformCalculator (`calibration/transform_calculator.py`)

**Purpose:** Calculates perspective transformations from calibration points.

**Class Interface:**

```python
class TransformCalculator:
    """
    Computes geometric transformations for calibration.
    """
    
    @staticmethod
    def calculate_perspective_transform(
        image_points: np.ndarray,
        world_points: np.ndarray
    ) -> np.ndarray:
        """
        Calculate perspective transform matrix.
        
        Args:
            image_points: Points in image coordinates (4, 2)
            world_points: Corresponding points in world coordinates (4, 2)
            
        Returns:
            3x3 perspective transform matrix
        """
        
    @staticmethod
    def calculate_inverse_transform(transform: np.ndarray) -> np.ndarray:
        """
        Calculate inverse of transform for reverse mapping.
        
        Args:
            transform: Forward transform matrix
            
        Returns:
            Inverse transform matrix
        """
        
    @staticmethod
    def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply transform to points.
        
        Args:
            points: Input points (N, 2)
            transform: Transform matrix (3, 3)
            
        Returns:
            Transformed points (N, 2)
        """
        
    @staticmethod
    def validate_transform_quality(
        transform: np.ndarray,
        test_points: np.ndarray,
        expected_points: np.ndarray
    ) -> float:
        """
        Compute quality metric for transform.
        
        Args:
            transform: Transform to validate
            test_points: Input test points
            expected_points: Expected output points
            
        Returns:
            Quality score [0.0, 1.0], 1.0 is perfect
        """
```

---

### CalibrationData (`calibration/calibration_data.py`)

**Purpose:** Data structure for storing calibration parameters.

**Class Interface:**

```python
@dataclass
class CalibrationData:
    """
    Complete calibration data for the system.
    
    Attributes:
        version: Calibration format version
        timestamp: When calibration was performed
        camera_left_transform: Perspective transform for left camera
        camera_right_transform: Perspective transform for right camera
        zone_boundaries: Physical boundaries of 32 zones
        height_thresholds: Z-coordinate thresholds for 6 air sensor levels in cm
                          [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]
                          (Air 0 at 17.9cm, subsequent levels at 3.4cm intervals)
        reference_board_size: Size of calibration board (width, height) in cm
        stereo_baseline: Distance between camera centers in cm
    """
    version: str
    timestamp: datetime
    camera_left_transform: np.ndarray
    camera_right_transform: np.ndarray
    zone_boundaries: np.ndarray  # shape (32, 4, 2) for zone corners
    height_thresholds: np.ndarray  # shape (6,) - [17.9, 21.3, 24.7, 28.1, 31.5, 34.9] cm
    reference_board_size: Tuple[float, float]
    stereo_baseline: float
    
    def save(self, path: str) -> None:
        """Save to YAML file."""
        
    @classmethod
    def load(cls, path: str) -> 'CalibrationData':
        """Load from YAML file."""
        
    def validate(self) -> bool:
        """Validate data integrity."""
```

---

## Output Adapters

### BaseOutput (`output/base_output.py`)

**Purpose:** Abstract base class for all output adapters.

**Class Interface:**

```python
class BaseOutput(ABC):
    """
    Abstract base class for output adapters.
    
    All output implementations must inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output with configuration.
        
        Args:
            config: Output-specific configuration parameters
        """
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the output connection/device.
        
        Returns:
            True if initialization successful, False otherwise
        """
        
    @abstractmethod
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> bool:
        """
        Send current state to output.
        
        Args:
            touch_state: Current touch zone states
            height_state: Current height level states
            
        Returns:
            True if send successful, False otherwise
        """
        
    @abstractmethod
    def close(self) -> None:
        """Close output connection and cleanup resources."""
        
    def is_connected(self) -> bool:
        """
        Check if output is connected/ready.
        
        Returns:
            True if connected and ready, False otherwise
        """
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get output statistics (packets sent, errors, etc.).
        
        Returns:
            Dictionary with statistics
        """
```

---

### SerialOutput (`output/serial_output.py`)

**Purpose:** Virtual serial port output.

**Class Interface:**

```python
class SerialOutput(BaseOutput):
    """
    Output via virtual serial port.
    
    Creates a pair of virtual serial ports (master/slave).
    Game connects to slave port to receive data.
    """
    
    def __init__(self, config: SerialConfig):
        """
        Initialize serial output.
        
        Args:
            config: Serial port configuration (baud rate, etc.)
        """
        
    def initialize(self) -> bool:
        """Create virtual serial port pair."""
        
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> bool:
        """
        Send state as binary packet.
        
        Packet format (38 bytes):
        - Bytes 0-31: Touch zones (0x00 or 0x01)
        - Bytes 32-37: Height levels (0x00 or 0x01)
        """
        
    def get_slave_port_path(self) -> str:
        """Get path to slave port for game to connect."""
```

---

### HIDOutput (`output/hid_output.py`)

**Purpose:** Virtual HID device output.

**Class Interface:**

```python
class HIDOutput(BaseOutput):
    """
    Output via virtual HID device.
    
    Creates a virtual HID device with 38 buttons.
    """
    
    def __init__(self, config: HIDConfig):
        """Initialize HID output with device descriptor."""
        
    def initialize(self) -> bool:
        """Create virtual HID device (uses uhid on Linux)."""
        
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> bool:
        """
        Send state as HID report.
        
        Report format:
        - Report ID: 1
        - 38 button states (bit-packed into 5 bytes)
        """
```

---

### KeyboardOutput (`output/keyboard_output.py`)

**Purpose:** Virtual keyboard output.

**Class Interface:**

```python
class KeyboardOutput(BaseOutput):
    """
    Output via virtual keyboard.
    
    Maps each zone/height to a keyboard key.
    """
    
    def __init__(self, config: KeyboardConfig):
        """
        Initialize keyboard output with key mappings.
        
        Args:
            config: Key mapping configuration
        """
        
    def initialize(self) -> bool:
        """Create virtual keyboard device (uses uinput on Linux)."""
        
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> bool:
        """
        Send key press/release events based on state changes.
        """
        
    def set_key_mapping(self, zone_keys: List[str], height_keys: List[str]) -> None:
        """
        Update key mappings.
        
        Args:
            zone_keys: List of 32 key codes for zones
            height_keys: List of 6 key codes for heights
        """
```

---

### UDPOutput (`output/udp_output.py`)

**Purpose:** UDP network output.

**Class Interface:**

```python
class UDPOutput(BaseOutput):
    """
    Output via UDP packets.
    
    Sends state updates to configured IP and port.
    """
    
    def __init__(self, config: UDPConfig):
        """
        Initialize UDP output.
        
        Args:
            config: Target IP, port, and format
        """
        
    def initialize(self) -> bool:
        """Initialize UDP socket."""
        
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> bool:
        """
        Send state as UDP packet.
        
        Packet format (JSON or binary):
        JSON: {"timestamp": float, "touch": [0,1,...], "height": [0,0,1,0,0,0]}
        Binary: 46 bytes (8 timestamp + 32 touch + 6 height)
        """
        
    def set_format(self, format: str) -> None:
        """
        Set packet format.
        
        Args:
            format: "json" or "binary"
        """
```

---

### OutputManager (`output/output_manager.py`)

**Purpose:** Manages multiple simultaneous outputs.

**Class Interface:**

```python
class OutputManager:
    """
    Manages multiple output adapters simultaneously.
    
    Broadcasts state updates to all active outputs.
    """
    
    def __init__(self):
        """Initialize output manager."""
        
    def add_output(self, name: str, output: BaseOutput) -> None:
        """
        Register an output adapter.
        
        Args:
            name: Unique name for this output
            output: BaseOutput instance
        """
        
    def remove_output(self, name: str) -> None:
        """Remove an output adapter."""
        
    def send_state(self, 
                   touch_state: TouchState, 
                   height_state: HeightState) -> Dict[str, bool]:
        """
        Send state to all registered outputs.
        
        Args:
            touch_state: Current touch states
            height_state: Current height states
            
        Returns:
            Dictionary mapping output names to success status
        """
        
    def get_active_outputs(self) -> List[str]:
        """Get list of active output names."""
```

---

## Configuration Modules

### Settings (`config/settings.py`)

```python
@dataclass
class Settings:
    """
    Global application settings.
    """
    mode: str = "run"  # "run", "calibration", "debug"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    performance_monitoring: bool = True
    config_dir: str = "configs"
    calibration_file: str = "configs/calibration.yaml"
    
    @classmethod
    def load(cls, path: str) -> 'Settings':
        """Load settings from YAML file."""
        
    def save(self, path: str) -> None:
        """Save settings to YAML file."""
```

---

### ZoneConfig (`config/zone_config.py`)

```python
@dataclass
class ZoneConfig:
    """
    Touch zone layout configuration.
    
    Zone numbering: Bottom-right is zone 1, bottom row has odd numbers (31,29...1 from left),
    top row has even numbers (32,30...2 from left).
    """
    num_rows: int = 2
    num_cols: int = 16
    zone_width: float = 2.75  # cm (27.5mm per zone)
    zone_height: float = 4.5  # cm (45mm per zone)
    origin_x: float = 22.0  # cm (adjusted for 16 zones @ 2.75cm each)
    origin_y: float = 0.0  # cm (bottom row)
    touch_threshold_z: float = 2.0  # cm above surface
    
    def get_zone_boundary(self, zone_id: int) -> np.ndarray:
        """Get boundary polygon for zone (zone_id: 1-32)."""
        
    def get_zone_center(self, zone_id: int) -> np.ndarray:
        """Get center point of zone (zone_id: 1-32)."""
```

---

### CameraConfig (`config/camera_config.py`)

```python
@dataclass
class CameraConfig:
    """
    Camera hardware configuration.
    """
    left_camera_index: int = 0
    right_camera_index: int = 1
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 60
    exposure: int = -1  # Auto
    left_camera_position: np.ndarray = field(default_factory=lambda: np.array([-10, 15, 30]))
    right_camera_position: np.ndarray = field(default_factory=lambda: np.array([10, 15, 30]))
    baseline_distance: float = 20.0  # cm
```

---

## Utility Modules

### Logger (`utils/logger.py`)

```python
class Logger:
    """
    Centralized logging system.
    """
    
    @staticmethod
    def setup(level: str, log_file: Optional[str] = None) -> None:
        """Configure logging."""
        
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get logger instance for module."""
```

---

### PerformanceMonitor (`utils/performance.py`)

```python
class PerformanceMonitor:
    """
    Tracks system performance metrics.
    """
    
    def record_frame(self) -> None:
        """Record frame processing completion."""
        
    def record_latency(self, stage: str, latency_ms: float) -> None:
        """Record latency for a pipeline stage."""
        
    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
```

---

### StateManager (`utils/state_manager.py`)

```python
class StateManager:
    """
    Manages game state and detects changes.
    """
    
    def update_state(self, 
                     touch_state: TouchState, 
                     height_state: HeightState) -> bool:
        """
        Update current state and detect changes.
        
        Returns:
            True if state changed, False otherwise
        """
        
    def get_current_state(self) -> Tuple[TouchState, HeightState]:
        """Get current state."""
        
    def get_changes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state changes since last update.
        
        Returns:
            Tuple of (touch_changes, height_changes)
            Changes are arrays with -1 (released), 0 (no change), 1 (pressed)
        """
```

---

## Data Structures

### PointCloud3D

```python
@dataclass
class PointCloud3D:
    """
    3D point cloud representation.
    """
    points: np.ndarray  # shape (N, 3), xyz coordinates in cm
    colors: Optional[np.ndarray] = None  # shape (N, 3), RGB
    timestamp: float = 0.0
```

### PerformanceStats

```python
@dataclass
class PerformanceStats:
    """
    Performance monitoring statistics.
    """
    fps: float
    avg_latency_ms: float
    stage_latencies: Dict[str, float]  # Latency per pipeline stage
    memory_usage_mb: float
    dropped_frames: int
```

---

## Module Dependencies

```
vision/vision_pipeline.py
├── vision/camera_manager.py
│   └── oculus.camera
├── vision/stereo_processor.py
├── vision/hand_detector.py
├── vision/touch_detector.py
│   └── config/zone_config.py
├── vision/height_estimator.py
└── utils/state_manager.py

calibration/calibrator.py
├── calibration/zone_selector.py
├── calibration/transform_calculator.py
├── calibration/calibration_data.py
└── vision/camera_manager.py

output/output_manager.py
├── output/serial_output.py
├── output/hid_output.py
├── output/keyboard_output.py
└── output/udp_output.py
```

This specification provides the foundation for implementing each module with clear interfaces and responsibilities.
