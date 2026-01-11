# Hardware Testing Guide

This directory contains hardware tests that require physical Oculus Rift CV1 cameras.

## Test Files

- **test_hardware_oculus.py**: Tests for individual Oculus camera functionality
  - Camera detection and enumeration
  - Serial number handling
  - Camera initialization and properties
  - Streaming and frame capture
  - Calibration and undistortion

- **test_hardware_camera_manager.py**: Tests for dual camera management
  - Dual camera initialization
  - Synchronized frame capture
  - FPS measurement
  - Multiple start/stop cycles
  - Resource management

## Running Hardware Tests

### Prerequisites

1. Connect at least 2 Oculus Rift CV1 cameras via USB
2. Ensure proper USB drivers are installed
3. Verify cameras are detected by the system

### Run Commands

```bash
# Run only hardware tests
pytest -m hardware -v

# Run all tests except hardware tests (for CI/automated testing)
pytest -m "not hardware" -v

# Run all tests including hardware tests
pytest -v

# Run specific hardware test file
pytest tests/test_hardware_oculus.py -v

# Run specific hardware test
pytest tests/test_hardware_oculus.py::TestOculusCameraHardware::test_camera_detection -v
```

## Test Coverage

### Oculus Camera Hardware Tests (11 tests)

1. **test_list_oculus_cameras**: Verify camera enumeration
2. **test_camera_detection**: Detect at least one camera
3. **test_dual_camera_detection**: Verify dual cameras available
4. **test_camera_serial_numbers_unique**: Ensure unique serials
5. **test_open_camera_by_serial**: Open camera by serial number
6. **test_open_nonexistent_serial**: Error handling for invalid serial
7. **test_context_manager_with_real_camera**: Context manager usage
8. **test_dual_camera_initialization**: Simultaneous camera opening
9. **test_camera_streaming_start_stop**: Frame streaming
10. **test_undistort_with_real_calibration**: Calibration verification
11. **test_camera_properties**: Camera property reading

### Camera Manager Hardware Tests (11 tests)

1. **test_camera_manager_initialization**: Basic initialization
2. **test_camera_manager_initialization_fails_invalid_serial**: Error handling
3. **test_camera_manager_get_camera_info**: Camera info retrieval
4. **test_camera_manager_streaming**: Dual camera streaming
5. **test_camera_manager_synchronized_frames**: Frame synchronization
6. **test_camera_manager_fps_measurement**: FPS tracking
7. **test_camera_manager_context_manager**: Context manager usage
8. **test_camera_manager_release_while_streaming**: Resource cleanup
9. **test_camera_manager_multiple_start_stop_cycles**: Cycle testing
10. **test_camera_manager_reconnection**: Re-initialization
11. **test_camera_manager_frame_data_integrity**: Frame validation

## Expected Behavior

- Tests will **automatically skip** if required cameras are not connected (via `@pytest.mark.skipif`)
- Tests use actual USB communication (not mocked)
- Frame data is validated for integrity
- Resource cleanup is verified
- No manual marker selection needed - tests auto-detect hardware availability

## Troubleshooting

### No cameras detected

- Check USB connections
- Verify USB drivers installed
- Run `lsusb | grep Oculus` to see if OS detects cameras

### Permission errors

- Linux: May need udev rules for USB device access
- Add user to appropriate groups (e.g., `plugdev`)

### Initialization failures

- Ensure no other application is using the cameras
- Check USB bandwidth (use USB 3.0 ports)
- Verify both cameras have different serial numbers

## Test Implementation Pattern

All hardware tests use automatic hardware detection:

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
class TestOculusCameraHardware:
    """Hardware tests automatically skip when cameras unavailable."""

    def test_camera_detection(self):
        cameras = list_oculus_cameras()
        assert len(cameras) > 0
```

The `@pytest.mark.skipif` decorator automatically skips tests when `is_hardware_available()` returns `False`.

## Integration with CI/CD

For automated testing without hardware:

```yaml
# .github/workflows/tests.yml
- name: Run tests (auto-skips hardware tests)
  run: pytest -v

# Or explicitly exclude hardware tests
- name: Run tests (skip hardware)
  run: pytest -m "not hardware" -v
```

For manual hardware validation:

```yaml
# Run locally or on hardware-enabled runners
- name: Run hardware tests
  run: pytest -m hardware -v
```
