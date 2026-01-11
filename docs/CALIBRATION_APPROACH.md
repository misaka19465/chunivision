# Calibration Approach - ChunIVision

## Overview

This document clarifies the calibration approach for ChunIVision, which differs from traditional stereo vision calibration systems.

## Two Types of Calibration

### 1. Lens Distortion Correction (Automatic)

**Handled By**: Oculus camera library (factory-calibrated)

**What it does**:
- Corrects radial and tangential lens distortion
- Uses factory-calibrated parameters stored in camera firmware
- Produces geometrically correct (undistorted) images

**User action required**: **NONE** - This is automatic

**Technical details**:
- The Oculus Rift CV1 cameras come with pre-calibrated distortion parameters
- The `oculus_camera.py` library automatically retrieves these via `get_calibration_params()`
- All frames from the camera are already lens-distortion-corrected
- Parameters include: focal length (fx, fy), optical center (cx, cy), distortion coefficients (k0-k3)

### 2. Perspective Transformation (User Calibration)

**Handled By**: ChunIVision calibration workflow

**What it does**:
- Maps camera image coordinates to physical touch zone coordinates
- Establishes spatial relationship between camera view and game surface
- Calibrates height detection thresholds

**User action required**: **YES** - Interactive calibration process

**Calibration steps**:

1. **Place calibration board** on touch surface
   - Use a plain rectangular board (NOT a checkerboard pattern)
   - Board size should match touch zone area (e.g., 44cm × 9cm)

2. **Select 4 corner points** for each camera
   - Click on the 4 corners of the calibration board in camera view
   - Order: bottom-left, bottom-right, top-right, top-left (clockwise)
   - This establishes the perspective transformation matrix

3. **Calibrate height thresholds**
   - Place hand at specified heights
   - System records Z-coordinate thresholds for 6 air sensor levels

4. **Save calibration data**
   - Perspective transforms for both cameras
   - Height thresholds
   - Zone boundary information

## Why This Approach?

### Traditional Stereo Calibration (What We DON'T Do)

Traditional stereo vision systems require:
- Capturing 10+ image pairs of a checkerboard pattern
- Computing intrinsic parameters (camera matrix, distortion coefficients)
- Computing extrinsic parameters (rotation, translation between cameras)
- Computing rectification transforms

**Problems with this approach**:
- Time-consuming (many image captures needed)
- Requires precision checkerboard pattern
- Redundant - Oculus cameras already have factory calibration
- Overcomplicated for our use case

### Our Simplified Approach (What We DO)

We leverage the fact that:
- Oculus cameras are **already calibrated** at the factory
- We only need to map the **undistorted** camera view to game surface coordinates
- A simple 4-point perspective transformation is sufficient

**Benefits**:
- Fast calibration (2-3 minutes)
- Simple rectangular board (no special pattern)
- Accurate enough for touch zone detection
- User-friendly

## Code Architecture

### What Uses Factory Calibration

```python
# oculus/oculus_camera.py
camera = OculusRiftCV1Camera(serial_number)
calibration_params = camera.get_calibration_params()
# Returns: fx, fy, cx, cy, k[0-3], max_r2

# oculus/viewer.py
distortion_cal = DistortionCalibration(
    center=(calibration_params["cx"], calibration_params["cy"]),
    kappas=calibration_params["k"][:3],
    # ... automatically undistorts images
)
```

### What Uses User Calibration

```python
# calibration/calibrator.py
calibrator = Calibrator(
    board_physical_size=(44.0, 9.0),  # Physical size in cm
    zone_grid=(16, 2)  # Number of zones
)

# User selects 4 corners interactively
calibration_data = calibrator.run_interactive_calibration(left_cam, right_cam)

# calibration/transform_calculator.py
# Computes 3x3 homography matrix
transform = TransformCalculator.calculate_perspective_transform(
    image_points,  # 4 clicked corners in pixels
    world_points   # 4 physical corners in cm
)

# At runtime: convert image coords to physical coords
hand_pixel = np.array([320, 240])  # Hand position in image
hand_physical = TransformCalculator.apply_transform(hand_pixel, transform)
# Returns: [x_cm, y_cm] on game surface
```

## Calibration Data Storage

Saved calibration includes:

```yaml
version: "1.0"
timestamp: "2026-01-11T10:30:00"

# Perspective transforms (image → physical coords)
camera_left_transform:
  - [1.2, 0.1, -50.0]
  - [0.0, 1.3, -30.0]
  - [0.001, 0.0, 1.0]

camera_right_transform:
  - [1.1, -0.1, -45.0]
  - [0.0, 1.2, -28.0]
  - [-0.001, 0.0, 1.0]

# Zone boundaries (derived from grid + board size)
zone_boundaries:
  grid: [16, 2]
  zone_size: [2.75, 4.5]
  zones: [...]

# Height detection thresholds (cm above surface)
height_thresholds: [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

# Physical parameters
stereo_baseline: 20.0  # cm between cameras
reference_board_size: [44.0, 9.0]  # cm

# NOTE: Lens distortion parameters are NOT stored here
# They come from Oculus camera factory calibration
```

## Summary

| Aspect                   | Lens Distortion        | Perspective Transform     |
| ------------------------ | ---------------------- | ------------------------- |
| **Purpose**              | Fix lens warping       | Map image to game surface |
| **Handled by**           | Oculus library         | ChunIVision calibration   |
| **Calibration method**   | Factory (one-time)     | User (per setup)          |
| **Calibration pattern**  | Checkerboard (factory) | Plain rectangle (user)    |
| **When calibrated**      | At manufacture         | Each installation         |
| **User involvement**     | None                   | Select 4 corners          |
| **Stored where**         | Camera firmware        | calibration.yaml          |
| **Recalibration needed** | Never                  | When cameras move         |

## References

- [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) - User guide for calibration
- [TRANSFORM_CALCULATOR.md](TRANSFORM_CALCULATOR.md) - Technical details of perspective transforms
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
