# TransformCalculator Module

## Overview

The `TransformCalculator` module provides utilities for computing and manipulating perspective transformations, which are essential for the calibration process in ChunIVision. It maps image coordinates from the camera view to real-world coordinates on the game surface.

## Purpose

During calibration, users select corner points of a known-size calibration board in the camera view. The `TransformCalculator` computes a perspective transformation matrix that maps these image coordinates to real-world coordinates, allowing the system to accurately track hand positions on the game surface.

## Key Features

- **Perspective Transform Calculation**: Compute 3x3 homography matrices from point correspondences
- **Transform Application**: Apply transforms to single points or batches
- **Inverse Transforms**: Calculate reverse mappings (world → image)
- **Quality Validation**: Assess transform accuracy with multiple metrics
- **Decomposition**: Break down transforms into interpretable components

## Quick Start

```python
from chunivision.calibration import TransformCalculator
import numpy as np

# Define corresponding points
image_points = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
world_points = np.array([[0, 0], [44, 0], [44, 9], [0, 9]], dtype=np.float32)

# Calculate transform
transform = TransformCalculator.calculate_perspective_transform(
    image_points, world_points
)

# Apply to new point
test_point = np.array([300, 250], dtype=np.float32)
world_coord = TransformCalculator.apply_transform(test_point, transform)
print(f"Position: {world_coord[0]:.2f}cm, {world_coord[1]:.2f}cm")
```

## API Reference

### calculate_perspective_transform()

Calculates a perspective transformation matrix from point correspondences.

**Parameters:**
- `image_points` (np.ndarray): Points in image coordinates, shape (4, 2) or (N, 2)
- `world_points` (np.ndarray): Corresponding points in world coordinates

**Returns:**
- np.ndarray: 3x3 perspective transform matrix

**Example:**
```python
transform = TransformCalculator.calculate_perspective_transform(
    image_corners, board_corners
)
```

### apply_transform()

Applies a perspective transform to points.

**Parameters:**
- `points` (np.ndarray): Input points, shape (2,) or (N, 2)
- `transform` (np.ndarray): 3x3 transform matrix

**Returns:**
- np.ndarray: Transformed points, same shape as input

**Example:**
```python
# Single point
world_pt = TransformCalculator.apply_transform(image_pt, transform)

# Multiple points
world_pts = TransformCalculator.apply_transform(image_pts, transform)
```

### calculate_inverse_transform()

Calculates the inverse transform for reverse mapping.

**Parameters:**
- `transform` (np.ndarray): Forward transform matrix (3, 3)

**Returns:**
- np.ndarray: Inverse transform matrix (3, 3)

**Example:**
```python
inverse = TransformCalculator.calculate_inverse_transform(transform)
image_pt = TransformCalculator.apply_transform(world_pt, inverse)
```

### validate_transform_quality()

Computes quality metrics for a transform.

**Parameters:**
- `transform` (np.ndarray): Transform to validate
- `test_points` (np.ndarray): Input test points
- `expected_points` (np.ndarray): Expected output points
- `max_error_threshold` (float): Maximum acceptable error (default: 5.0)

**Returns:**
- Tuple[float, float, float]: (quality_score, mean_error, max_error)
  - quality_score: Overall quality [0.0, 1.0], 1.0 is perfect
  - mean_error: Mean Euclidean distance
  - max_error: Maximum error for any point

**Example:**
```python
quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
    transform, image_points, world_points
)
if quality > 0.95:
    print("Excellent calibration!")
```

### estimate_reprojection_error()

Estimates the mean reprojection error.

**Parameters:**
- `transform` (np.ndarray): Transform matrix
- `image_points` (np.ndarray): Source points in image coordinates
- `world_points` (np.ndarray): Expected points in world coordinates

**Returns:**
- float: Mean reprojection error in world units

**Example:**
```python
error = TransformCalculator.estimate_reprojection_error(
    transform, image_points, world_points
)
```

### decompose_transform()

Decomposes a perspective transform into interpretable components.

**Parameters:**
- `transform` (np.ndarray): Transform matrix (3, 3)

**Returns:**
- dict: Dictionary containing:
  - 'translation': (tx, ty) translation vector
  - 'scale': (sx, sy) scale factors
  - 'rotation': rotation angle in radians
  - 'shear': shear factor
  - 'perspective': (p1, p2) perspective components

**Example:**
```python
components = TransformCalculator.decompose_transform(transform)
print(f"Rotation: {np.degrees(components['rotation']):.2f}°")
```

## Usage in Calibration Workflow

1. **User selects calibration points**: Click corners of calibration board in camera view
2. **Calculate transform**: Map image coordinates to known world coordinates
3. **Validate quality**: Check if transform meets quality threshold
4. **Save transform**: Store in CalibrationData for runtime use
5. **Runtime usage**: Transform detected hand positions from image to world coordinates

## Quality Metrics

### Quality Score

The quality score ranges from 0.0 (worst) to 1.0 (perfect):
- **> 0.95**: Excellent - production ready
- **0.90 - 0.95**: Good - acceptable for most use cases
- **0.70 - 0.90**: Fair - may need recalibration
- **< 0.70**: Poor - recalibration required

### Mean Reprojection Error

Average distance between transformed points and expected positions:
- **< 0.5 cm**: Excellent accuracy
- **0.5 - 1.0 cm**: Good accuracy
- **1.0 - 2.0 cm**: Acceptable
- **> 2.0 cm**: Poor - recalibrate

## Technical Details

### Algorithm

- Uses OpenCV's `getPerspectiveTransform` for 4-point transforms
- Uses OpenCV's `findHomography` with RANSAC for > 4 points
- Homogeneous coordinates for perspective division
- Matrix inversion via numpy.linalg.inv

### Coordinate Systems

- **Image coordinates**: Pixels (x, y) from top-left corner
- **World coordinates**: Centimeters (x, y) from board origin
- **Transform**: Maps image → world
- **Inverse transform**: Maps world → image

### Error Handling

- Validates input shapes and dimensions
- Checks for singular matrices
- Handles points at infinity
- Provides informative error messages

## Examples

See [examples/transform_calculator_example.py](../examples/transform_calculator_example.py) for comprehensive usage examples including:
- Basic transforms
- Inverse transforms
- Quality validation
- Realistic calibration workflow

## Testing

The module includes 48 comprehensive unit tests covering:
- Identity, translation, scaling, and rotation transforms
- Perspective transforms
- Inverse calculations
- Error handling
- Quality validation
- Integration scenarios

Run tests:
```bash
pytest tests/test_calibration/test_transform_calculator.py -v
```

## See Also

- [CalibrationData](calibration_data.py) - Storage for calibration parameters
- [ZoneSelector](zone_selector.py) - GUI for selecting calibration points
- [Calibrator](calibrator.py) - Complete calibration workflow
