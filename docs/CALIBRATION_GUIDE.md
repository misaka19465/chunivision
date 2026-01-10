# Calibration Guide

This guide explains how to calibrate the ChunIVision system for accurate touch detection and height estimation.

## Overview

Calibration establishes the relationship between camera images and physical touch zone positions. This involves:

1. **Camera Positioning**: Physical setup of cameras
2. **Zone Calibration**: Mapping camera view to 32 touch zones
3. **Height Calibration**: Setting height level thresholds
4. **Verification**: Testing calibration accuracy

---

## Prerequisites

### Hardware Requirements

- **Two infrared cameras**: Compatible with oculus library, with distortion correction
- **Camera mounting**: Stable mounts allowing ~45° angle toward touch surface
- **Calibration board**: Rectangular board matching touch zone area dimensions
  - Recommended size: 40cm × 10cm (matches 2 rows × 16 columns of ~2.5cm zones)
  - Material: Rigid, flat surface (cardboard, acrylic, wood)
  - Color: Any (infrared cameras don't depend on color)

### Software Requirements

- ChunIVision installed and configured
- Camera drivers and oculus library working
- Display for viewing camera feed during calibration

---

## Camera Physical Setup

### Positioning Guidelines

```
                     Top View
                        
    Camera 1                      Camera 2
      (Left)                       (Right)
         \                        /
          \                      /
           \                    /
            \                  /
             \    45°      45°/
              \              /
               \            /
                \          /
                 v        v
            ┌────────────────────┐
            │                    │
            │   Touch Surface    │
            │      (origin)      │
            │                    │
            └────────────────────┘
            
            
                     Side View
                     
    Camera 1/2
         \
          \  ~45°
           \
            v
    ┌────────────────────┐
    │  Touch Surface     │
    └────────────────────┘
    
Height: 25-35cm above surface (adjustable)
```

**Key Parameters:**

- **Height**: 25-35cm above touch surface (adjust based on room and desired detection range)
- **Angle**: ~45° pointing toward origin (between columns 8 and 9)
- **Separation**: 15-25cm between cameras (stereo baseline)
- **Origin**: Physical center point between columns 8 and 9, at bottom row

**Important Considerations:**

1. **Stable mounting**: Cameras must not move during gameplay
2. **Lighting**: Avoid bright infrared sources (sunlight, halogen lamps)
3. **Background**: Minimize clutter in camera view
4. **Coverage**: Both cameras should see entire touch area

---

## Running Calibration

### Step 1: Launch Calibration Mode

```bash
cd /path/to/chunivision
python -m chunivision.calibration
```

Or specify config file:

```bash
python -m chunivision.main --mode calibration --config configs/calibration.yaml
```

### Step 2: Camera Initialization

The calibration tool will:

1. Detect and initialize both cameras
2. Display live feed from each camera
3. Verify cameras are working

**Expected Output:**

```
[INFO] Initializing cameras...
[INFO] Left camera (index 0): OK (640x480 @ 60fps)
[INFO] Right camera (index 1): OK (640x480 @ 60fps)
[INFO] Cameras synchronized
```

**Troubleshooting:**

- **Camera not found**: Check camera connections and indices in config
- **Low framerate**: Reduce resolution or close other camera applications
- **Cameras not synchronized**: Ensure both cameras are same model/settings

### Step 3: Place Calibration Board

1. Place rectangular calibration board on touch surface
2. Align board edges with zone boundaries:
   - **Bottom edge**: Aligned with row 0 (bottom row)
   - **Top edge**: Aligned with row 1 (top row)
   - **Left edge**: At column 0
   - **Right edge**: At column 15
   - **Center**: Board center should be at origin (between columns 8/9)

**Visual Guide:**

```
Touch Zone Layout (Top View):

Row 1: [ 1  2  3  4  5  6  7  8 | 9  10 11 12 13 14 15 16]
       ┌────────────────────────┬────────────────────────┐
Row 0: │17 18 19 20 21 22 23 24 │25 26 27 28 29 30 31 32│
       └────────────────────────┴────────────────────────┘
                           Origin (center mark)

Place calibration board covering entire grid.
Mark the 4 corners clearly.
```

### Step 4: Select Calibration Points

For each camera, you'll select 4 corner points of the calibration board.

**Left Camera Calibration:**

1. Window shows left camera view with board visible
2. Click on **bottom-left corner** of board → Point marked with "1"
3. Click on **bottom-right corner** of board → Point marked with "2"
4. Click on **top-right corner** of board → Point marked with "3"
5. Click on **top-left corner** of board → Point marked with "4"

**Order is important**: Always go clockwise starting from bottom-left.

**Tips:**

- Zoom in for precise selection (use mouse wheel)
- Click exactly on the corner (not inside board edge)
- If you make a mistake, press `U` to undo last point
- Press `R` to restart selection

**Right Camera Calibration:**

Repeat the same process for the right camera view.

### Step 5: Review and Confirm

After selecting points for both cameras:

1. **Preview Display**: See the transformed view with zone grid overlay
2. **Verify Alignment**: Check that zones align with board edges
3. **Adjust if Needed**: Return to point selection if alignment is off

**Good Calibration Indicators:**

- Grid lines match board edges closely
- Origin mark at center of board
- No significant warping or distortion in transformed view

**Poor Calibration Indicators:**

- Grid lines don't match board
- Zones appear skewed or rotated
- Large gaps between grid and board edges

### Step 6: Height Calibration

After zone calibration, set height level thresholds:

1. **Level 0 (Surface)**: Tool auto-detects touch surface as Z=0
2. **Levels 1-5**: Place your hand at indicated heights:
   - Tool prompts: "Place hand at ~5cm height, press SPACE"
   - Position hand at requested height above surface
   - Press SPACE to record threshold
   - Repeat for each level (5cm, 10cm, 15cm, 20cm, 25cm)

**Tips:**

- Use a ruler or measuring tape for accuracy
- Hold hand steady for 1-2 seconds before pressing SPACE
- Place hand in center of touch area (near origin)

### Step 7: Save Calibration

1. Review all parameters
2. Enter calibration name (default: current date/time)
3. Confirm to save

Calibration saved to: `configs/calibration.yaml`

**Output:**

```yaml
version: "1.0"
timestamp: "2026-01-10T14:30:00"
camera_left_transform:
  - [1.2, 0.1, -50.0]
  - [0.0, 1.3, -30.0]
  - [0.001, 0.0, 1.0]
camera_right_transform:
  - [1.1, -0.1, -45.0]
  - [0.0, 1.2, -28.0]
  - [-0.001, 0.0, 1.0]
zone_boundaries: ...
height_thresholds: [0.0, 5.2, 10.1, 15.3, 20.4, 25.6]
stereo_baseline: 20.0
reference_board_size: [40.0, 10.0]
```

---

## Verification

### Test Touch Detection

After calibration, run test mode:

```bash
python -m chunivision.main --mode test
```

**Test Procedure:**

1. Touch each zone sequentially (1-32)
2. Verify correct zone detection in terminal output
3. Move hand through height levels (0-5)
4. Verify correct height detection

**Expected Output:**

```
[INFO] Zone 1 touched (confidence: 0.95)
[INFO] Zone 2 touched (confidence: 0.98)
...
[INFO] Height level 2 detected (15cm)
```

### Calibration Quality Metrics

The system provides quality metrics:

```
Calibration Quality Report:
  Reprojection Error: 1.2 pixels (Good)
  Zone Coverage: 99.5% (Excellent)
  Height Accuracy: ±2.1cm (Good)
  Overall Score: 94/100 (Grade A)
```

**Quality Grades:**

- **A (90-100)**: Excellent, ready for use
- **B (80-89)**: Good, acceptable for most use
- **C (70-79)**: Fair, consider recalibrating
- **D (<70)**: Poor, recalibration required

---

## Recalibration

Recalibrate when:

- **Cameras moved**: Any change in camera position/angle
- **Detection issues**: Zones not detecting correctly
- **After firmware updates**: Camera firmware changes
- **Periodic maintenance**: Recommended monthly

**Quick Recalibration:**

If only height thresholds need adjustment (cameras didn't move):

```bash
python -m chunivision.calibration --quick --heights-only
```

This skips zone calibration and only updates height thresholds.

---

## Advanced Calibration

### Multi-Point Calibration

For higher accuracy, use more than 4 points:

```bash
python -m chunivision.calibration --mode advanced --points 9
```

Select 9 points across the touch area for more precise transformation.

### Custom Zone Layout

If using non-standard zone layout:

1. Edit `configs/zone_config.yaml`:

```yaml
zone_layout:
  rows: 2
  cols: 16
  zone_width: 2.5  # cm
  zone_height: 5.0  # cm
  custom_zones:
    - id: 0
      corners: [[0, 0], [2.5, 0], [2.5, 5], [0, 5]]
    # Define each zone...
```

1. Run calibration with custom config:

```bash
python -m chunivision.calibration --config configs/zone_config.yaml
```

### Stereo Calibration Refinement

For improved depth accuracy:

```bash
python -m chunivision.calibration --stereo-refine
```

Uses checkerboard pattern to refine stereo camera parameters.

---

## Troubleshooting

### Issue: Grid doesn't align with board

**Causes:**

- Incorrect point selection order
- Camera moved during calibration
- Board not flat or not aligned

**Solutions:**

1. Verify point selection order (clockwise from bottom-left)
2. Ensure cameras are stable
3. Use rigid, flat calibration board
4. Retake calibration points

### Issue: Height detection inaccurate

**Causes:**

- Hand not steady during height calibration
- Lighting changes
- Stereo baseline incorrect

**Solutions:**

1. Recalibrate heights in consistent lighting
2. Hold hand very steady during threshold recording
3. Verify stereo baseline measurement in config

### Issue: Edge zones not detecting

**Causes:**

- Cameras don't see entire area
- Point selection cut off edges

**Solutions:**

1. Increase camera height or adjust angles
2. Select points at absolute board edges, not inside
3. Verify both cameras can see all corners

### Issue: Frequent false positives

**Causes:**

- Background objects in view
- Reflective surfaces nearby
- Touch threshold too high

**Solutions:**

1. Clear background of unnecessary objects
2. Cover reflective surfaces
3. Lower touch threshold in config:

```yaml
vision:
  touch_threshold_z: 1.5  # Lower from default 2.0
```

---

## Best Practices

1. **Lighting**: Calibrate in same lighting as gameplay
2. **Stability**: Don't touch/move cameras after calibration
3. **Documentation**: Note camera positions and settings
4. **Backup**: Keep multiple calibration profiles for different setups
5. **Regular Testing**: Verify calibration before important sessions

---

## Calibration Profiles

Save multiple calibrations for different scenarios:

```bash
# Tournament setup
python -m chunivision.calibration --save tournament_2026_01

# Home setup
python -m chunivision.calibration --save home_setup

# Load specific calibration
python -m chunivision.main --calibration tournament_2026_01
```

Profiles stored in: `configs/calibrations/`

---

## Automated Calibration (Experimental)

Future feature using computer vision to auto-detect calibration board:

```bash
python -m chunivision.calibration --auto
```

Automatically detects board corners using edge detection and pattern matching.

---

## Summary

Proper calibration is critical for accurate touch detection. Take your time during calibration, verify results thoroughly, and recalibrate if cameras are moved or detection seems off.

For questions or issues, see [DEVELOPMENT.md](DEVELOPMENT.md#debugging) for debugging tips.
