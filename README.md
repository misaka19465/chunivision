# ChunIVision

A vision-based controller system for Chunithm rhythm game using dual infrared cameras for touch detection and hand height tracking.

## Project Overview

This project replaces traditional touch sensors and infrared height detection with pure computer vision, using two wide-angle infrared cameras positioned at approximately 45-degree angles pointing toward the origin (one front-left, one front-right).

### Key Features

- **32 Touch Zones**: 2 rows × 16 columns (origin at column 8/9 boundary, bottom row)
- **Vision-based Height Detection**: Emulates 6 height levels traditionally detected by infrared sensors
- **Dual Camera System**: Stereo vision using oculus library for distortion-corrected camera input
- **Flexible Output**: Multiple communication protocols (Virtual Serial, HID, Keyboard, UDP)
- **Interactive Calibration**: User-guided touch zone calibration with rectangular reference board
- **Modular Architecture**: Easy to extend, replace, or modify individual components

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ChunIVision System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌─────────────────┐                   │
│  │   Camera     │────▶│  Vision         │                   │
│  │   Manager    │     │  Processing     │                   │
│  │ (oculus lib) │     │  Pipeline       │                   │
│  └──────────────┘     └────────┬────────┘                   │
│                                 │                             │
│                                 ▼                             │
│                    ┌────────────────────┐                    │
│                    │  Touch Detection   │                    │
│                    │  & Height Tracking │                    │
│                    └─────────┬──────────┘                    │
│                              │                               │
│                              ▼                               │
│                    ┌────────────────────┐                    │
│                    │  State Manager     │                    │
│                    │  (32 zones + 6 ht) │                    │
│                    └─────────┬──────────┘                    │
│                              │                               │
│         ┌────────────────────┼────────────────────┐         │
│         ▼                    ▼                    ▼          │
│  ┌──────────┐      ┌──────────────┐      ┌──────────┐      │
│  │ Virtual  │      │   Virtual    │      │   UDP    │      │
│  │  Serial  │      │  HID/Keyboard│      │  Output  │      │
│  └──────────┘      └──────────────┘      └──────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Calibration System                         │   │
│  │  (User-guided zone selection with reference board)   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
chunivision/
├── chunivision/              # Main package
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   │
│   ├── vision/              # Vision processing modules
│   │   ├── __init__.py
│   │   ├── camera_manager.py      # Camera initialization and frame capture
│   │   ├── stereo_processor.py    # Dual camera stereo processing
│   │   ├── hand_detector.py       # Hand detection and tracking
│   │   ├── touch_detector.py      # Touch zone detection
│   │   ├── height_estimator.py    # Height level estimation (6 levels)
│   │   └── vision_pipeline.py     # Orchestrates vision processing
│   │
│   ├── calibration/         # Calibration system
│   │   ├── __init__.py
│   │   ├── calibrator.py          # Main calibration orchestrator
│   │   ├── zone_selector.py       # Interactive zone selection UI
│   │   ├── transform_calculator.py # Perspective transform calculation
│   │   └── calibration_data.py    # Calibration data storage/loading
│   │
│   ├── output/              # Output adapters
│   │   ├── __init__.py
│   │   ├── base_output.py         # Abstract base class for outputs
│   │   ├── serial_output.py       # Virtual serial port output
│   │   ├── hid_output.py          # Virtual HID device output
│   │   ├── keyboard_output.py     # Virtual keyboard output
│   │   ├── udp_output.py          # UDP network output
│   │   └── output_manager.py      # Manages multiple output channels
│   │
│   ├── config/              # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py            # Application settings
│   │   ├── zone_config.py         # Touch zone layout configuration
│   │   └── camera_config.py       # Camera parameters
│   │
│   └── utils/               # Utility modules
│       ├── __init__.py
│       ├── logger.py              # Logging utilities
│       ├── performance.py         # Performance monitoring
│       ├── geometry.py            # Geometric calculations
│       └── state_manager.py       # Game state management
│
    ├── oculus/              # Camera library (Oculus Rift CV1)
│   ├── __init__.py
│   ├── camera.py
│   ├── triple_buffer.py
│   └── viewer.py
│
├── tests/                   # Unit and integration tests
│   ├── test_vision/
│   ├── test_calibration/
│   ├── test_output/
│   └── test_integration.py
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # Detailed architecture documentation
│   ├── MODULE_SPECS.md      # Module specifications and APIs
│   ├── CALIBRATION_GUIDE.md # Calibration process guide
│   ├── OUTPUT_PROTOCOLS.md  # Output protocol specifications
│   └── DEVELOPMENT.md       # Development guidelines
│
├── configs/                 # Configuration files
│   ├── default.yaml         # Default configuration
│   ├── calibration.yaml     # Calibration settings
│   └── camera.yaml          # Camera settings
│
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
└── README.md               # This file
```

## Core Modules

### Vision Processing (`chunivision/vision/`)

- **camera_manager**: Interfaces with oculus library for dual camera setup
- **stereo_processor**: Combines images from both cameras for 3D position estimation
- **hand_detector**: Detects hands in the camera field of view
- **touch_detector**: Determines which of the 32 zones are being touched
- **height_estimator**: Estimates hand height to emulate 6-level infrared sensors
- **vision_pipeline**: Coordinates all vision processing components

### Calibration System (`chunivision/calibration/`)

- **calibrator**: Main calibration workflow controller
- **zone_selector**: GUI for user to select 4 corner points of reference board
- **transform_calculator**: Computes perspective transformation matrices
- **calibration_data**: Persists and loads calibration data

### Output Adapters (`chunivision/output/`)

- **base_output**: Abstract interface all outputs must implement
- **serial_output**: Virtual serial port (e.g., using pty or pyserial)
- **hid_output**: Virtual HID device (e.g., using uhid)
- **keyboard_output**: Virtual keyboard (e.g., using uinput or pynput)
- **udp_output**: UDP network protocol
- **output_manager**: Allows simultaneous multiple outputs

### Configuration (`chunivision/config/`)

- **settings**: Global application settings
- **zone_config**: Touch zone layout (2×16 grid, origin position)
- **camera_config**: Camera positions, angles, resolution, etc.

### Utilities (`chunivision/utils/`)

- **logger**: Centralized logging with configurable levels
- **performance**: FPS monitoring, latency measurement
- **geometry**: Coordinate transformations, zone mapping
- **state_manager**: Tracks current touch/height state, detects changes

## Design Principles

### 1. Modularity

- Each module has a single, well-defined responsibility
- Modules communicate through clear interfaces
- Easy to replace any module (e.g., swap UDP output for serial output)

### 2. Extensibility

- Output adapters follow plugin architecture (inherit from `base_output`)
- Vision processing pipeline is a sequence of pluggable processors
- Configuration-driven behavior (minimal hardcoding)

### 3. Robustness

- Error handling at module boundaries
- Graceful degradation (e.g., fallback to single camera if one fails)
- State validation before sending outputs
- Performance monitoring to detect issues

### 4. AI-Friendly Documentation

- Clear module responsibilities in docstrings
- Type hints throughout codebase
- Detailed comments explaining "why" not just "what"
- Separation of configuration from code

## Touch Zone Layout

```
User facing forward (↓)

Row 1: [32][30][28][26][24][22][20][18]|[16][14][12][10][ 8][ 6][ 4][ 2]
       ─────────────────────────────────┼─────────────────────────────────
Row 0: [31][29][27][25][23][21][19][17]|[15][13][11][ 9][ 7][ 5][ 3][ 1]
                                        ▲
                                     Origin
                    (Zone 1 at bottom-right corner)

Zone Dimensions: 27.5mm wide × 45mm high
Bottom row (0): Odd numbers (1, 3, 5, ..., 31) from right to left
Top row (1): Even numbers (2, 4, 6, ..., 32) from right to left
```

## Height Levels (Air Sensors)

Six height levels emulate the traditional infrared air sensor array:

- Air 0 (Level 0): 17.9cm above touch surface
- Air 1 (Level 1): 21.3cm (17.9 + 3.4)
- Air 2 (Level 2): 24.7cm (17.9 + 6.8)
- Air 3 (Level 3): 28.1cm (17.9 + 10.2)
- Air 4 (Level 4): 31.5cm (17.9 + 13.6)
- Air 5 (Level 5): 34.9cm (17.9 + 17.0)

Each sensor is positioned 3.4cm apart vertically.

## Quick Start

### Prerequisites

- Python 3.8+
- Two infrared cameras compatible with oculus library
- Windows system (required for HID/keyboard output)

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Calibration

```bash
python -m chunivision.calibration
# Follow on-screen instructions to select 4 corners of reference board
```

### Running

```bash
python -m chunivision.main --output udp --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to customize:

- Camera indices and parameters
- Touch zone dimensions and positions
- Output protocol selection
- Vision processing parameters
- Performance tuning

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:

- Code style guidelines
- Adding new output adapters
- Extending vision processing pipeline
- Writing tests
- Contributing guidelines

## Performance Targets

- **Latency**: < 10ms from hand movement to output
- **Frame Rate**: ≥ 60 FPS processing
- **Touch Accuracy**: ≥ 95% detection rate
- **Height Precision**: ± 2cm (within calibrated range)

## License

[To be determined]

## Acknowledgments

- Uses oculus library for camera distortion correction and frame capture
