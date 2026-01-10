# Getting Started with ChunIVision Development

**Author**: Misaka 19465  
**Platform**: Windows Only

This guide helps you start implementing the ChunIVision framework.

## Project Status

**Framework Status**: ✅ Complete  
**Implementation Status**: 🚧 Ready for development  
**Target Platform**: Windows 10/11 only

The complete modular framework has been designed and documented. All module interfaces, data structures, and protocols are specified. Now ready for implementation.

## System Requirements

### Hardware

- **Windows PC**: Windows 10/11 (64-bit)
- **Dual IR Cameras**: 2× compatible with oculus library
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Quad-core processor (for real-time processing)
- **USB Ports**: 2× USB 3.0 for cameras

### Software

- **Python**: 3.8 or higher
- **Windows SDK**: For HID device emulation
- **Visual Studio Build Tools**: For compiling Python extensions (if needed)
- **com0com**: For virtual serial port emulation (optional)
- **ViGEm**: For HID controller emulation (optional)

## What's Been Created

### 1. Directory Structure

```
chunivision/
├── chunivision/          # Main package with module structure
│   ├── vision/          # Vision processing modules
│   ├── calibration/     # Calibration system
│   ├── output/          # Output adapters
│   ├── config/          # Configuration management
│   └── utils/           # Utility modules
├── docs/                # Comprehensive documentation
├── configs/             # Configuration files
└── tests/              # Test structure
```

### 2. Documentation (in `docs/`)

- **README.md**: Project overview, quick start, architecture diagram
- **ARCHITECTURE.md**: Detailed system architecture and data flow
- **MODULE_SPECS.md**: Complete API specifications for all modules
- **OUTPUT_PROTOCOLS.md**: Communication protocol specifications
- **DEVELOPMENT.md**: Development guidelines and best practices
- **CALIBRATION_GUIDE.md**: User guide for calibration process

### 3. Configuration

- **configs/default.yaml**: Complete configuration template
- **requirements.txt**: All Python dependencies
- **setup.py**: Package installation configuration

### 4. Module Interfaces

All module classes are specified with:

- Clear responsibilities
- Type-hinted method signatures
- Docstring documentation
- Error handling strategies
- Configuration parameters

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Get basic infrastructure working

1. **Implement core utilities** (`chunivision/utils/`)
   - `logger.py`: Logging system
   - `geometry.py`: Coordinate transformations
   - `state_manager.py`: State tracking

2. **Implement configuration** (`chunivision/config/`)
   - `settings.py`: YAML loading/saving
   - `zone_config.py`: Zone layout
   - `camera_config.py`: Camera parameters

3. **Basic tests**
   - Test configuration loading
   - Test utility functions

### Phase 2: Camera & Vision (Weeks 3-5)

**Goal**: Get camera capture and basic detection working

1. **Camera management** (`chunivision/vision/`)
   - `camera_manager.py`: Interface with oculus library
   - Test dual camera capture and synchronization

2. **Stereo processing**
   - `stereo_processor.py`: Depth map generation
   - Integrate OpenCV stereo algorithms

3. **Hand detection**
   - `hand_detector.py`: Basic clustering algorithm
   - `touch_detector.py`: Zone mapping
   - `height_estimator.py`: Height level assignment

4. **Vision pipeline**
   - `vision_pipeline.py`: Orchestrate all components
   - Test end-to-end processing

### Phase 3: Calibration (Weeks 6-7)

**Goal**: Interactive calibration system

1. **Calibration UI** (`chunivision/calibration/`)
   - `zone_selector.py`: Point selection GUI
   - `transform_calculator.py`: Perspective transform

2. **Calibration workflow**
   - `calibrator.py`: Orchestrate calibration process
   - `calibration_data.py`: Save/load calibration

3. **Testing**
   - Test calibration with real cameras
   - Verify transformation accuracy

### Phase 4: Output Adapters (Weeks 8-9)

**Goal**: Implement all communication methods

1. **Base infrastructure** (`chunivision/output/`)
   - `base_output.py`: Abstract base class
   - `output_manager.py`: Multi-output management

2. **Implement adapters** (in priority order)
   - `udp_output.py`: Highest priority (lowest latency)
   - `serial_output.py`: Second priority
   - `hid_output.py`: Third priority
   - `keyboard_output.py`: Fourth priority

3. **Protocol testing**
   - Test each protocol independently
   - Test multi-output mode

### Phase 5: Integration & Testing (Weeks 10-11)

**Goal**: Complete system integration

1. **Main application**
   - `main.py`: Application entry point
   - Command-line interface
   - Mode switching (run/calibration/debug)

2. **Integration testing**
   - End-to-end workflow tests
   - Performance benchmarking
   - Latency measurement

3. **Game integration**
   - Test with actual game
   - Fine-tune parameters
   - Optimize performance

### Phase 6: Polish & Documentation (Week 12)

**Goal**: Production-ready release

1. **Performance optimization**
   - Profile and optimize bottlenecks
   - GPU acceleration (optional)
   - Multi-threading optimization

2. **User documentation**
   - Installation guide
   - Troubleshooting guide
   - Video tutorials (optional)

3. **Release preparation**
   - Package for distribution
   - Create installation scripts
   - Final testing

## Development Tips

### Start Small

Don't implement everything at once. Start with:

1. Configuration loading
2. Basic camera capture (single camera first)
3. Simple visualization
4. Gradually add complexity

### Use Existing Libraries

- **OpenCV**: Stereo vision, calibration
- **NumPy**: Fast array operations
- **SciPy**: Clustering, signal processing

### Test Incrementally

Write tests as you implement:

```python
# Example: Test camera manager
def test_camera_initialization():
    config = CameraConfig(left_index=0, right_index=1)
    manager = CameraManager(config)
    assert manager.initialize()
    assert manager.is_ready()
```

### Use Debug Visualization

Display intermediate results:

```python
cv2.imshow("Depth Map", depth_map)
cv2.imshow("Detections", annotated_frame)
cv2.waitKey(1)
```

### AI-Assisted Development

This framework is designed for AI assistance:

- Clear module boundaries
- Comprehensive docstrings
- Type hints throughout
- Example implementations in docs

**Effective prompts for AI:**

- "Implement the HandDetector class according to MODULE_SPECS.md"
- "Add unit tests for the StereoProcessor module"
- "Optimize the vision pipeline for lower latency"

## Quick Start Commands

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests (when implemented)
pytest tests/
```

### Run Application (After Implementation)

```bash
# Calibration mode
python -m chunivision.main --mode calibration

# Run mode with UDP output
python -m chunivision.main --mode run --output udp

# Debug mode
python -m chunivision.main --mode debug --log-level DEBUG
```

## Key Design Decisions

### Why This Architecture?

1. **Modularity**: Each component is independent and replaceable
   - Example: Swap stereo algorithm without changing other code

2. **Extensibility**: Easy to add new features
   - Example: Add gesture recognition by inserting new module in pipeline

3. **Testability**: Each module can be tested independently
   - Example: Mock camera input for testing vision pipeline

4. **Performance**: Architecture supports optimization
   - Example: Multi-threading, GPU acceleration, caching

5. **AI-Friendly**: Clear interfaces and documentation
   - Example: AI can implement one module at a time

### Design Patterns Used

- **Pipeline Pattern**: Vision processing as sequential stages
- **Adapter Pattern**: Multiple output protocols with common interface
- **Observer Pattern**: State change callbacks
- **Strategy Pattern**: Configurable algorithms (stereo matching, etc.)
- **Singleton Pattern**: Logger, performance monitor

## Common Implementation Challenges

### 1. Camera Synchronization

**Challenge**: Ensuring frames from both cameras are captured at same time  
**Solution**: Use hardware sync if available, or timestamp-based matching

### 2. Real-time Performance

**Challenge**: Meeting <10ms latency target  
**Solution**: Optimize critical path, use compiled code (Cython), GPU acceleration

### 3. Calibration Accuracy

**Challenge**: Users struggle with precise point selection  
**Solution**: Implement zoom, undo, visual feedback in calibration UI

### 4. Zone Boundary Handling

**Challenge**: Ambiguous zone assignment at boundaries  
**Solution**: Use smooth transitions, confidence scoring

### 5. Height Estimation Stability

**Challenge**: Noisy height readings  
**Solution**: Implement hysteresis, Kalman filtering, temporal smoothing

## Resources

### Computer Vision

- OpenCV documentation: <https://docs.opencv.org/>
- Stereo vision tutorials: Search "OpenCV stereo calibration"

### Python Development

- Type hints: PEP 484
- Async programming: `asyncio` for concurrent operations
- Performance: `cProfile` for profiling

### Hardware

- Infrared camera specifications
- Linux uinput documentation (for HID/keyboard)
- UDP socket programming

## Getting Help

1. **Read the docs**: Start with ARCHITECTURE.md and MODULE_SPECS.md
2. **Check examples**: Look at test templates in tests/
3. **Use AI**: Provide context from documentation
4. **Debug incrementally**: Test each component separately

## Success Criteria

You'll know the implementation is successful when:

✅ Both cameras capture synchronized frames at 60 FPS  
✅ Calibration produces accurate zone mapping  
✅ Touch detection has >95% accuracy  
✅ Height estimation is within ±2cm  
✅ End-to-end latency is <10ms  
✅ All output protocols work correctly  
✅ System runs stably for hours without issues  

## Next Steps

1. **Review all documentation** in `docs/` folder
2. **Set up development environment** with dependencies
3. **Start with Phase 1** (foundation utilities)
4. **Implement incrementally**, testing as you go
5. **Use AI assistance** with context from docs

Good luck with your ChunIVision development! The framework is designed to be robust, extensible, and AI-friendly. Take it one module at a time, and you'll have a working vision-based controller.
