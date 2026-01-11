"""
Calibration system for ChunIVision.

This module provides a comprehensive two-stage calibration workflow:

1. Lens Distortion Correction:
   - Uses fixed factory-calibrated parameters for Oculus Rift CV1 cameras
   - Handles radial and tangential distortion

2. Stage 1 - Chessboard Calibration:
   - User places A4 printed chessboard on horizontal surface
   - Computes camera intrinsics and stereo geometry
   - Quality-first approach - continues until thresholds are met

3. Stage 2 - Touch Zone Calibration:
   - User places white paper matching game touch area size
   - Computes perspective transform for touch zone mapping
   - Validates by overlaying zone grid on camera view
"""

from .calibration_data import CalibrationData, CalibrationDataError
from .calibrator import (
    CalibrationConfig,
    CalibrationError,
    Calibrator,
    FullCalibrationResult,
)
from .chessboard_calibration import (
    ChessboardCalibrator,
    ChessboardCalibrationResult,
    ChessboardCapture,
    ChessboardConfig,
    StereoCalibrationResult,
)
from .lens_distortion import LensDistortion, LensDistortionParams
from .touchzone_calibration import (
    TouchZoneCalibrator,
    TouchZoneCalibrationResult,
    TouchZoneConfig,
    ZoneBoundary,
)
from .transform_calculator import TransformCalculator
from .zone_selector import SelectionState, ZoneSelector

__all__ = [
    # Main calibrator
    "Calibrator",
    "CalibrationConfig",
    "CalibrationError",
    "FullCalibrationResult",
    # Calibration data
    "CalibrationData",
    "CalibrationDataError",
    # Lens distortion
    "LensDistortion",
    "LensDistortionParams",
    # Chessboard calibration
    "ChessboardCalibrator",
    "ChessboardCalibrationResult",
    "ChessboardCapture",
    "ChessboardConfig",
    "StereoCalibrationResult",
    # Touch zone calibration
    "TouchZoneCalibrator",
    "TouchZoneCalibrationResult",
    "TouchZoneConfig",
    "ZoneBoundary",
    # Utilities
    "TransformCalculator",
    "SelectionState",
    "ZoneSelector",
]
