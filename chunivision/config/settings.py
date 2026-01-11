"""
Global application settings for ChunIVision.

Provides centralized configuration management with:
- YAML file loading and saving
- Environment variable overrides
- Validation
- Default values
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import yaml
import os


@dataclass
class AppSettings:
    """Application metadata and mode settings."""

    name: str = "ChunIVision"
    version: str = "1.0.0"
    mode: str = "run"  # Options: "run", "calibration", "debug", "test"


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    rotation: bool = True
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class CameraSettings:
    """Camera hardware configuration."""

    # USB serial numbers for camera identification
    left_camera_serial: str = ""
    right_camera_serial: str = ""

    resolution: List[int] = field(default_factory=lambda: [640, 480])
    fps: int = 60
    exposure: int = -1
    left_camera_position: List[float] = field(
        default_factory=lambda: [-10.0, 15.0, 30.0]
    )
    right_camera_position: List[float] = field(
        default_factory=lambda: [10.0, 15.0, 30.0]
    )
    baseline_distance: float = 20.0
    left_camera_angle: float = 45.0
    right_camera_angle: float = 45.0


@dataclass
class VisionSettings:
    """Vision processing configuration."""

    touch_threshold_z: float = 2.0
    min_hand_size: int = 100
    max_hands: int = 10
    height_levels: int = 6
    height_thresholds: List[float] = field(
        default_factory=lambda: [17.9, 21.3, 24.7, 28.1, 31.5, 34.9]
    )
    height_hysteresis: float = 1.0
    stereo_algorithm: str = "SGBM"
    min_depth: float = 0.0
    max_depth: float = 35.0
    enable_gpu: bool = False
    num_threads: int = 4


@dataclass
class ZoneSettings:
    """Touch zone layout configuration."""

    num_rows: int = 2
    num_cols: int = 16
    total_zones: int = 32
    zone_width: float = 2.75
    zone_height: float = 4.5
    origin_x: float = 22.0
    origin_y: float = 0.0


@dataclass
class OutputSettings:
    """Output adapter configurations."""

    serial: Dict[str, Any] = field(default_factory=dict)
    hid: Dict[str, Any] = field(default_factory=dict)
    keyboard: Dict[str, Any] = field(default_factory=dict)
    udp: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSettings:
    """Performance monitoring configuration."""

    enabled: bool = True
    log_interval: float = 5.0
    fps_warning_threshold: int = 50
    latency_warning_threshold: int = 15


@dataclass
class CalibrationSettings:
    """Calibration configuration."""

    file: str = "configs/calibration.yaml"
    auto_load: bool = True
    validation_threshold: float = 0.7
    board_width: float = 40.0
    board_height: float = 10.0


@dataclass
class DebugSettings:
    """Debug and visualization options."""

    show_camera_view: bool = False
    show_depth_map: bool = False
    show_detections: bool = False
    save_debug_frames: bool = False
    debug_output_dir: str = "debug_output/"
    visualize_zones: bool = False


@dataclass
class PathSettings:
    """Application paths."""

    config_dir: str = "configs/"
    calibration_dir: str = "configs/calibrations/"
    log_dir: str = "logs/"
    data_dir: str = "data/"


@dataclass
class Settings:
    """
    Global application settings.

    Provides centralized configuration management for all ChunIVision components.
    """

    app: AppSettings = field(default_factory=AppSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    zones: ZoneSettings = field(default_factory=ZoneSettings)
    outputs: OutputSettings = field(default_factory=OutputSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    @classmethod
    def load(cls, path: str) -> "Settings":
        """
        Load settings from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Settings instance with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)

        # Create settings with nested dataclasses
        settings = cls(
            app=AppSettings(**data.get("app", {})),
            logging=LoggingSettings(**data.get("logging", {})),
            camera=CameraSettings(**data.get("camera", {})),
            vision=VisionSettings(**data.get("vision", {})),
            zones=ZoneSettings(**data.get("zones", {})),
            outputs=OutputSettings(**data.get("outputs", {})),
            performance=PerformanceSettings(**data.get("performance", {})),
            calibration=CalibrationSettings(**data.get("calibration", {})),
            debug=DebugSettings(**data.get("debug", {})),
            paths=PathSettings(**data.get("paths", {})),
        )

        return settings

    def save(self, path: str) -> None:
        """
        Save settings to YAML file.

        Args:
            path: Path to output YAML file
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        data = {
            "app": asdict(self.app),
            "logging": asdict(self.logging),
            "camera": asdict(self.camera),
            "vision": asdict(self.vision),
            "zones": asdict(self.zones),
            "outputs": asdict(self.outputs),
            "performance": asdict(self.performance),
            "calibration": asdict(self.calibration),
            "debug": asdict(self.debug),
            "paths": asdict(self.paths),
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables should be named: CHUNIVISION_SECTION_KEY
        Examples:
            CHUNIVISION_LOGGING_LEVEL=DEBUG
            CHUNIVISION_CAMERA_FPS=120

        Args:
            data: Configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        prefix = "CHUNIVISION_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Parse environment variable name
            key_parts = env_key[len(prefix) :].lower().split("_", 1)
            if len(key_parts) != 2:
                continue

            section, key = key_parts

            if section not in data:
                data[section] = {}

            # Convert value to appropriate type
            converted_value = Settings._convert_env_value(env_value)
            data[section][key] = converted_value

        return data

    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """
        Convert environment variable string to appropriate Python type.

        Args:
            value: Environment variable value

        Returns:
            Converted value (bool, int, float, or str)
        """
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    def validate(self) -> List[str]:
        """
        Validate settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate mode
        valid_modes = ["run", "calibration", "debug", "test"]
        if self.app.mode not in valid_modes:
            errors.append(
                f"Invalid mode '{self.app.mode}'. Must be one of {valid_modes}"
            )

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_levels:
            errors.append(
                f"Invalid log level '{self.logging.level}'. Must be one of {valid_levels}"
            )

        # Validate camera settings
        if self.camera.fps <= 0:
            errors.append(f"Invalid FPS {self.camera.fps}. Must be > 0")

        if len(self.camera.resolution) != 2:
            errors.append(
                f"Invalid resolution {self.camera.resolution}. Must be [width, height]"
            )

        # Validate zone configuration
        if self.zones.num_rows * self.zones.num_cols != self.zones.total_zones:
            errors.append(
                f"Zone count mismatch: {self.zones.num_rows} * {self.zones.num_cols} != {self.zones.total_zones}"
            )

        # Validate vision settings
        if len(self.vision.height_thresholds) != self.vision.height_levels:
            errors.append(
                f"Height threshold count mismatch: {len(self.vision.height_thresholds)} != {self.vision.height_levels}"
            )

        return errors
