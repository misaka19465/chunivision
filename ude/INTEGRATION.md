# ChunIVision Integration Guide

This guide explains how to integrate the UDE driver into the ChunIVision output pipeline.

## Python Integration

```python
from ude.python import ChunithmUDEDevice, DeviceNotFoundError

# Initialize device
try:
    device = ChunithmUDEDevice()
    device.open()
except DeviceNotFoundError:
    print("UDE driver not installed or device not ready")
    exit(1)

# Send controller state
touch_zones = [False] * 32  # All zones released
air_sensors = [0] * 6        # All air sensors idle

touch_zones[0] = True   # Zone 1 touched
air_sensors[0] = 100    # Air sensor 1 at height

device.send_report(touch_zones, air_sensors)

# Cleanup
device.close()
```

## Adding to OutputManager

```python
# src/output/output_manager.py
from ude.python import ChunithmUDEDevice

class OutputManager:
    def __init__(self, config):
        self.devices = []
        
        # Add UDE adapter if enabled
        if config.get('output.hid.enabled', False):
            try:
                ude_device = ChunithmUDEDevice()
                ude_device.open()
                self.devices.append(ude_device)
            except Exception as e:
                logger.warning(f"Failed to initialize UDE: {e}")
    
    def send_state(self, touch_zones, air_sensors):
        for device in self.devices:
            try:
                device.send_report(touch_zones, air_sensors)
            except Exception as e:
                logger.error(f"Failed to send report: {e}")
```

## Configuration

Add UDE output to your ChunIVision configuration:

```yaml
# configs/default.yaml
output:
  hid:
    enabled: true
    device_path: "auto"  # Auto-detect device
```

## Requirements

- Windows 10/11 (64-bit)
- ChunithmUDE driver installed and running
- Python 3.8+

## Installation

See main README.md for driver build and installation instructions.

## Troubleshooting

**Driver not found**:

```cmd
sc query ChunithmUDE
```

**Check device status**:

```python
from ude.python import ChunithmUDEDevice
device = ChunithmUDEDevice()
try:
    device.open()
    print("Driver is working")
    device.close()
except Exception as e:
    print(f"Error: {e}")
```

**View driver logs**:

- Use DbgView from Sysinternals
- Check Event Viewer → Windows Logs → System
