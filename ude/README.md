# ChunithmUDE Driver

Windows kernel-mode USB Device Emulation (UDE) driver that creates a virtual Chunithm controller for ChunIVision integration.

## Overview

ChunithmUDE is a Windows kernel driver that uses the USB Device Emulation framework to present a virtual Chunithm controller (VID: 0x1973, PID: 0x2001) to the operating system. This allows ChunIVision's vision processing output to be recognized by games without requiring physical USB hardware.

**Platform**: Windows 10/11 (64-bit)  
**Author**: Misaka 19465

## Features

- Emulates official Chunithm controller USB device
- Kernel-mode driver with Python bindings
- Low latency (~1-2ms) from Python to USB stack
- Zero-dependency Python interface (pure ctypes)
- Support for 32 touch zones and 6 air sensors

## Architecture

```
┌─────────────────────────────────────────┐
│  ChunIVision (Python User Mode)        │
│  - Vision processing                    │
│  - Touch/air sensor detection          │
└──────────────┬──────────────────────────┘
               │
               │ IOCTL calls
               ▼
┌─────────────────────────────────────────┐
│  UDE Python Binding (ctypes/cffi)      │
│  - Device handle management             │
│  - HID report formatting                │
│  - Error handling                       │
└──────────────┬──────────────────────────┘
               │
               │ DeviceIoControl
               ▼
┌─────────────────────────────────────────┐
│  ChunithmUDE.sys (Kernel Driver)       │
│  - USB device emulation                 │
│  - HID report dispatch                  │
│  - Interrupt endpoint handling          │
└──────────────┬──────────────────────────┘
               │
               │ UDE Framework
               ▼
┌─────────────────────────────────────────┐
│  Windows USB Stack                      │
│  - Game application reads HID reports   │
└─────────────────────────────────────────┘
```

## Project Structure

```
ude/
├── driver/             Kernel-mode driver source
│   ├── ChunithmUDE.c/h
│   ├── Device.c
│   ├── Queue.c
│   ├── Usb.c
│   ├── UsbCallbacks.c
│   ├── EndpointQueue.c
│   ├── Descriptors.c
│   └── ChunithmUDE.inf
├── python/             Python bindings
│   ├── ude_device.py
│   ├── constants.py
│   └── exceptions.py
├── build/              Build scripts
├── tests/              Unit tests
├── examples/           Usage examples
└── docs/               Technical documentation
```

## Quick Start

### Prerequisites

- Windows 10/11 (64-bit)
- Visual Studio 2019+ with C++ workload
- Windows Driver Kit (WDK) 10.0.22621.0 or later
- Python 3.8+

### Build Driver

```cmd
cd ude\build
build_driver.cmd
```

### Install Driver

Enable test signing (requires administrator and reboot):

```cmd
bcdedit /set testsigning on
shutdown /r /t 0
```

After reboot, install the driver:

```cmd
cd ude\build
install_driver.cmd
```

### Python Usage

```python
from ude.python import ChunithmUDEDevice

# Open device
device = ChunithmUDEDevice()
device.open()

# Send controller state
touch_zones = [False] * 32  # 32 touch zones
air_sensors = [0] * 6        # 6 air sensors

touch_zones[0] = True   # Zone 1 pressed
air_sensors[0] = 128    # Air sensor 1 active

device.send_report(touch_zones, air_sensors)

# Cleanup
device.close()

# Or use context manager
with ChunithmUDEDevice() as device:
    device.send_report([False]*32, [0]*6)
```

## USB Device Specification

| Property | Value |
|----------|-------|
| Vendor ID | 0x1973 (ZHOUSENSOR) |
| Product ID | 0x2001 (YubiDeck) |
| Manufacturer | "ZHOUSENSOR I/O SYSTEM" |
| Product | "ZhouSensor YubiDeck" |
| Serial | "OK" |
| Class | HID (0x03) |
| Polling Rate | 1ms (1000 Hz) |

### Endpoints

- **EP0**: Control endpoint (bidirectional, 64 bytes)
- **EP1 IN**: Interrupt endpoint (45 bytes, HID input reports)
- **EP2 OUT**: Interrupt endpoint (61 bytes, RGB output - unused)

### HID Report Format

Input report (45 bytes, device to host):

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | IRValue | Air sensors (6 bits) |
| 1 | 1 | Buttons | Function buttons (unused) |
| 2-33 | 32 | TouchValue[] | Touch zones (0x00=off, 0x64=on) |
| 34 | 1 | CardStatus | Card reader status (always 0) |
| 35-44 | 10 | CardID[] | Card ID (always 0) |

## Python API Reference

### ChunithmUDEDevice

Main device interface class.

#### Methods

- `open()` - Open connection to driver
- `send_report(touch_zones: List[bool], air_sensors: List[int])` - Send controller state
- `get_status() -> dict` - Get driver statistics
- `reset()` - Reset device state
- `close()` - Close connection

#### Context Manager

```python
with ChunithmUDEDevice() as device:
    device.send_report(touch_zones, air_sensors)
```

### Exceptions

- `DeviceNotFoundError` - Driver not loaded or device unavailable
- `DeviceNotReadyError` - Device not in ready state
- `InvalidReportError` - Report validation failed
- `DriverCommunicationError` - IOCTL communication error

## Integration

See [INTEGRATION.md](INTEGRATION.md) for detailed integration guide with ChunIVision.

## Technical Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Driver architecture and implementation details
- [INTEGRATION.md](INTEGRATION.md) - ChunIVision integration patterns

## Troubleshooting

### Driver won't load

Check if test signing is enabled:

```cmd
bcdedit /enum {current} | findstr testsigning
```

Should show: `testsigning Yes`

### Python cannot find device

Verify driver is running:

```cmd
sc query ChunithmUDE
```

Check Device Manager under "Human Interface Devices" for "Chunithm USB Device Emulation".

### Game doesn't recognize controller

1. Verify device appears in Device Manager
2. Check VID:PID with USBDeview (should be 0x1973:0x2001)
3. Ensure HID class driver is loaded

## Performance

- Latency: ~1-2ms from Python to USB stack
- Maximum throughput: 1000 reports/second
- CPU usage: <0.1% at 60 FPS
- Memory footprint: ~100KB kernel pool

## Security Notes

Test signing mode reduces Windows security. For production use:

- Obtain a valid code signing certificate
- Sign driver with production certificate
- Disable test signing mode
- Consider WHQL certification

## License

Part of the ChunIVision project by Misaka 19465.

## References

- [Windows UDE Framework](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/usb-emulated-device--ude--architecture)
- [WDF Driver Development](https://learn.microsoft.com/en-us/windows-hardware/drivers/wdf/)
- [HID Usage Tables](https://www.usb.org/sites/default/files/hut1_3_0.pdf)
