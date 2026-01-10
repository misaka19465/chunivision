# Output Protocol Specifications

This document details the communication protocols for each output adapter, allowing game integration via multiple methods.

**Author**: Misaka 19465  
**Platform**: Windows only  
**Note**: RGB/LED data is not implemented as this controller has no light effects.

## Overview

ChunIVision supports four output methods for Windows:

1. **Virtual Serial Port (COM)**: Standard Chunithm serial protocol via virtual COM ports
2. **Virtual HID Device (USB)**: Native Chunithm USB device emulation (requires hardware)
3. **Virtual Keyboard**: Maps zones/air sensors to keyboard keys (testing/fallback)
4. **UDP Network**: Custom low-latency protocol for custom games

Each method transmits the same logical state:

- **Touch Zones**: 32 boolean values (numbered 1-32, bottom-right is 1, odd=bottom row, even=top row)
- **Air Sensors**: 6 boolean values (Air 0-5 at heights: 17.9cm, 21.3cm, 24.7cm, 28.1cm, 31.5cm, 34.9cm)

---

## 1. Virtual Serial Port (COM)

### Overview

Standard Chunithm controller communication via virtual COM port on Windows.

**Characteristics:**

- Latency: ~5-8ms
- Baud Rate: 115200
- Data Bits: 8
- Parity: None
- Stop Bits: 1
- Flow Control: None

### Protocol Specification

#### Packet Structure (Input Report)

**Total Size**: 33 bytes

```
Byte   | Description                          | Value
-------|--------------------------------------|------------------
0      | Report ID                            | 0x01
1-4    | Touch zones 1-32 (4 bytes, bit-packed) | Bitfield
5-10   | Air sensors 0-5 (6 bytes)           | 0x00 or 0x01 each
11-32  | Reserved/Padding                     | 0x00
```

#### Touch Zone Bit Packing

Zones are packed into 4 bytes (32 bits):

```
Byte 1 (zones 1-8):   bit 0=zone1, bit 1=zone2, ..., bit 7=zone8
Byte 2 (zones 9-16):  bit 0=zone9, bit 1=zone10, ..., bit 7=zone16
Byte 3 (zones 17-24): bit 0=zone17, bit 1=zone18, ..., bit 7=zone24
Byte 4 (zones 25-32): bit 0=zone25, bit 1=zone26, ..., bit 7=zone32
```

Each bit: `1` = touched, `0` = not touched

#### Air Sensor Format

6 bytes for air sensors (not bit-packed for easier processing):

```
Byte 5:  Air 0 (0x00=inactive, 0x01=active)
Byte 6:  Air 1
Byte 7:  Air 2
Byte 8:  Air 3
Byte 9:  Air 4
Byte 10: Air 5
```

**Note**: Bytes 11-32 are reserved for future use (e.g., pressure data) but currently unused.

#### Example Packet

Zone 1 and Zone 2 touched, Air sensor 2 active:

```hex
01                    // Report ID
03 00 00 00          // Zones: 0x03 = 0b00000011 (zone 1 and 2)
00 00 01 00 00 00    // Airs: Air 2 active
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  // Reserved
```

### Windows COM Port Setup

Using `com0com` (Null-modem emulator) to create virtual port pair:

1. Install com0com from SourceForge
2. Create port pair: COM10 <-> COM11
3. ChunIVision writes to COM10
4. Game reads from COM11

**Alternative**: Use Windows built-in virtual serial port or `hub4com`

### Implementation Notes

```python
import serial

# ChunIVision side (writer)
port = serial.Serial('COM10', 115200, timeout=0.01)

def send_state(touch_zones, air_sensors):
    # Pack touch zones into 4 bytes
    touch_bytes = bytearray(4)
    for i in range(32):
        if touch_zones[i]:
            byte_idx = i // 8
            bit_idx = i % 8
            touch_bytes[byte_idx] |= (1 << bit_idx)
    
    # Pack air sensors (6 bytes)
    air_bytes = bytearray([1 if air_sensors[i] else 0 for i in range(6)])
    
    # Build packet
    packet = bytearray([0x01]) + touch_bytes + air_bytes + bytearray(22)
    
    port.write(packet)
```

---

## 2. Virtual HID Device (USB Emulation)

### Overview

Emulates real Chunithm controller as USB HID device using **Windows USB Device Emulation (UDE)** framework.

**Characteristics:**

- Latency: ~2-4ms (kernel-mode UDE)
- Pure software solution (no hardware required)
- Official Chunithm USB parameters
- Requires custom WDK kernel driver

**USB Device Parameters** (from reference hardware):

- **Vendor ID**: 0x1973
- **Product ID**: 0x2001
- **Manufacturer**: "ZHOUSENSOR I/O SYSTEM"
- **Product**: "ZhouSensor YubiDeck"
- **Serial**: "OK"

### HID Descriptor

Raw HID descriptor based on actual Chunithm controller:

```c
#define RAWHID_USAGE_PAGE 0xFFC0
#define RAWHID_USAGE      0x0C00

uint8_t const desc_hid_report[] = {
    0x06, lowByte(RAWHID_USAGE_PAGE), highByte(RAWHID_USAGE_PAGE),
    0x0A, lowByte(RAWHID_USAGE), highByte(RAWHID_USAGE),
    
    0xA1, 0x01,       // Collection 0x01
    0x75, 0x08,       // report size = 8 bits
    0x15, 0x00,       // logical minimum = 0
    0x26, 0xFF, 0x00, // logical maximum = 255
    
    0x95, 45,         // report count TX (input to PC)
    0x09, 0x01,       // usage
    0x81, 0x02,       // Input (array)
    
    0x95, 61,         // report count RX (output from PC, for RGB - not used)
    0x09, 0x02,       // usage
    0x91, 0x02,       // Output (array)
    0xC0              // end collection
};
```

### HID Input Report Format

**Report Size**: 45 bytes (sent to PC)

```c
struct inputdata {
  uint8_t IRValue;         // Byte 0: IR sensors (6 air sensors in low 6 bits)
  uint8_t Buttons;         // Byte 1: Function buttons (low 3 bits)
  uint8_t TouchValue[32];  // Byte 2-33: Touch zone values (0-255 per zone)
  uint8_t CardStatus;      // Byte 34: Card reader status (0=none, 1=aime, 2=felica)
  uint8_t CardID[10];      // Byte 35-44: Card ID data
};
```

**Field Details:**

- **IRValue** (byte 0): Air sensor states
  - Bit 0: Air sensor 0 (17.9cm)
  - Bit 1: Air sensor 1 (21.3cm)
  - Bit 2: Air sensor 2 (24.7cm)
  - Bit 3: Air sensor 3 (28.1cm)
  - Bit 4: Air sensor 4 (31.5cm)
  - Bit 5: Air sensor 5 (34.9cm)
  - Bits 6-7: Reserved

- **Buttons** (byte 1): Function buttons (not used in ChunIVision)
  - Bit 0-2: Test, Service, Coin buttons
  - Bits 3-7: Reserved

- **TouchValue[32]** (bytes 2-33): Touch zone pressure values
  - 0x00 = Not touched
  - 0x01-0xFF = Touch pressure (ChunIVision uses 0x00 or 0x64 for binary)
  - Order: Zone 1-32 in array index 0-31

- **CardStatus** (byte 34): Always 0 (no card reader)
- **CardID[10]** (bytes 35-44): Always 0 (no card reader)

### Windows Implementation using UDE

**USB Device Emulation (UDE)** is a Windows kernel-mode framework that allows creating virtual USB devices entirely in software.

#### Architecture

```text
ChunIVision (User Mode)
       ↓
UDE Client Driver (Kernel Mode)
       ↓
USB Device Emulation (UDE) Framework
       ↓
Windows USB Stack
       ↓
Game Application
```

#### UDE Driver Implementation

Create a kernel-mode driver using Windows Driver Kit (WDK):

**1. Driver Entry Point** (`ChunithmUDE.c`):

```c
#include <ntddk.h>
#include <wdf.h>
#include <usbdi.h>
#include <udevcx.h>

// Device context
typedef struct _DEVICE_CONTEXT {
    UDECXUSBDEVICE UdecxUsbDevice;
    WDFQUEUE ControlQueue;
    WDFQUEUE InterruptInQueue;
} DEVICE_CONTEXT, *PDEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT, GetDeviceContext)

// USB Device Descriptor
USB_DEVICE_DESCRIPTOR g_UsbDeviceDescriptor = {
    sizeof(USB_DEVICE_DESCRIPTOR),      // bLength
    USB_DEVICE_DESCRIPTOR_TYPE,         // bDescriptorType
    0x0200,                             // bcdUSB (USB 2.0)
    0x00,                               // bDeviceClass
    0x00,                               // bDeviceSubClass
    0x00,                               // bDeviceProtocol
    0x40,                               // bMaxPacketSize0 (64 bytes)
    0x1973,                             // idVendor
    0x2001,                             // idProduct
    0x0100,                             // bcdDevice
    1,                                  // iManufacturer
    2,                                  // iProduct
    3,                                  // iSerialNumber
    1                                   // bNumConfigurations
};

NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PUNICODE_STRING RegistryPath
) {
    WDF_DRIVER_CONFIG config;
    NTSTATUS status;

    WDF_DRIVER_CONFIG_INIT(&config, DeviceAdd);
    
    status = WdfDriverCreate(
        DriverObject,
        RegistryPath,
        WDF_NO_OBJECT_ATTRIBUTES,
        &config,
        WDF_NO_HANDLE
    );

    return status;
}
```

**2. Device Initialization**:

```c
NTSTATUS DeviceAdd(
    _In_ WDFDRIVER       Driver,
    _In_ PWDFDEVICE_INIT DeviceInit
) {
    NTSTATUS status;
    WDFDEVICE device;
    PDEVICE_CONTEXT deviceContext;
    UDECX_USB_DEVICE_STATE_CHANGE_CALLBACKS callbacks;
    
    // Configure UDE
    status = UdecxInitializeWdfDeviceInit(DeviceInit);
    if (!NT_SUCCESS(status)) return status;
    
    // Create device
    WDF_OBJECT_ATTRIBUTES deviceAttributes;
    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&deviceAttributes, DEVICE_CONTEXT);
    
    status = WdfDeviceCreate(&DeviceInit, &deviceAttributes, &device);
    if (!NT_SUCCESS(status)) return status;
    
    deviceContext = GetDeviceContext(device);
    
    // Create USB device
    UDECX_USB_DEVICE_CALLBACKS udecxCallbacks;
    UDECX_USB_DEVICE_CALLBACKS_INIT(&udecxCallbacks);
    
    UDECXUSBDEVICE_INIT* usbDeviceInit;
    usbDeviceInit = UdecxUsbDeviceInitAllocate(device);
    
    // Set descriptors
    UdecxUsbDeviceInitSetDeviceDescriptor(usbDeviceInit, &g_UsbDeviceDescriptor);
    
    // Create UDE USB device
    status = UdecxUsbDeviceCreate(&usbDeviceInit, 
                                   WDF_NO_OBJECT_ATTRIBUTES,
                                   &deviceContext->UdecxUsbDevice);
    
    return status;
}
```

**3. HID Report Handling**:

```c
NTSTATUS SendHIDReport(
    _In_ PDEVICE_CONTEXT DeviceContext,
    _In_ PUCHAR ReportData,
    _In_ ULONG ReportLength
) {
    WDFREQUEST request;
    NTSTATUS status;
    
    // Get pending interrupt IN request
    status = WdfIoQueueRetrieveNextRequest(
        DeviceContext->InterruptInQueue,
        &request
    );
    
    if (!NT_SUCCESS(status)) {
        return status;
    }
    
    // Copy report data to request
    PVOID buffer;
    size_t bufferLength;
    status = WdfRequestRetrieveOutputBuffer(
        request,
        ReportLength,
        &buffer,
        &bufferLength
    );
    
    if (NT_SUCCESS(status)) {
        RtlCopyMemory(buffer, ReportData, ReportLength);
        WdfRequestCompleteWithInformation(request, STATUS_SUCCESS, ReportLength);
    }
    
    return status;
}
```

#### User-Mode Communication

Create a user-mode application to communicate with the UDE driver:

```python
import ctypes
from ctypes import wintypes
import struct

class ChunithmUDEController:
    def __init__(self, device_path=r"\\.\ChunithmController"):
        """Initialize connection to UDE driver."""
        self.device_path = device_path
        self.handle = None
        
        # IOCTL codes for driver communication
        self.IOCTL_SEND_REPORT = self._CTL_CODE(0x8000, 0x800, 0, 3)
        
        self._open_device()
    
    def _CTL_CODE(self, DeviceType, Function, Method, Access):
        """Calculate IOCTL control code."""
        return (DeviceType << 16) | (Access << 14) | (Function << 2) | Method
    
    def _open_device(self):
        """Open handle to UDE driver."""
        kernel32 = ctypes.windll.kernel32
        
        self.handle = kernel32.CreateFileW(
            self.device_path,
            0xC0000000,  # GENERIC_READ | GENERIC_WRITE
            0,           # No sharing
            None,
            3,           # OPEN_EXISTING
            0,
            None
        )
        
        if self.handle == -1:
            raise RuntimeError(f"Failed to open device: {self.device_path}")
    
    def send_state(self, touch_zones, air_sensors):
        """Send controller state to UDE driver."""
        # Build IR value (6 air sensors in low 6 bits)
        ir_value = 0
        for i in range(6):
            if air_sensors[i]:
                ir_value |= (1 << i)
        
        # Build touch values (0x00 = not touched, 0x64 = touched)
        touch_values = [0x64 if touch_zones[i] else 0x00 for i in range(32)]
        
        # Pack HID report (45 bytes)
        report = struct.pack(
            'BB32B11B',
            ir_value,           # IRValue
            0x00,               # Buttons (not used)
            *touch_values,      # TouchValue[32]
            0,                  # CardStatus (0 = no card)
            *([0] * 10)         # CardID[10] (all zeros)
        )
        
        # Send to driver via IOCTL
        bytes_returned = wintypes.DWORD()
        kernel32 = ctypes.windll.kernel32
        
        success = kernel32.DeviceIoControl(
            self.handle,
            self.IOCTL_SEND_REPORT,
            report,
            len(report),
            None,
            0,
            ctypes.byref(bytes_returned),
            None
        )
        
        return success
    
    def close(self):
        """Close device handle."""
        if self.handle:
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None

# Usage example
controller = ChunithmUDEController()

# Send state
touch_zones = [False] * 32
air_sensors = [False] * 6

touch_zones[0] = True  # Zone 1 touched
air_sensors[2] = True  # Air 2 active

controller.send_state(touch_zones, air_sensors)
controller.close()
```

#### Building and Installing UDE Driver

**Requirements:**

- Windows Driver Kit (WDK) 10
- Visual Studio 2019 or later
- Windows 10/11 (64-bit)
- Test signing enabled (for development)

**Build Steps:**

1. Create WDK driver project in Visual Studio
2. Add UDE library dependencies in project settings
3. Build driver (`.sys` file)
4. Create INF file for driver installation
5. Sign driver (test signing for development)
6. Install using Device Manager or `pnputil`

**Enable Test Signing** (for development):

```cmd
bcdedit /set testsigning on
```

**Install Driver:**

```cmd
pnputil /add-driver ChunithmUDE.inf /install
```

#### Alternative: User-Mode USB Emulation

For easier development without kernel drivers, use **USB/IP** with user-mode stub:

```python
# This requires usbip-win installation
import usb.core
import usb.backend.libusb1

# Create virtual USB device using python-usbip
# (Simplified - actual implementation more complex)
```

**Note**: UDE kernel driver provides best performance (2-4ms latency) but requires driver development skills. For prototyping, consider hardware-based USB emulation (Raspberry Pi Pico, Arduino) as an alternative.

### Configuration

```yaml
hid:
  enabled: false  # Requires UDE driver installation
  
  # USB device parameters (matching real Chunithm controller)
  vendor_id: 0x1973
  product_id: 0x2001
  manufacturer: "ZHOUSENSOR I/O SYSTEM"
  product: "ZhouSensor YubiDeck"
  serial: "OK"
  
  # UDE driver settings
  driver_path: "\\\\.\\ChunithmController"  # Device path for IOCTL communication
  use_ude: true  # Use Windows UDE (kernel driver)
  
  # Alternative: Hardware USB emulation
  # use_ude: false
  # hardware_type: "pico"  # Options: "pico", "arduino", "teensy"
  # serial_port: "COM5"    # Serial port for hardware communication
```

### Game Integration Example

OS sees device as standard game controller. Use any HID library:

```python
import hid

# Find ChunIVision device
device = hid.device()
device.open(0x1234, 0x5678)

while True:
    # Read HID report (5 bytes)
    report = device.read(5)
    if report[0] != 0x01:
        continue
    
    # Parse bit-packed state
    touch_zones = []
    for byte_idx in range(4):
        for bit_idx in range(8):
            touch_zones.append((report[byte_idx] >> bit_idx) & 1)
    
    air_sensors = [(report[4] >> i) & 1 for i in range(6)]
    
    # Process state...
```

---

## 3. Virtual Keyboard

### Overview

Maps touch zones and air sensors to keyboard keys for games without native controller support.

**Characteristics:**

- Latency: ~10-15ms
- Compatible with any game accepting keyboard input
- Easy setup, no configuration needed
- Limited by keyboard rollover (typically 6-10 keys)

### Key Mapping

Default key assignments (customizable in config):

**Touch Zones (32 keys):**

- **Bottom Row (zones 31, 29, 27, ..., 3, 1):**
  - `1 2 3 4 5 6 7 8 9 0 - = [ ] \ ;` (16 keys)

- **Top Row (zones 32, 30, 28, ..., 4, 2):**
  - `Q W E R T Y U I O P { } : " | ?` (16 keys)

**Air Sensors (6 keys):**

- Air 0 (17.9cm): `A`
- Air 1 (21.3cm): `S`
- Air 2 (24.7cm): `D`
- Air 3 (28.1cm): `F`
- Air 4 (31.5cm): `G`
- Air 5 (34.9cm): `H`

### Windows Implementation

Using `pynput` library for cross-platform keyboard emulation:

```python
from pynput.keyboard import Controller, Key

class KeyboardAdapter:
    def __init__(self):
        self.keyboard = Controller()
        
        # Zone key mapping (zone 1-32 -> keys)
        self.zone_keys = [
            '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '0', '-', '=', '[', ']', '\\', ';',
            'q', 'w', 'e', 'r', 't', 'y', 'u', 'i',
            'o', 'p', '{', '}', ':', '"', '|', '?'
        ]
        
        # Air sensor keys
        self.air_keys = ['a', 's', 'd', 'f', 'g', 'h']
        
        # Current state tracking
        self.zone_state = [False] * 32
        self.air_state = [False] * 6
    
    def send_state(self, touch_zones, air_sensors):
        # Handle touch zones
        for i in range(32):
            if touch_zones[i] and not self.zone_state[i]:
                # Press key
                self.keyboard.press(self.zone_keys[i])
                self.zone_state[i] = True
            elif not touch_zones[i] and self.zone_state[i]:
                # Release key
                self.keyboard.release(self.zone_keys[i])
                self.zone_state[i] = False
        
        # Handle air sensors
        for i in range(6):
            if air_sensors[i] and not self.air_state[i]:
                self.keyboard.press(self.air_keys[i])
                self.air_state[i] = True
            elif not air_sensors[i] and self.air_state[i]:
                self.keyboard.release(self.air_keys[i])
                self.air_state[i] = False
```

### Configuration

```yaml
keyboard:
  enabled: true
  
  # Touch zone keys (32 keys, mapped to zones 1-32)
  touch_zone_keys:
    - "1"  # Zone 1 (bottom-right)
    - "2"  # Zone 2 (top-right)
    - "3"  # Zone 3
    # ... (32 total)
  
  # Air sensor keys (6 keys)
  air_sensor_keys:
    - "a"  # Air 0 (17.9cm)
    - "s"  # Air 1 (21.3cm)
    - "d"  # Air 2 (24.7cm)
    - "f"  # Air 3 (28.1cm)
    - "g"  # Air 4 (31.5cm)
    - "h"  # Air 5 (34.9cm)
  
  # Optional key modifiers (e.g., for special combinations)
  use_modifiers: false
  modifier_keys: []  # e.g., ["shift", "ctrl"]
```

### Limitations

1. **Key Rollover**: Most keyboards support 6-10 simultaneous keys (NKRO keyboards support more)
2. **Interference**: May conflict with other keyboard input
3. **Latency**: Slightly higher than direct protocols due to OS keyboard pipeline
4. **No Pressure**: Binary on/off only, no pressure sensitivity

### Game Integration

Game receives standard keyboard input. Simply map keys in game settings:

1. Open game key configuration
2. Assign touch zones to keys `1-9`, `0`, `-`, `=`, etc.
3. Assign air sensors to `A`, `S`, `D`, `F`, `G`, `H`
4. Save configuration and test

**Recommendation**: Use keyboard adapter only for testing or games without native controller support. For production use, prefer UDP or HID.

---

## 4. UDP Network Protocol

### Overview

Custom UDP-based protocol for low-latency local or network communication.

**Characteristics:**

- Latency: ~3-5ms (localhost)
- No connection overhead (UDP)
- Simple packet format
- Supports binary and JSON modes

### Binary Protocol (Recommended)

**Packet Structure:**

```text
Offset | Size | Type     | Description
-------|------|----------|----------------------------------
0x00   | 4    | char[4]  | Magic number "CHUN"
0x04   | 2    | uint16   | Protocol version (0x0001)
0x06   | 2    | uint16   | Packet sequence number
0x08   | 8    | uint64   | Timestamp (microseconds since epoch)
0x10   | 4    | byte[4]  | Touch zones 1-32 (bit-packed)
0x14   | 1    | byte     | Air sensors 0-5 (bit 0-5)
0x15   | 1    | byte     | Reserved flags
0x16   | 2    | uint16   | CRC-16 checksum
```

**Total Size**: 24 bytes per packet

#### Bit Packing Details

**Touch Zones** (4 bytes):

```text
Byte 0: zones 1-8   (LSB = zone 1, MSB = zone 8)
Byte 1: zones 9-16
Byte 2: zones 17-24
Byte 3: zones 25-32
```

**Air Sensors** (1 byte):

```text
Bit 0: Air 0
Bit 1: Air 1
Bit 2: Air 2
Bit 3: Air 3
Bit 4: Air 4
Bit 5: Air 5
Bit 6-7: Unused (set to 0)
```

#### Example Binary Packet

```hex
43 48 55 4E              // Magic "CHUN"
00 01                    // Version 1
12 34                    // Sequence 0x1234
00 00 01 86 A0 00 00 00  // Timestamp
03 00 00 00              // Touch: zones 1,2 active
04                       // Air: sensor 2 active
00                       // Reserved
A5 3C                    // CRC-16
```

### JSON Protocol (Debug Mode)

For easier debugging and development:

```json
{
  "magic": "CHUN",
  "version": 1,
  "sequence": 4660,
  "timestamp": 1704931200000000,
  "touch": {
    "1": true,
    "2": true,
    "3": false,
    ...
    "32": false
  },
  "air": {
    "0": false,
    "1": false,
    "2": true,
    "3": false,
    "4": false,
    "5": false
  }
}
```

**Size**: ~400-500 bytes (varies with formatting)

### Network Configuration

```yaml
udp:
  mode: "binary"           # or "json"
  target_ip: "127.0.0.1"   # Localhost for local game
  target_port: 28888       # Default port (28888 = "CHUN" in phone keypad)
  bind_port: 0             # Source port (0 = any)
  enable_multicast: false  # For multiple receivers
  multicast_group: "239.255.28.88"  # If multicast enabled
```

### Python Sender Implementation (ChunIVision)

```python
import socket
import struct
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = ('127.0.0.1', 28888)
sequence = 0

def pack_binary_packet(touch_zones, air_sensors):
    global sequence
    
    # Pack touch zones into 4 bytes
    touch_bytes = 0
    for i in range(32):
        if touch_zones[i]:
            touch_bytes |= (1 << i)
    
    # Pack air sensors into 1 byte
    air_byte = 0
    for i in range(6):
        if air_sensors[i]:
            air_byte |= (1 << i)
    
    # Build packet
    packet = struct.pack(
        '<4sHHQ4sBB',
        b'CHUN',              # Magic
        1,                    # Version
        sequence & 0xFFFF,    # Sequence
        int(time.time() * 1e6),  # Timestamp (microseconds)
        touch_bytes.to_bytes(4, 'little'),  # Touch zones
        air_byte,             # Air sensors
        0                     # Reserved
    )
    
    # Calculate CRC-16
    crc = calculate_crc16(packet)
    packet += struct.pack('<H', crc)
    
    sequence += 1
    return packet

def send_state(touch_zones, air_sensors):
    packet = pack_binary_packet(touch_zones, air_sensors)
    sock.sendto(packet, target)
```

### Python Receiver Implementation (Game)

```python
import socket
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 28888))

while True:
    data, addr = sock.recvfrom(1024)
    
    # Verify magic number
    if data[0:4] != b'CHUN':
        continue
    
    # Unpack packet
    magic, version, seq, timestamp = struct.unpack('<4sHHQ', data[0:16])
    touch_bytes = struct.unpack('<I', data[16:20])[0]
    air_byte = data[20]
    crc = struct.unpack('<H', data[22:24])[0]
    
    # Verify CRC
    if not verify_crc16(data[0:22], crc):
        continue
    
    # Extract touch zones
    touch_zones = [(touch_bytes >> i) & 1 for i in range(32)]
    
    # Extract air sensors
    air_sensors = [(air_byte >> i) & 1 for i in range(6)]
    
    # Process state...
```

---

## Protocol Comparison

| Protocol | Latency | Packet Size | Ease of Use | Compatibility | Best For |
|----------|---------|-------------|-------------|---------------|----------|
| HID (USB) | 2-4ms | 45 bytes | Hard (hardware) | Excellent | Official compatibility |
| UDP (Binary) | 3-5ms | 24 bytes | Medium | Excellent | Custom games |
| Serial (COM) | 5-8ms | 33 bytes | Easy | Good | Legacy support |
| UDP (JSON) | 5-8ms | ~450 bytes | Easy (debug) | Excellent | Development |
| Keyboard | 10-15ms | N/A | Easy | Universal | Testing/fallback |

**Recommendation**:

- **Official Games**: HID USB (native Chunithm protocol, requires hardware)
- **Custom Games**: UDP Binary (lowest latency, pure software)
- **Development**: UDP JSON (easy debugging)
- **Legacy Support**: Serial COM (standard emulator compatibility)
- **Testing/Fallback**: Keyboard (universal compatibility)

---

## CRC-16 Implementation

Used in UDP protocol for data integrity:

```python
def calculate_crc16(data):
    """Calculate CRC-16-CCITT (0xFFFF initial value)"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc

def verify_crc16(data, expected_crc):
    """Verify CRC-16 checksum"""
    calculated_crc = calculate_crc16(data)
    return calculated_crc == expected_crc
```

---

## Error Handling

All protocols should implement:

1. **Connection Loss**: Detect and attempt reconnection
2. **Packet Loss**: UDP uses sequence numbers to detect drops
3. **Corruption**: CRC checks for UDP, HID has built-in error detection
4. **Timeout**: Receiver should alert if no data for >100ms

---

## Multi-Output Mode

ChunIVision can output to multiple protocols simultaneously:

```yaml
outputs:
  serial:
    enabled: true
    port: "COM10"
  
  udp:
    enabled: true
    mode: "binary"
    target_port: 28888
  
  hid:
    enabled: false  # Requires USB hardware emulation
  
  keyboard:
    enabled: false  # Enable for testing or fallback
```

All enabled outputs receive identical state updates in parallel.

---

## Performance Optimization

### Minimize Latency

1. Use UDP binary protocol for lowest latency
2. Set thread priority to HIGH for output thread
3. Disable Nagle's algorithm for network sockets
4. Use non-blocking I/O where possible

### Reduce CPU Usage

1. Only send updates on state changes (not every frame)
2. Batch multiple state changes if needed
3. Use bit-packing to reduce packet size

---

## Testing Tools

### UDP Listener (Python)

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 28888))

print("Listening for ChunIVision packets on port 28888...")

while True:
    data, addr = sock.recvfrom(1024)
    print(f"Received {len(data)} bytes from {addr}")
    print(f"  Hex: {data.hex()}")
    if len(data) >= 24 and data[0:4] == b'CHUN':
        touch = int.from_bytes(data[16:20], 'little')
        air = data[20]
        print(f"  Touch zones: {bin(touch)[2:].zfill(32)}")
        print(f"  Air sensors: {bin(air)[2:].zfill(6)}")
```

### COM Port Monitor

Use tools like:

- **RealTerm**: Serial port terminal
- **Termite**: Serial debugger
- **com0com**: Virtual COM port pair creator

---

## Future Extensions

Potential protocol enhancements:

1. **Pressure Data**: Add force/pressure per touch zone (0-255)
2. **Touch Position**: Include X/Y coordinates within each zone
3. **Firmware Updates**: Over-the-air updates via UDP
4. **Diagnostics**: Sensor health, calibration status

These can be added while maintaining backward compatibility through protocol versioning.

---

**Protocol Version**: 1.0  
**Last Updated**: 2024-01-10  
**Author**: Misaka 19465  
**Platform**: Windows Only
