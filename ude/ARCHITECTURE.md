# ChunithmUDE Architecture

Technical architecture documentation for the ChunithmUDE kernel driver.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Mode (Python)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ChunithmUDEDevice (ude/python/ude_device.py)   │   │
│  │  - send_report(touch_zones, air_sensors)        │   │
│  │  - get_status()                                  │   │
│  │  - reset()                                       │   │
│  └───────────────────┬─────────────────────────────┘   │
│                      │ IOCTL (DeviceIoControl)          │
└──────────────────────┼──────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────┐
│                      ▼ Kernel Mode                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ChunithmUDE.sys (Kernel Driver)                │   │
│  │                                                  │   │
│  │  Queue.c:  IOCTL handlers                       │   │
│  │    ├─ IOCTL_CHUNITHM_SEND_REPORT               │   │
│  │    ├─ IOCTL_CHUNITHM_GET_STATUS                │   │
│  │    └─ IOCTL_CHUNITHM_RESET                      │   │
│  │                                                  │   │
│  │  Usb.c:  UDE device creation                    │   │
│  │    ├─ USB descriptors (VID:1973, PID:2001)     │   │
│  │    ├─ HID report descriptor                     │   │
│  │    └─ Endpoint creation (EP1 IN, EP2 OUT)      │   │
│  │                                                  │   │
│  │  UsbCallbacks.c:  USB control requests          │   │
│  │    ├─ String descriptors                        │   │
│  │    └─ Control URB handling                      │   │
│  │                                                  │   │
│  │  EndpointQueue.c:  Interrupt IN management      │   │
│  │    └─ Complete HID input reports to host        │   │
│  └─────────────────────────────────────────────────┘   │
│                      │                                   │
│                      ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  UDE Framework (Windows 10+)                    │   │
│  │  - UdecxUsbDeviceCreate()                       │   │
│  │  - UdecxUsbDevicePlugIn()                       │   │
│  │  - Emulates USB bus traffic                     │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────┐
│                      ▼ USB Stack                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  HID Class Driver (hidclass.sys)                │   │
│  │  - Exposes HID device interface                 │   │
│  │  - Game recognizes as Chunithm controller       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Driver Components

### ChunithmUDE.c

Main driver entry point and device initialization.

**Functions**:

- `DriverEntry()` - Initializes WDF driver object and registers callbacks
- `EvtDeviceAdd()` - Creates WDF device object for each device instance
- `ChunithmUDE_CreateDevice()` - Initializes device context, queues, and spinlocks
- `ChunithmUDE_InitializeReport()` - Sets default HID report values
- `ChunithmUDE_ValidateReport()` - Validates report data before transmission

### Device.c

PnP and power management event handlers.

**Callbacks**:

- `EvtDevicePrepareHardware()` - Called when device starts, initializes hardware state
- `EvtDeviceReleaseHardware()` - Called when device stops, releases resources
- `EvtDeviceD0Entry()` - Power up transition, plugs in UDE device
- `EvtDeviceD0Exit()` - Power down transition, unplugs UDE device

### Queue.c

I/O queue management and IOCTL request handling.

**IOCTL Codes**:

- `IOCTL_CHUNITHM_SEND_REPORT` (0x222800) - Receives 45-byte HID report from Python
- `IOCTL_CHUNITHM_GET_STATUS` (0x222804) - Returns driver statistics
- `IOCTL_CHUNITHM_RESET` (0x222808) - Resets device state

**Functions**:

- `Queue_Initialize()` - Creates default I/O queue for IOCTL requests
- `EvtIoDeviceControl()` - Dispatches IOCTL requests to appropriate handlers
- `ChunithmUDE_SendReport()` - Validates and stores report, completes pending USB requests

### Usb.c

USB device and endpoint creation using UDE framework.

**Functions**:

- `Usb_CreateUsbDevice()` - Creates UDE USB device with proper descriptors
- `Usb_CreateEndpoints()` - Creates interrupt IN/OUT endpoints
- `Usb_GetDescriptors()` - Builds USB device and configuration descriptors

**USB Configuration**:

- Vendor ID: 0x1973 (ZHOUSENSOR)
- Product ID: 0x2001 (YubiDeck)
- Class: HID (0x03)
- Endpoints:
  - EP1 IN (0x81): Interrupt, 45 bytes, 1ms interval (input reports)
  - EP2 OUT (0x02): Interrupt, 61 bytes, 1ms interval (RGB output)

### UsbCallbacks.c

USB control request handling and string descriptors.

**Functions**:

- `EvtUsbDeviceDefaultEndpointAdd()` - Creates control endpoint (EP0)
- `EvtControlUrb()` - Handles USB control URBs (GET_DESCRIPTOR, etc.)
- `UsbGetStringDescriptor()` - Returns string descriptors

**String Descriptors**:

- Index 0: Language ID (0x0409 - English US)
- Index 1: Manufacturer ("ZHOUSENSOR I/O SYSTEM")
- Index 2: Product ("ZhouSensor YubiDeck")
- Index 3: Serial Number ("OK")

### EndpointQueue.c

Interrupt IN endpoint queue management for sending HID reports to host.

**Data Flow**:

1. Host (game) sends interrupt IN request to EP1
2. Request queued in `InterruptInQueue`
3. Python calls `send_report()` → driver completes queued request
4. Host receives 45-byte HID input report

**Functions**:

- `EndpointQueue_Initialize()` - Creates manual queue for interrupt endpoint
- `ChunithmUDE_CompleteInterruptInRequest()` - Completes pending USB read with report data
- `EvtEndpointQueueReadyNotification()` - Called when host is ready for more data

### Descriptors.c

Complete USB and HID descriptor generation.

**Functions**:

- `Descriptors_GetCompleteConfigDescriptor()` - Full configuration with interface/HID/endpoints
- `Descriptors_GetHidDescriptor()` - HID descriptor (bcdHID 1.11, report lengths)
- `Descriptors_GetHidReportDescriptor()` - HID report descriptor (vendor-defined)

**HID Report Descriptor**:

```
Usage Page (Vendor 0xFFC0)
Usage (0x0C00)
Collection (Application)
  Report Size (8 bits)
  Logical Minimum (0)
  Logical Maximum (255)
  
  # Input Report (45 bytes)
  Report Count (45)
  Usage (1)
  Input (Data, Variable, Absolute)
  
  # Output Report (61 bytes)
  Report Count (61)
  Usage (2)
  Output (Data, Variable, Absolute)
End Collection
```

## Data Structures

### Device Context

```c
typedef struct _DEVICE_CONTEXT {
  WDFDEVICE Device;
  UDECXUSBDEVICE UdecxUsbDevice;
  UDECXUSBENDPOINT InterruptInEndpoint;
  WDFQUEUE DefaultQueue;
  WDFQUEUE InterruptInQueue;
  WDFSPINLOCK ReportLock;
  UCHAR CurrentReport[45];
  BOOLEAN DeviceReady;
  BOOLEAN PendingReportAvailable;
  ULONG ReportsSent;
  ULONG ErrorCount;
} DEVICE_CONTEXT;
```

### HID Input Report

```c
#pragma pack(push, 1)
typedef struct _CHUNITHM_INPUT_REPORT {
  UCHAR IRValue;         // Byte 0: Air sensors (6 bits)
  UCHAR Buttons;         // Byte 1: Function buttons (unused)
  UCHAR TouchValue[32];  // Byte 2-33: Touch zones (0x00 or 0x64)
  UCHAR CardStatus;      // Byte 34: Card reader status
  UCHAR CardID[10];      // Byte 35-44: Card ID
} CHUNITHM_INPUT_REPORT;
#pragma pack(pop)
```

## Communication Flow

### Python to Kernel (Send Report)

```
1. Python: device.send_report(touch_zones, air_sensors)
2. Python: Pack 45-byte report structure
3. Python: DeviceIoControl(IOCTL_CHUNITHM_SEND_REPORT, buffer)
4. Kernel: EvtIoDeviceControl() receives IOCTL
5. Kernel: ChunithmUDE_SendReport() validates report
6. Kernel: Store report in device context (with spinlock)
7. Kernel: ChunithmUDE_CompleteInterruptInRequest()
8. Kernel: Complete pending USB interrupt IN request
9. USB Stack: Host receives HID input report
10. Game: Reads controller state
```

### USB Host to Driver (Read Input)

```
1. Host: Sends interrupt IN request to EP1
2. UDE Framework: Forwards request to driver
3. Driver: Queues request in InterruptInQueue (manual queue)
4. Driver: Waits for report data from Python
5. Python: Calls send_report()
6. Driver: Retrieves queued request
7. Driver: Copies report data to request buffer
8. Driver: Completes request with STATUS_SUCCESS
9. Host: Receives 45-byte report
10. HID Class Driver: Parses report and updates state
```

## Synchronization

### Report Buffer Protection

The `CurrentReport` buffer is protected by a spinlock (`ReportLock`) to prevent race conditions between:

- IOCTL handler updating the report
- Interrupt IN completion reading the report

```c
WdfSpinLockAcquire(deviceContext->ReportLock, &lockHandle);
RtlCopyMemory(deviceContext->CurrentReport, report, 45);
WdfSpinLockRelease(deviceContext->ReportLock, lockHandle);
```

### Queue Synchronization

- `DefaultQueue`: Sequential dispatch for IOCTLs (WdfIoQueueDispatchSequential)
- `InterruptInQueue`: Manual dispatch for USB requests (WdfIoQueueDispatchManual)

Manual queue allows the driver to control when USB requests are completed based on report availability.

## Memory Management

All allocations use proper pool tags for tracking:

- Pool tag: `'edUC'` (ChunithmUDE)
- Allocation type: `NonPagedPoolNx` for DMA-safe memory

Memory is freed in proper cleanup callbacks:

- `EvtDriverContextCleanup()` - Driver unload
- `EvtDeviceReleaseHardware()` - Device removal

## Error Handling

All functions return `NTSTATUS` codes:

- `STATUS_SUCCESS` - Operation successful
- `STATUS_INVALID_PARAMETER` - Invalid input
- `STATUS_BUFFER_TOO_SMALL` - Output buffer too small
- `STATUS_INSUFFICIENT_RESOURCES` - Allocation failed
- `STATUS_DEVICE_NOT_READY` - Device not in D0 state

Python bindings translate NTSTATUS to appropriate exceptions.

## Build Configuration

### Debug Build

- Debugging symbols enabled
- Assertions active
- KdPrint() messages compiled in

### Release Build

- Optimizations enabled
- Debugging symbols in PDB
- KdPrint() messages removed

### Driver Verifier

The driver is compatible with Driver Verifier for testing:

```cmd
verifier /standard /driver ChunithmUDE.sys
```

## Performance Characteristics

- **Latency**: 1-2ms from Python IOCTL to USB stack completion
- **Throughput**: Up to 1000 reports/second (limited by 1ms endpoint interval)
- **CPU Overhead**: Minimal (<0.1% at 60 FPS)
- **Memory Footprint**: ~100KB kernel pool per device instance
- **Interrupt Level**: PASSIVE_LEVEL for most operations, DISPATCH_LEVEL for spinlock

## Security Model

- IOCTLs use `METHOD_BUFFERED` for safe data transfer
- All input validated before use
- Report size checked against maximum
- Touch zone values clamped to valid range (0x00 or 0x64)
- Air sensor values validated (0-255 range, 6 sensors)

## Limitations

- Single device instance per driver load
- Output reports (RGB LEDs) not implemented
- Windows 10/11 only (UDE framework requirement)
- Requires test signing or valid code signing certificate
- 64-bit only (no 32-bit support)

## Future Considerations

Potential enhancements beyond current scope:

- Multiple virtual controller instances
- Output report handling for RGB LED control
- Dynamic VID/PID configuration
- ETW event tracing for diagnostics
- WHQL certification for production deployment
