/*++

Module Name:
    ChunithmUDE.h

Abstract:
    Header file for ChunithmUDE kernel driver.
    Emulates Chunithm controller using Windows UDE framework.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#ifndef _CHUNITHM_UDE_H_
#define _CHUNITHM_UDE_H_

#include <initguid.h>
#include <ntddk.h>
#include <udevcx.h>
#include <usb.h>
#include <usbdi.h>
#include <usbdlib.h>
#include <wdf.h>

//
// Device Interface GUID
// {8F5E9A1C-7D4B-4E3F-9C2A-1B8E7F6D5C4A}
//
DEFINE_GUID(GUID_DEVINTERFACE_CHUNITHM_UDE, 0x8f5e9a1c, 0x7d4b, 0x4e3f, 0x9c,
            0x2a, 0x1b, 0x8e, 0x7f, 0x6d, 0x5c, 0x4a);

//
// Pool tags for memory allocation
//
#define CHUNITHM_UDE_POOL_TAG 'edUC'  // 'CUde'

//
// USB device parameters (official Chunithm controller)
//
#define CHUNITHM_VENDOR_ID 0x1973
#define CHUNITHM_PRODUCT_ID 0x2001
#define CHUNITHM_DEVICE_VERSION 0x0100

#define CHUNITHM_MANUFACTURER L"ZHOUSENSOR I/O SYSTEM"
#define CHUNITHM_PRODUCT L"ZhouSensor YubiDeck"
#define CHUNITHM_SERIAL L"OK"

//
// HID report sizes
//
#define CHUNITHM_INPUT_REPORT_SIZE 45   // Input to PC
#define CHUNITHM_OUTPUT_REPORT_SIZE 61  // Output from PC (RGB - not used)

//
// IOCTL codes for user-mode communication
//
#define IOCTL_CHUNITHM_SEND_REPORT \
  CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define IOCTL_CHUNITHM_GET_STATUS \
  CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801, METHOD_BUFFERED, FILE_READ_DATA)

#define IOCTL_CHUNITHM_RESET \
  CTL_CODE(FILE_DEVICE_UNKNOWN, 0x802, METHOD_NEITHER, FILE_ANY_ACCESS)

//
// Device context structure
//
typedef struct _DEVICE_CONTEXT {
  WDFDEVICE Device;
  UDECXUSBDEVICE UdecxUsbDevice;
  UDECXUSBENDPOINT InterruptInEndpoint;
  WDFQUEUE DefaultQueue;
  WDFQUEUE InterruptInQueue;
  WDFSPINLOCK ReportLock;
  UCHAR CurrentReport[CHUNITHM_INPUT_REPORT_SIZE];
  BOOLEAN DeviceReady;
  BOOLEAN PendingReportAvailable;
  ULONG ReportsSent;
  ULONG ErrorCount;
} DEVICE_CONTEXT, *PDEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT, GetDeviceContext)

//
// Endpoint context structure
//
typedef struct _ENDPOINT_QUEUE_CONTEXT {
  UDECXUSBENDPOINT UdecxUsbEndpoint;
} ENDPOINT_QUEUE_CONTEXT, *PENDPOINT_QUEUE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(ENDPOINT_QUEUE_CONTEXT,
                                   GetEndpointQueueContext)

//
// HID input report structure (45 bytes)
//
#pragma pack(push, 1)
typedef struct _CHUNITHM_INPUT_REPORT {
  UCHAR IRValue;         // Byte 0: IR sensors (6 air sensors in low 6 bits)
  UCHAR Buttons;         // Byte 1: Function buttons (not used)
  UCHAR TouchValue[32];  // Byte 2-33: Touch zone values (0x00 or 0x64)
  UCHAR CardStatus;      // Byte 34: Card reader status (always 0)
  UCHAR CardID[10];      // Byte 35-44: Card ID (always 0)
} CHUNITHM_INPUT_REPORT, *PCHUNITHM_INPUT_REPORT;
#pragma pack(pop)

C_ASSERT(sizeof(CHUNITHM_INPUT_REPORT) == CHUNITHM_INPUT_REPORT_SIZE);

//
// Driver status structure for IOCTL_CHUNITHM_GET_STATUS
//
typedef struct _CHUNITHM_DRIVER_STATUS {
  BOOLEAN DeviceReady;
  ULONG ReportsSent;
  ULONG ErrorCount;
} CHUNITHM_DRIVER_STATUS, *PCHUNITHM_DRIVER_STATUS;

//
// Function declarations
//

// Driver entry and unload
DRIVER_INITIALIZE DriverEntry;
EVT_WDF_DRIVER_DEVICE_ADD EvtDeviceAdd;
EVT_WDF_OBJECT_CONTEXT_CLEANUP EvtDriverContextCleanup;

// Device initialization and callbacks
NTSTATUS ChunithmUDE_CreateDevice(_In_ PWDFDEVICE_INIT DeviceInit);
EVT_WDF_DEVICE_PREPARE_HARDWARE EvtDevicePrepareHardware;
EVT_WDF_DEVICE_RELEASE_HARDWARE EvtDeviceReleaseHardware;
EVT_WDF_DEVICE_D0_ENTRY EvtDeviceD0Entry;
EVT_WDF_DEVICE_D0_EXIT EvtDeviceD0Exit;

// USB device and endpoint creation
NTSTATUS Usb_CreateUsbDevice(_In_ WDFDEVICE WdfDevice);
NTSTATUS Usb_CreateEndpoints(_In_ PDEVICE_CONTEXT DeviceContext);
NTSTATUS Usb_GetDescriptors(
    _Out_ PUSB_DEVICE_DESCRIPTOR DeviceDescriptor,
    _Out_ PUSB_CONFIGURATION_DESCRIPTOR* ConfigDescriptor,
    _Out_ PULONG ConfigDescriptorSize);

// I/O queue handlers
NTSTATUS Queue_Initialize(_In_ WDFDEVICE Device);
EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL EvtIoDeviceControl;
EVT_WDF_IO_QUEUE_IO_READ EvtIoRead;
EVT_WDF_IO_QUEUE_IO_WRITE EvtIoWrite;

// UDE endpoint callbacks
EVT_UDECX_USB_ENDPOINT_RESET EvtEndpointReset;
EVT_UDECX_USB_ENDPOINT_START EvtEndpointStart;
EVT_UDECX_USB_ENDPOINT_PURGE EvtEndpointPurge;

// UDE device callbacks
EVT_UDECX_USB_DEVICE_DEFAULT_ENDPOINT_ADD EvtUsbDeviceDefaultEndpointAdd;
EVT_UDECX_USB_DEVICE_ENDPOINT_ADD EvtUsbDeviceEndpointAdd;
VOID EvtControlUrb(_In_ WDFREQUEST Request, _In_ WDFMEMORY Memory);

// USB string descriptors
NTSTATUS UsbGetStringDescriptor(_In_ UCHAR Index,
                                _Out_writes_bytes_(*Length) PVOID Buffer,
                                _Inout_ PULONG Length);

// Endpoint queue management
NTSTATUS EndpointQueue_Initialize(_In_ PDEVICE_CONTEXT DeviceContext);
VOID EvtEndpointQueueReadyNotification(_In_ WDFQUEUE Queue,
                                       _In_ WDFCONTEXT Context);
EVT_WDF_IO_QUEUE_IO_STOP EvtEndpointQueueIoStop;
VOID EvtEndpointReadUrb(_In_ WDFREQUEST Request);

// USB descriptors
NTSTATUS Descriptors_GetCompleteConfigDescriptor(_Out_writes_bytes_(*Length)
                                                     PVOID Buffer,
                                                 _Inout_ PULONG Length);
NTSTATUS Descriptors_GetHidDescriptor(_Out_writes_bytes_(*Length) PVOID Buffer,
                                      _Inout_ PULONG Length);
NTSTATUS Descriptors_GetHidReportDescriptor(_Out_writes_bytes_(*Length)
                                                PVOID Buffer,
                                            _Inout_ PULONG Length);

// Report handling
NTSTATUS ChunithmUDE_SendReport(_In_ PDEVICE_CONTEXT DeviceContext,
                                _In_ PCHUNITHM_INPUT_REPORT Report);

NTSTATUS ChunithmUDE_CompleteInterruptInRequest(
    _In_ PDEVICE_CONTEXT DeviceContext);

// Helper functions
VOID ChunithmUDE_InitializeReport(_Out_ PCHUNITHM_INPUT_REPORT Report);
NTSTATUS ChunithmUDE_ValidateReport(_In_ PCHUNITHM_INPUT_REPORT Report);

#endif  // _CHUNITHM_UDE_H_
