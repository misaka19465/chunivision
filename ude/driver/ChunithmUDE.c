/*++

Module Name:
    ChunithmUDE.c

Abstract:
    Main driver module for ChunithmUDE.
    Implements driver entry point and device initialization.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(INIT, DriverEntry)
#pragma alloc_text(PAGE, EvtDeviceAdd)
#pragma alloc_text(PAGE, EvtDriverContextCleanup)
#endif

//
// Driver entry point
//
NTSTATUS
DriverEntry(_In_ PDRIVER_OBJECT DriverObject,
            _In_ PUNICODE_STRING RegistryPath) {
  WDF_DRIVER_CONFIG config;
  NTSTATUS status;
  WDF_OBJECT_ATTRIBUTES attributes;

  KdPrint(("ChunithmUDE: DriverEntry\n"));

  //
  // Initialize driver configuration
  //
  WDF_DRIVER_CONFIG_INIT(&config, EvtDeviceAdd);

  //
  // Set cleanup callback
  //
  WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
  attributes.EvtCleanupCallback = EvtDriverContextCleanup;

  //
  // Create the driver object
  //
  status = WdfDriverCreate(DriverObject, RegistryPath, &attributes, &config,
                           WDF_NO_HANDLE);

  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: WdfDriverCreate failed: 0x%x\n", status));
    return status;
  }

  KdPrint(("ChunithmUDE: Driver loaded successfully\n"));
  return STATUS_SUCCESS;
}

//
// Device add callback
//
NTSTATUS
EvtDeviceAdd(_In_ WDFDRIVER Driver, _In_ PWDFDEVICE_INIT DeviceInit) {
  NTSTATUS status;

  UNREFERENCED_PARAMETER(Driver);

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtDeviceAdd\n"));

  //
  // Initialize UDE
  //
  status = UdecxInitializeWdfDeviceInit(DeviceInit);
  if (!NT_SUCCESS(status)) {
    KdPrint(
        ("ChunithmUDE: UdecxInitializeWdfDeviceInit failed: 0x%x\n", status));
    return status;
  }

  //
  // Create device
  //
  status = ChunithmUDE_CreateDevice(DeviceInit);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: ChunithmUDE_CreateDevice failed: 0x%x\n", status));
    return status;
  }

  return STATUS_SUCCESS;
}

//
// Driver cleanup callback
//
VOID EvtDriverContextCleanup(_In_ WDFOBJECT DriverObject) {
  UNREFERENCED_PARAMETER(DriverObject);

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtDriverContextCleanup\n"));
}

//
// Create and initialize device
//
NTSTATUS
ChunithmUDE_CreateDevice(_In_ PWDFDEVICE_INIT DeviceInit) {
  WDF_OBJECT_ATTRIBUTES deviceAttributes;
  WDFDEVICE device;
  PDEVICE_CONTEXT deviceContext;
  NTSTATUS status;
  WDF_PNPPOWER_EVENT_CALLBACKS pnpPowerCallbacks;

  PAGED_CODE();

  //
  // Set PnP and power callbacks
  //
  WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&pnpPowerCallbacks);
  pnpPowerCallbacks.EvtDevicePrepareHardware = EvtDevicePrepareHardware;
  pnpPowerCallbacks.EvtDeviceReleaseHardware = EvtDeviceReleaseHardware;
  pnpPowerCallbacks.EvtDeviceD0Entry = EvtDeviceD0Entry;
  pnpPowerCallbacks.EvtDeviceD0Exit = EvtDeviceD0Exit;

  WdfDeviceInitSetPnpPowerEventCallbacks(DeviceInit, &pnpPowerCallbacks);

  //
  // Initialize device attributes with context
  //
  WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&deviceAttributes, DEVICE_CONTEXT);

  //
  // Create WDFDEVICE
  //
  status = WdfDeviceCreate(&DeviceInit, &deviceAttributes, &device);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: WdfDeviceCreate failed: 0x%x\n", status));
    return status;
  }

  //
  // Get device context and initialize
  //
  deviceContext = GetDeviceContext(device);
  deviceContext->WdfDevice = device;
  deviceContext->DeviceReady = FALSE;

  //
  // Initialize spinlock for report protection
  //
  WDF_OBJECT_ATTRIBUTES spinlockAttributes;
  WDF_OBJECT_ATTRIBUTES_INIT(&spinlockAttributes);
  spinlockAttributes.ParentObject = device;

  status = WdfSpinLockCreate(&spinlockAttributes, &deviceContext->ReportLock);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: WdfSpinLockCreate failed: 0x%x\n", status));
    return status;
  }

  //
  // Create device interface
  //
  status = WdfDeviceCreateDeviceInterface(
      device, &GUID_DEVINTERFACE_CHUNITHM_UDE, NULL);
  if (!NT_SUCCESS(status)) {
    KdPrint(
        ("ChunithmUDE: WdfDeviceCreateDeviceInterface failed: 0x%x\n", status));
    return status;
  }

  //
  // Initialize I/O queues
  //
  status = Queue_Initialize(device);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: Queue_Initialize failed: 0x%x\n", status));
    return status;
  }

  //
  // Create UDE USB device
  //
  status = Usb_CreateUsbDevice(device);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: Usb_CreateUsbDevice failed: 0x%x\n", status));
    return status;
  }

  KdPrint(("ChunithmUDE: Device created successfully\n"));
  return STATUS_SUCCESS;
}

//
// Initialize default report values
//
VOID ChunithmUDE_InitializeReport(_Out_ PCHUNITHM_INPUT_REPORT Report) {
  RtlZeroMemory(Report, sizeof(CHUNITHM_INPUT_REPORT));
  Report->IRValue = 0x00;
  Report->Buttons = 0x00;
  Report->CardStatus = 0x00;
}

//
// Validate report data
//
NTSTATUS
ChunithmUDE_ValidateReport(_In_ PCHUNITHM_INPUT_REPORT Report) {
  // Basic validation: air sensors should use only low 6 bits
  if (Report->IRValue & 0xC0) {
    return STATUS_INVALID_PARAMETER;
  }

  // Card status should be 0-2
  if (Report->CardStatus > 2) {
    return STATUS_INVALID_PARAMETER;
  }

  return STATUS_SUCCESS;
}
