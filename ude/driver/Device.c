/*++

Module Name:
    Device.c

Abstract:
    PnP and power management callbacks for ChunithmUDE driver.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(PAGE, EvtDevicePrepareHardware)
#pragma alloc_text(PAGE, EvtDeviceReleaseHardware)
#pragma alloc_text(PAGE, EvtDeviceD0Entry)
#pragma alloc_text(PAGE, EvtDeviceD0Exit)
#endif

//
// Prepare hardware callback
//
NTSTATUS
EvtDevicePrepareHardware(_In_ WDFDEVICE Device, _In_ WDFCMRESLIST ResourcesRaw,
                         _In_ WDFCMRESLIST ResourcesTranslated) {
  PDEVICE_CONTEXT deviceContext;

  UNREFERENCED_PARAMETER(ResourcesRaw);
  UNREFERENCED_PARAMETER(ResourcesTranslated);

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtDevicePrepareHardware\n"));

  deviceContext = GetDeviceContext(Device);

  //
  // Initialize default report
  //
  ChunithmUDE_InitializeReport(
      (PCHUNITHM_INPUT_REPORT)deviceContext->CurrentReport);

  return STATUS_SUCCESS;
}

//
// Release hardware callback
//
NTSTATUS
EvtDeviceReleaseHardware(_In_ WDFDEVICE Device,
                         _In_ WDFCMRESLIST ResourcesTranslated) {
  UNREFERENCED_PARAMETER(Device);
  UNREFERENCED_PARAMETER(ResourcesTranslated);

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtDeviceReleaseHardware\n"));

  return STATUS_SUCCESS;
}

//
// D0 entry callback (power up)
//
NTSTATUS
EvtDeviceD0Entry(_In_ WDFDEVICE Device,
                 _In_ WDF_POWER_DEVICE_STATE PreviousState) {
  PDEVICE_CONTEXT deviceContext;

  UNREFERENCED_PARAMETER(PreviousState);

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtDeviceD0Entry\n"));

  deviceContext = GetDeviceContext(Device);
  deviceContext->DeviceReady = TRUE;

  //
  // Framework: Actual implementation would plug in the UDE device here
  // using UdecxUsbDevicePlugIn
  //

  return STATUS_SUCCESS;
}

//
// D0 exit callback (power down)
//
NTSTATUS
EvtDeviceD0Exit(_In_ WDFDEVICE Device,
                _In_ WDF_POWER_DEVICE_STATE TargetState) {
  PDEVICE_CONTEXT deviceContext;

  UNREFERENCED_PARAMETER(TargetState);

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtDeviceD0Exit\n"));

  deviceContext = GetDeviceContext(Device);
  deviceContext->DeviceReady = FALSE;

  //
  // Framework: Actual implementation would unplug the UDE device here
  // using UdecxUsbDevicePlugOutAndDelete
  //

  return STATUS_SUCCESS;
}
