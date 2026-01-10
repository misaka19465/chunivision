/*++

Module Name:
    EndpointQueue.c

Abstract:
    Endpoint queue management for ChunithmUDE driver.
    Handles interrupt IN endpoint queue for HID input reports.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(PAGE, EndpointQueue_Initialize)
#pragma alloc_text(PAGE, EvtEndpointQueueReadyNotification)
#endif

//
// Initialize endpoint queue
//
NTSTATUS
EndpointQueue_Initialize(_In_ PDEVICE_CONTEXT DeviceContext) {
  NTSTATUS status;
  WDF_IO_QUEUE_CONFIG queueConfig;
  WDFQUEUE queue;
  WDF_OBJECT_ATTRIBUTES attributes;

  PAGED_CODE();

  //
  // Create manual queue for interrupt IN endpoint
  //
  WDF_IO_QUEUE_CONFIG_INIT(&queueConfig, WdfIoQueueDispatchManual);

  queueConfig.EvtIoStop = EvtEndpointQueueIoStop;

  WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
  attributes.ParentObject = DeviceContext->Device;

  status = WdfIoQueueCreate(DeviceContext->Device, &queueConfig, &attributes,
                            &queue);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: Failed to create endpoint queue: 0x%x\n", status));
    return status;
  }

  DeviceContext->InterruptInQueue = queue;

  //
  // Assign queue to endpoint
  //
  UdecxUsbEndpointSetWdfIoQueue(DeviceContext->InterruptInEndpoint, queue);

  //
  // Initialize ready notification
  //
  UdecxUsbEndpointInitSetCallbacks

      KdPrint(("ChunithmUDE: Endpoint queue initialized\n"));
  return STATUS_SUCCESS;
}

//
// Endpoint queue ready notification
//
VOID EvtEndpointQueueReadyNotification(_In_ WDFQUEUE Queue,
                                       _In_ WDFCONTEXT Context) {
  PDEVICE_CONTEXT deviceContext = (PDEVICE_CONTEXT)Context;
  KLOCK_QUEUE_HANDLE lockHandle;

  PAGED_CODE();

  UNREFERENCED_PARAMETER(Queue);

  KdPrint(("ChunithmUDE: Endpoint queue ready\n"));

  //
  // Complete pending request with current report
  //
  WdfSpinLockAcquire(deviceContext->ReportLock, &lockHandle);

  if (deviceContext->PendingReportAvailable) {
    ChunithmUDE_CompleteInterruptInRequest(deviceContext);
    deviceContext->PendingReportAvailable = FALSE;
  }

  WdfSpinLockRelease(deviceContext->ReportLock, lockHandle);
}

//
// Endpoint queue I/O stop callback
//
VOID EvtEndpointQueueIoStop(_In_ WDFQUEUE Queue, _In_ WDFREQUEST Request,
                            _In_ ULONG ActionFlags) {
  UNREFERENCED_PARAMETER(Queue);

  if (ActionFlags & WdfRequestStopActionSuspend) {
    WdfRequestStopAcknowledge(Request, FALSE);
  } else if (ActionFlags & WdfRequestStopActionPurge) {
    WdfRequestComplete(Request, STATUS_CANCELLED);
  }
}

//
// Complete interrupt IN request with report data
//
NTSTATUS
ChunithmUDE_CompleteInterruptInRequest(_In_ PDEVICE_CONTEXT DeviceContext) {
  NTSTATUS status;
  WDFREQUEST request;
  PVOID buffer;
  size_t bufferLength;

  //
  // Retrieve next request from manual queue
  //
  status =
      WdfIoQueueRetrieveNextRequest(DeviceContext->InterruptInQueue, &request);
  if (!NT_SUCCESS(status)) {
    //
    // No pending request, set flag for later completion
    //
    if (status == STATUS_NO_MORE_ENTRIES) {
      DeviceContext->PendingReportAvailable = TRUE;
      return STATUS_SUCCESS;
    }
    return status;
  }

  //
  // Get request buffer
  //
  status = WdfRequestRetrieveOutputBuffer(request, CHUNITHM_INPUT_REPORT_SIZE,
                                          &buffer, &bufferLength);
  if (!NT_SUCCESS(status)) {
    WdfRequestComplete(request, status);
    return status;
  }

  //
  // Copy report data to buffer
  //
  RtlCopyMemory(buffer, DeviceContext->CurrentReport,
                CHUNITHM_INPUT_REPORT_SIZE);

  //
  // Complete request
  //
  WdfRequestSetInformation(request, CHUNITHM_INPUT_REPORT_SIZE);
  WdfRequestComplete(request, STATUS_SUCCESS);

  //
  // Update statistics
  //
  InterlockedIncrement(&DeviceContext->ReportsSent);

  KdPrint(("ChunithmUDE: Interrupt IN request completed\n"));

  return STATUS_SUCCESS;
}

//
// Forward interrupt IN request to endpoint queue
//
VOID EvtEndpointReadUrb(_In_ WDFREQUEST Request) {
  PDEVICE_CONTEXT deviceContext;
  WDFDEVICE device;

  device = WdfIoQueueGetDevice(WdfRequestGetIoQueue(Request));
  deviceContext = GetDeviceContext(device);

  //
  // Forward to manual queue
  //
  WdfRequestForwardToIoQueue(Request, deviceContext->InterruptInQueue);
}
