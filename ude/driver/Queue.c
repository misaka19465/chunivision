/*++

Module Name:
    Queue.c

Abstract:
    I/O queue initialization and handlers for ChunithmUDE driver.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(PAGE, Queue_Initialize)
#pragma alloc_text(PAGE, EvtIoDeviceControl)
#endif

//
// Initialize I/O queues
//
NTSTATUS
Queue_Initialize(_In_ WDFDEVICE Device) {
  NTSTATUS status;
  WDF_IO_QUEUE_CONFIG queueConfig;
  WDFQUEUE queue;
  PDEVICE_CONTEXT deviceContext;

  PAGED_CODE();

  deviceContext = GetDeviceContext(Device);

  //
  // Create default queue for IOCTL requests
  //
  WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&queueConfig,
                                         WdfIoQueueDispatchSequential);

  queueConfig.EvtIoDeviceControl = EvtIoDeviceControl;
  queueConfig.EvtIoRead = EvtIoRead;
  queueConfig.EvtIoWrite = EvtIoWrite;

  status =
      WdfIoQueueCreate(Device, &queueConfig, WDF_NO_OBJECT_ATTRIBUTES, &queue);

  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: WdfIoQueueCreate failed: 0x%x\n", status));
    return status;
  }

  deviceContext->DefaultQueue = queue;

  KdPrint(("ChunithmUDE: I/O queues initialized\n"));
  return STATUS_SUCCESS;
}

//
// IOCTL handler
//
VOID EvtIoDeviceControl(_In_ WDFQUEUE Queue, _In_ WDFREQUEST Request,
                        _In_ size_t OutputBufferLength,
                        _In_ size_t InputBufferLength,
                        _In_ ULONG IoControlCode) {
  NTSTATUS status = STATUS_INVALID_DEVICE_REQUEST;
  PDEVICE_CONTEXT deviceContext;
  PVOID inputBuffer;
  size_t inputBufferSize;

  UNREFERENCED_PARAMETER(OutputBufferLength);

  PAGED_CODE();

  deviceContext = GetDeviceContext(WdfIoQueueGetDevice(Queue));

  switch (IoControlCode) {
    case IOCTL_CHUNITHM_SEND_REPORT:
      //
      // Send HID input report
      //
      if (InputBufferLength < CHUNITHM_INPUT_REPORT_SIZE) {
        status = STATUS_BUFFER_TOO_SMALL;
        break;
      }

      status = WdfRequestRetrieveInputBuffer(
          Request, CHUNITHM_INPUT_REPORT_SIZE, &inputBuffer, &inputBufferSize);

      if (NT_SUCCESS(status)) {
        status = ChunithmUDE_SendReport(deviceContext,
                                        (PCHUNITHM_INPUT_REPORT)inputBuffer);
      }
      break;

    case IOCTL_CHUNITHM_GET_STATUS:
      //
      // Get driver status (not implemented in framework)
      //
      status = STATUS_NOT_IMPLEMENTED;
      break;

    case IOCTL_CHUNITHM_RESET:
      //
      // Reset device (not implemented in framework)
      //
      status = STATUS_NOT_IMPLEMENTED;
      break;

    default:
      status = STATUS_INVALID_DEVICE_REQUEST;
      break;
  }

  WdfRequestComplete(Request, status);
}

//
// Read handler (not used)
//
VOID EvtIoRead(_In_ WDFQUEUE Queue, _In_ WDFREQUEST Request,
               _In_ size_t Length) {
  UNREFERENCED_PARAMETER(Queue);
  UNREFERENCED_PARAMETER(Length);

  PAGED_CODE();

  WdfRequestComplete(Request, STATUS_NOT_IMPLEMENTED);
}

//
// Write handler (not used)
//
VOID EvtIoWrite(_In_ WDFQUEUE Queue, _In_ WDFREQUEST Request,
                _In_ size_t Length) {
  UNREFERENCED_PARAMETER(Queue);
  UNREFERENCED_PARAMETER(Length);

  PAGED_CODE();

  WdfRequestComplete(Request, STATUS_NOT_IMPLEMENTED);
}

//
// Send HID report to pending interrupt IN request
//
NTSTATUS
ChunithmUDE_SendReport(_In_ PDEVICE_CONTEXT DeviceContext,
                       _In_ PCHUNITHM_INPUT_REPORT Report) {
  NTSTATUS status;
  KLOCK_QUEUE_HANDLE lockHandle;

  //
  // Validate report
  //
  status = ChunithmUDE_ValidateReport(Report);
  if (!NT_SUCCESS(status)) {
    return status;
  }

  //
  // Store report in device context
  //
  WdfSpinLockAcquire(DeviceContext->ReportLock, &lockHandle);
  RtlCopyMemory(DeviceContext->CurrentReport, Report,
                CHUNITHM_INPUT_REPORT_SIZE);
  WdfSpinLockRelease(DeviceContext->ReportLock, lockHandle);

  //
  // Complete any pending interrupt IN request
  //
  status = ChunithmUDE_CompleteInterruptInRequest(DeviceContext);

  return status;
}

//
// Complete pending interrupt IN request (stub)
//
NTSTATUS
ChunithmUDE_CompleteInterruptInRequest(_In_ PDEVICE_CONTEXT DeviceContext) {
  UNREFERENCED_PARAMETER(DeviceContext);

  //
  // Framework: actual implementation would retrieve pending
  // request from InterruptInQueue and complete it with report data
  //

  return STATUS_SUCCESS;
}
