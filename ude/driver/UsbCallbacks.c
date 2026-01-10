/*++

Module Name:
    UsbCallbacks.c

Abstract:
    USB device callbacks for ChunithmUDE driver.
    Handles USB control requests and string descriptors.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(PAGE, EvtUsbDeviceDefaultEndpointAdd)
#pragma alloc_text(PAGE, EvtUsbDeviceEndpointAdd)
#pragma alloc_text(PAGE, EvtControlUrb)
#endif

//
// USB device default endpoint add callback
//
NTSTATUS
EvtUsbDeviceDefaultEndpointAdd(_In_ UDECXUSBDEVICE UdecxUsbDevice,
                               _In_ PUDECXUSBENDPOINT_INIT EndpointInit) {
  NTSTATUS status;
  UDECX_USB_ENDPOINT_CALLBACKS callbacks;
  UDECXUSBENDPOINT endpoint;
  WDF_OBJECT_ATTRIBUTES attributes;

  PAGED_CODE();

  KdPrint(("ChunithmUDE: EvtUsbDeviceDefaultEndpointAdd\n"));

  //
  // Initialize endpoint callbacks for control endpoint
  //
  UDECX_USB_ENDPOINT_CALLBACKS_INIT(&callbacks, EvtEndpointReset);

  UdecxUsbEndpointInitSetCallbacks(EndpointInit, &callbacks);

  //
  // Set endpoint address (EP0)
  //
  UdecxUsbEndpointInitSetEndpointAddress(EndpointInit, 0);

  //
  // Create endpoint
  //
  WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
  attributes.ParentObject = UdecxUsbDevice;

  status = UdecxUsbEndpointCreate(&EndpointInit, &attributes, &endpoint);
  if (!NT_SUCCESS(status)) {
    KdPrint(("ChunithmUDE: Failed to create default endpoint: 0x%x\n", status));
    return status;
  }

  UdecxUsbEndpointSetWdfIoQueue(endpoint, WDF_NO_HANDLE);

  return STATUS_SUCCESS;
}

//
// USB device endpoint add callback
//
NTSTATUS
EvtUsbDeviceEndpointAdd(_In_ UDECXUSBDEVICE UdecxUsbDevice,
                        _In_ PUDECXUSBENDPOINT_INIT EndpointInit) {
  UNREFERENCED_PARAMETER(UdecxUsbDevice);
  UNREFERENCED_PARAMETER(EndpointInit);

  PAGED_CODE();

  //
  // Endpoints are created in Usb_CreateEndpoints, not dynamically
  //
  return STATUS_SUCCESS;
}

//
// USB control URB handler
//
VOID EvtControlUrb(_In_ WDFREQUEST Request, _In_ WDFMEMORY Memory) {
  NTSTATUS status = STATUS_SUCCESS;
  PURB urb;
  struct _URB_CONTROL_DESCRIPTOR_REQUEST* descriptorRequest;
  USHORT descriptorType;
  USHORT descriptorIndex;
  PVOID transferBuffer;
  ULONG transferBufferLength;

  PAGED_CODE();

  urb = (PURB)WdfMemoryGetBuffer(Memory, NULL);

  switch (urb->UrbHeader.Function) {
    case URB_FUNCTION_GET_DESCRIPTOR_FROM_DEVICE:
      descriptorRequest = &urb->UrbControlDescriptorRequest;
      descriptorType = descriptorRequest->DescriptorType;
      descriptorIndex = descriptorRequest->Index;
      transferBuffer = descriptorRequest->TransferBuffer;
      transferBufferLength = descriptorRequest->TransferBufferLength;

      switch (descriptorType) {
        case USB_STRING_DESCRIPTOR_TYPE:
          status = UsbGetStringDescriptor(descriptorIndex, transferBuffer,
                                          &transferBufferLength);
          if (NT_SUCCESS(status)) {
            descriptorRequest->TransferBufferLength = transferBufferLength;
          }
          break;

        case USB_DEVICE_DESCRIPTOR_TYPE:
        case USB_CONFIGURATION_DESCRIPTOR_TYPE:
          //
          // Device and config descriptors handled by UDE framework
          //
          status = STATUS_NOT_SUPPORTED;
          break;

        default:
          status = STATUS_INVALID_DEVICE_REQUEST;
          break;
      }
      break;

    case URB_FUNCTION_SELECT_CONFIGURATION:
      //
      // Configuration selection handled by framework
      //
      status = STATUS_SUCCESS;
      break;

    case URB_FUNCTION_SELECT_INTERFACE:
      //
      // Interface selection handled by framework
      //
      status = STATUS_SUCCESS;
      break;

    default:
      KdPrint(("ChunithmUDE: Unknown URB function: 0x%x\n",
               urb->UrbHeader.Function));
      status = STATUS_INVALID_DEVICE_REQUEST;
      break;
  }

  WdfRequestComplete(Request, status);
}

//
// Get USB string descriptor
//
NTSTATUS
UsbGetStringDescriptor(_In_ UCHAR Index,
                       _Out_writes_bytes_(*Length) PVOID Buffer,
                       _Inout_ PULONG Length) {
  PUSB_STRING_DESCRIPTOR stringDescriptor;
  const WCHAR* sourceString;
  SIZE_T stringLength;
  ULONG requiredLength;

  PAGED_CODE();

  stringDescriptor = (PUSB_STRING_DESCRIPTOR)Buffer;

  switch (Index) {
    case 0:
      //
      // Language ID descriptor
      //
      if (*Length < sizeof(USB_STRING_DESCRIPTOR)) {
        *Length = sizeof(USB_STRING_DESCRIPTOR);
        return STATUS_BUFFER_TOO_SMALL;
      }

      stringDescriptor->bLength = sizeof(USB_STRING_DESCRIPTOR);
      stringDescriptor->bDescriptorType = USB_STRING_DESCRIPTOR_TYPE;
      stringDescriptor->bString[0] = 0x0409;  // English (US)
      *Length = sizeof(USB_STRING_DESCRIPTOR);
      break;

    case 1:
      //
      // Manufacturer string
      //
      sourceString = CHUNITHM_MANUFACTURER;
      stringLength = wcslen(sourceString);
      requiredLength = (ULONG)(sizeof(USB_STRING_DESCRIPTOR) +
                               (stringLength - 1) * sizeof(WCHAR));

      if (*Length < requiredLength) {
        *Length = requiredLength;
        return STATUS_BUFFER_TOO_SMALL;
      }

      stringDescriptor->bLength = (UCHAR)requiredLength;
      stringDescriptor->bDescriptorType = USB_STRING_DESCRIPTOR_TYPE;
      RtlCopyMemory(stringDescriptor->bString, sourceString,
                    stringLength * sizeof(WCHAR));
      *Length = requiredLength;
      break;

    case 2:
      //
      // Product string
      //
      sourceString = CHUNITHM_PRODUCT;
      stringLength = wcslen(sourceString);
      requiredLength = (ULONG)(sizeof(USB_STRING_DESCRIPTOR) +
                               (stringLength - 1) * sizeof(WCHAR));

      if (*Length < requiredLength) {
        *Length = requiredLength;
        return STATUS_BUFFER_TOO_SMALL;
      }

      stringDescriptor->bLength = (UCHAR)requiredLength;
      stringDescriptor->bDescriptorType = USB_STRING_DESCRIPTOR_TYPE;
      RtlCopyMemory(stringDescriptor->bString, sourceString,
                    stringLength * sizeof(WCHAR));
      *Length = requiredLength;
      break;

    case 3:
      //
      // Serial number string
      //
      sourceString = CHUNITHM_SERIAL;
      stringLength = wcslen(sourceString);
      requiredLength = (ULONG)(sizeof(USB_STRING_DESCRIPTOR) +
                               (stringLength - 1) * sizeof(WCHAR));

      if (*Length < requiredLength) {
        *Length = requiredLength;
        return STATUS_BUFFER_TOO_SMALL;
      }

      stringDescriptor->bLength = (UCHAR)requiredLength;
      stringDescriptor->bDescriptorType = USB_STRING_DESCRIPTOR_TYPE;
      RtlCopyMemory(stringDescriptor->bString, sourceString,
                    stringLength * sizeof(WCHAR));
      *Length = requiredLength;
      break;

    default:
      return STATUS_INVALID_PARAMETER;
  }

  return STATUS_SUCCESS;
}
