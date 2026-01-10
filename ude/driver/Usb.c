/*++

Module Name:
    Usb.c

Abstract:
    USB device and endpoint creation for ChunithmUDE driver.
    Defines USB descriptors and creates UDE endpoints.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(PAGE, Usb_CreateUsbDevice)
#pragma alloc_text(PAGE, Usb_CreateEndpoints)
#pragma alloc_text(PAGE, Usb_GetDescriptors)
#endif

//
// Raw HID descriptor for Chunithm controller
//
static const UCHAR g_HidReportDescriptor[] = {
    0x06, 0xC0, 0xFF,  // Usage Page (Vendor 0xFFC0)
    0x0A, 0x00, 0x0C,  // Usage (0x0C00)
    0xA1, 0x01,        // Collection (Application)
    0x75, 0x08,        //   Report Size (8 bits)
    0x15, 0x00,        //   Logical Minimum (0)
    0x26, 0xFF, 0x00,  //   Logical Maximum (255)
    0x95, 45,          //   Report Count (45) - Input
    0x09, 0x01,        //   Usage (1)
    0x81, 0x02,        //   Input (Data,Var,Abs)
    0x95, 61,          //   Report Count (61) - Output
    0x09, 0x02,        //   Usage (2)
    0x91, 0x02,        //   Output (Data,Var,Abs)
    0xC0               //   End Collection
};

//
// Create UDE USB device
//
NTSTATUS
Usb_CreateUsbDevice(_In_ WDFDEVICE WdfDevice) {
  NTSTATUS status;
  PDEVICE_CONTEXT deviceContext;
  UDECX_USB_DEVICE_CALLBACKS callbacks;
  UDECXUSBDEVICE_INIT* usbDeviceInit = NULL;
  USB_DEVICE_DESCRIPTOR deviceDescriptor;
  PUSB_CONFIGURATION_DESCRIPTOR configDescriptor = NULL;
  ULONG configDescriptorSize;

  PAGED_CODE();

  deviceContext = GetDeviceContext(WdfDevice);

  //
  // Get USB descriptors
  //
  status = Usb_GetDescriptors(&deviceDescriptor, &configDescriptor,
                              &configDescriptorSize);
  if (!NT_SUCCESS(status)) {
    goto Exit;
  }

  //
  // Initialize UDE device callbacks (framework - not implemented)
  //
  UDECX_USB_DEVICE_CALLBACKS_INIT(&callbacks);

  //
  // Allocate UDE device init structure
  //
  usbDeviceInit = UdecxUsbDeviceInitAllocate(WdfDevice);
  if (usbDeviceInit == NULL) {
    status = STATUS_INSUFFICIENT_RESOURCES;
    goto Exit;
  }

  //
  // Set USB descriptors
  //
  UdecxUsbDeviceInitSetDeviceDescriptor(usbDeviceInit, &deviceDescriptor);

  //
  // Set speed (USB 2.0 Full Speed)
  //
  UdecxUsbDeviceInitSetSpeed(usbDeviceInit, UdecxUsbFullSpeed);

  //
  // Set endpoint characteristics
  //
  UdecxUsbDeviceInitSetEndpointsType(usbDeviceInit, UdecxEndpointTypeSimple);

  //
  // Create UDE USB device
  //
  status = UdecxUsbDeviceCreate(&usbDeviceInit, WDF_NO_OBJECT_ATTRIBUTES,
                                &deviceContext->UdecxUsbDevice);
  if (!NT_SUCCESS(status)) {
    goto Exit;
  }

  //
  // Create endpoints
  //
  status = Usb_CreateEndpoints(deviceContext);
  if (!NT_SUCCESS(status)) {
    goto Exit;
  }

  KdPrint(("ChunithmUDE: USB device created\n"));

Exit:
  if (configDescriptor != NULL) {
    ExFreePoolWithTag(configDescriptor, CHUNITHM_UDE_POOL_TAG);
  }

  if (usbDeviceInit != NULL) {
    UdecxUsbDeviceInitFree(usbDeviceInit);
  }

  return status;
}

//
// Create USB endpoints
//
NTSTATUS
Usb_CreateEndpoints(_In_ PDEVICE_CONTEXT DeviceContext) {
  NTSTATUS status;
  UDECX_USB_ENDPOINT_CALLBACKS callbacks;
  PUDECXUSBENDPOINT_INIT endpointInit = NULL;
  USB_ENDPOINT_DESCRIPTOR endpointDescriptor;
  WDF_OBJECT_ATTRIBUTES endpointAttributes;

  PAGED_CODE();

  //
  // Initialize endpoint callbacks
  //
  UDECX_USB_ENDPOINT_CALLBACKS_INIT(&callbacks, EvtEndpointReset);
  callbacks.EvtUsbEndpointStart = EvtEndpointStart;
  callbacks.EvtUsbEndpointPurge = EvtEndpointPurge;

  //
  // Create interrupt IN endpoint (HID input reports)
  //
  endpointInit =
      UdecxUsbSimpleEndpointInitAllocate(DeviceContext->UdecxUsbDevice);
  if (endpointInit == NULL) {
    status = STATUS_INSUFFICIENT_RESOURCES;
    goto Exit;
  }

  //
  // Set endpoint descriptor
  //
  RtlZeroMemory(&endpointDescriptor, sizeof(endpointDescriptor));
  endpointDescriptor.bLength = sizeof(USB_ENDPOINT_DESCRIPTOR);
  endpointDescriptor.bDescriptorType = USB_ENDPOINT_DESCRIPTOR_TYPE;
  endpointDescriptor.bEndpointAddress = 0x81;  // EP1 IN
  endpointDescriptor.bmAttributes = USB_ENDPOINT_TYPE_INTERRUPT;
  endpointDescriptor.wMaxPacketSize = CHUNITHM_INPUT_REPORT_SIZE;
  endpointDescriptor.bInterval = 1;  // 1ms polling

  UdecxUsbEndpointInitSetEndpointAddress(endpointInit,
                                         endpointDescriptor.bEndpointAddress);
  UdecxUsbEndpointInitSetCallbacks(endpointInit, &callbacks);

  //
  // Create endpoint object
  //
  WDF_OBJECT_ATTRIBUTES_INIT(&endpointAttributes);
  endpointAttributes.ParentObject = DeviceContext->UdecxUsbDevice;

  status = UdecxUsbEndpointCreate(&endpointInit, &endpointAttributes,
                                  &DeviceContext->InterruptInEndpoint);
  if (!NT_SUCCESS(status)) {
    goto Exit;
  }

  //
  // Framework: Output endpoint for RGB data not implemented
  //

  KdPrint(("ChunithmUDE: Endpoints created\n"));

Exit:
  if (endpointInit != NULL) {
    UdecxUsbEndpointInitFree(endpointInit);
  }

  return status;
}

//
// Get USB descriptors
//
NTSTATUS
Usb_GetDescriptors(_Out_ PUSB_DEVICE_DESCRIPTOR DeviceDescriptor,
                   _Out_ PUSB_CONFIGURATION_DESCRIPTOR* ConfigDescriptor,
                   _Out_ PULONG ConfigDescriptorSize) {
  PAGED_CODE();

  //
  // Device descriptor
  //
  RtlZeroMemory(DeviceDescriptor, sizeof(USB_DEVICE_DESCRIPTOR));
  DeviceDescriptor->bLength = sizeof(USB_DEVICE_DESCRIPTOR);
  DeviceDescriptor->bDescriptorType = USB_DEVICE_DESCRIPTOR_TYPE;
  DeviceDescriptor->bcdUSB = 0x0200;  // USB 2.0
  DeviceDescriptor->bDeviceClass = 0x00;
  DeviceDescriptor->bDeviceSubClass = 0x00;
  DeviceDescriptor->bDeviceProtocol = 0x00;
  DeviceDescriptor->bMaxPacketSize0 = 64;
  DeviceDescriptor->idVendor = CHUNITHM_VENDOR_ID;
  DeviceDescriptor->idProduct = CHUNITHM_PRODUCT_ID;
  DeviceDescriptor->bcdDevice = CHUNITHM_DEVICE_VERSION;
  DeviceDescriptor->iManufacturer = 1;
  DeviceDescriptor->iProduct = 2;
  DeviceDescriptor->iSerialNumber = 3;
  DeviceDescriptor->bNumConfigurations = 1;

  //
  // Configuration descriptor (framework - simplified)
  //
  *ConfigDescriptorSize = sizeof(USB_CONFIGURATION_DESCRIPTOR);
  *ConfigDescriptor = (PUSB_CONFIGURATION_DESCRIPTOR)ExAllocatePoolWithTag(
      NonPagedPoolNx, *ConfigDescriptorSize, CHUNITHM_UDE_POOL_TAG);

  if (*ConfigDescriptor == NULL) {
    return STATUS_INSUFFICIENT_RESOURCES;
  }

  RtlZeroMemory(*ConfigDescriptor, *ConfigDescriptorSize);
  (*ConfigDescriptor)->bLength = sizeof(USB_CONFIGURATION_DESCRIPTOR);
  (*ConfigDescriptor)->bDescriptorType = USB_CONFIGURATION_DESCRIPTOR_TYPE;
  (*ConfigDescriptor)->wTotalLength = (USHORT)*ConfigDescriptorSize;
  (*ConfigDescriptor)->bNumInterfaces = 1;
  (*ConfigDescriptor)->bConfigurationValue = 1;
  (*ConfigDescriptor)->bmAttributes = 0x80;  // Bus-powered
  (*ConfigDescriptor)->MaxPower = 50;        // 100mA

  return STATUS_SUCCESS;
}

//
// Endpoint reset callback (framework stub)
//
VOID EvtEndpointReset(_In_ UDECXUSBENDPOINT Endpoint, _In_ WDFREQUEST Request) {
  UNREFERENCED_PARAMETER(Endpoint);

  WdfRequestComplete(Request, STATUS_SUCCESS);
}

//
// Endpoint start callback (framework stub)
//
VOID EvtEndpointStart(_In_ UDECXUSBENDPOINT Endpoint) {
  UNREFERENCED_PARAMETER(Endpoint);

  KdPrint(("ChunithmUDE: Endpoint started\n"));
}

//
// Endpoint purge callback (framework stub)
//
VOID EvtEndpointPurge(_In_ UDECXUSBENDPOINT Endpoint) {
  UNREFERENCED_PARAMETER(Endpoint);

  KdPrint(("ChunithmUDE: Endpoint purged\n"));
}
