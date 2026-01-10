/*++

Module Name:
    Descriptors.c

Abstract:
    Complete USB and HID descriptors for ChunithmUDE driver.
    Provides full descriptor set for Chunithm controller emulation.

Author:
    Misaka 19465

Environment:
    Kernel mode

--*/

#include "ChunithmUDE.h"

#ifdef ALLOC_PRAGMA
#pragma alloc_text(PAGE, Descriptors_GetCompleteConfigDescriptor)
#pragma alloc_text(PAGE, Descriptors_GetHidDescriptor)
#pragma alloc_text(PAGE, Descriptors_GetHidReportDescriptor)
#endif

//
// HID descriptor structure
//
#pragma pack(push, 1)
typedef struct _HID_DESCRIPTOR {
  UCHAR bLength;
  UCHAR bDescriptorType;
  USHORT bcdHID;
  UCHAR bCountryCode;
  UCHAR bNumDescriptors;
  UCHAR bReportDescriptorType;
  USHORT wReportDescriptorLength;
} HID_DESCRIPTOR, *PHID_DESCRIPTOR;

//
// Complete configuration descriptor with interface, HID, and endpoints
//
typedef struct _COMPLETE_CONFIG_DESCRIPTOR {
  USB_CONFIGURATION_DESCRIPTOR ConfigDescriptor;
  USB_INTERFACE_DESCRIPTOR InterfaceDescriptor;
  HID_DESCRIPTOR HidDescriptor;
  USB_ENDPOINT_DESCRIPTOR EndpointDescriptorIn;
  USB_ENDPOINT_DESCRIPTOR EndpointDescriptorOut;
} COMPLETE_CONFIG_DESCRIPTOR, *PCOMPLETE_CONFIG_DESCRIPTOR;
#pragma pack(pop)

//
// HID report descriptor (from Usb.c)
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
// Get complete configuration descriptor with all sub-descriptors
//
NTSTATUS
Descriptors_GetCompleteConfigDescriptor(_Out_writes_bytes_(*Length)
                                            PVOID Buffer,
                                        _Inout_ PULONG Length) {
  COMPLETE_CONFIG_DESCRIPTOR descriptor;
  ULONG requiredLength = sizeof(COMPLETE_CONFIG_DESCRIPTOR);

  PAGED_CODE();

  if (*Length < requiredLength) {
    *Length = requiredLength;
    return STATUS_BUFFER_TOO_SMALL;
  }

  RtlZeroMemory(&descriptor, sizeof(descriptor));

  //
  // Configuration descriptor
  //
  descriptor.ConfigDescriptor.bLength = sizeof(USB_CONFIGURATION_DESCRIPTOR);
  descriptor.ConfigDescriptor.bDescriptorType =
      USB_CONFIGURATION_DESCRIPTOR_TYPE;
  descriptor.ConfigDescriptor.wTotalLength = (USHORT)requiredLength;
  descriptor.ConfigDescriptor.bNumInterfaces = 1;
  descriptor.ConfigDescriptor.bConfigurationValue = 1;
  descriptor.ConfigDescriptor.iConfiguration = 0;
  descriptor.ConfigDescriptor.bmAttributes = 0x80;  // Bus-powered
  descriptor.ConfigDescriptor.MaxPower = 50;        // 100mA

  //
  // Interface descriptor (HID interface)
  //
  descriptor.InterfaceDescriptor.bLength = sizeof(USB_INTERFACE_DESCRIPTOR);
  descriptor.InterfaceDescriptor.bDescriptorType =
      USB_INTERFACE_DESCRIPTOR_TYPE;
  descriptor.InterfaceDescriptor.bInterfaceNumber = 0;
  descriptor.InterfaceDescriptor.bAlternateSetting = 0;
  descriptor.InterfaceDescriptor.bNumEndpoints = 2;
  descriptor.InterfaceDescriptor.bInterfaceClass = 0x03;     // HID
  descriptor.InterfaceDescriptor.bInterfaceSubClass = 0x00;  // No subclass
  descriptor.InterfaceDescriptor.bInterfaceProtocol = 0x00;  // No protocol
  descriptor.InterfaceDescriptor.iInterface = 0;

  //
  // HID descriptor
  //
  descriptor.HidDescriptor.bLength = sizeof(HID_DESCRIPTOR);
  descriptor.HidDescriptor.bDescriptorType = 0x21;  // HID
  descriptor.HidDescriptor.bcdHID = 0x0111;         // HID 1.11
  descriptor.HidDescriptor.bCountryCode = 0;
  descriptor.HidDescriptor.bNumDescriptors = 1;
  descriptor.HidDescriptor.bReportDescriptorType = 0x22;  // Report
  descriptor.HidDescriptor.wReportDescriptorLength =
      sizeof(g_HidReportDescriptor);

  //
  // Interrupt IN endpoint (HID input reports)
  //
  descriptor.EndpointDescriptorIn.bLength = sizeof(USB_ENDPOINT_DESCRIPTOR);
  descriptor.EndpointDescriptorIn.bDescriptorType =
      USB_ENDPOINT_DESCRIPTOR_TYPE;
  descriptor.EndpointDescriptorIn.bEndpointAddress = 0x81;  // EP1 IN
  descriptor.EndpointDescriptorIn.bmAttributes = USB_ENDPOINT_TYPE_INTERRUPT;
  descriptor.EndpointDescriptorIn.wMaxPacketSize = CHUNITHM_INPUT_REPORT_SIZE;
  descriptor.EndpointDescriptorIn.bInterval = 1;  // 1ms polling

  //
  // Interrupt OUT endpoint (RGB LED data - not used in ChunIVision)
  //
  descriptor.EndpointDescriptorOut.bLength = sizeof(USB_ENDPOINT_DESCRIPTOR);
  descriptor.EndpointDescriptorOut.bDescriptorType =
      USB_ENDPOINT_DESCRIPTOR_TYPE;
  descriptor.EndpointDescriptorOut.bEndpointAddress = 0x02;  // EP2 OUT
  descriptor.EndpointDescriptorOut.bmAttributes = USB_ENDPOINT_TYPE_INTERRUPT;
  descriptor.EndpointDescriptorOut.wMaxPacketSize = CHUNITHM_OUTPUT_REPORT_SIZE;
  descriptor.EndpointDescriptorOut.bInterval = 1;  // 1ms polling

  //
  // Copy to output buffer
  //
  RtlCopyMemory(Buffer, &descriptor, requiredLength);
  *Length = requiredLength;

  return STATUS_SUCCESS;
}

//
// Get HID descriptor only
//
NTSTATUS
Descriptors_GetHidDescriptor(_Out_writes_bytes_(*Length) PVOID Buffer,
                             _Inout_ PULONG Length) {
  HID_DESCRIPTOR hidDescriptor;
  ULONG requiredLength = sizeof(HID_DESCRIPTOR);

  PAGED_CODE();

  if (*Length < requiredLength) {
    *Length = requiredLength;
    return STATUS_BUFFER_TOO_SMALL;
  }

  RtlZeroMemory(&hidDescriptor, sizeof(hidDescriptor));

  hidDescriptor.bLength = sizeof(HID_DESCRIPTOR);
  hidDescriptor.bDescriptorType = 0x21;  // HID
  hidDescriptor.bcdHID = 0x0111;         // HID 1.11
  hidDescriptor.bCountryCode = 0;
  hidDescriptor.bNumDescriptors = 1;
  hidDescriptor.bReportDescriptorType = 0x22;  // Report
  hidDescriptor.wReportDescriptorLength = sizeof(g_HidReportDescriptor);

  RtlCopyMemory(Buffer, &hidDescriptor, requiredLength);
  *Length = requiredLength;

  return STATUS_SUCCESS;
}

//
// Get HID report descriptor
//
NTSTATUS
Descriptors_GetHidReportDescriptor(_Out_writes_bytes_(*Length) PVOID Buffer,
                                   _Inout_ PULONG Length) {
  ULONG requiredLength = sizeof(g_HidReportDescriptor);

  PAGED_CODE();

  if (*Length < requiredLength) {
    *Length = requiredLength;
    return STATUS_BUFFER_TOO_SMALL;
  }

  RtlCopyMemory(Buffer, g_HidReportDescriptor, requiredLength);
  *Length = requiredLength;

  return STATUS_SUCCESS;
}
