"""
Unit tests for ChunithmUDE Python bindings.

Author: Misaka 19465

Note: These tests require the UDE driver to be installed and running.
Run with administrator privileges.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.ude_device import ChunithmUDEDevice
from python.exceptions import (
    DeviceNotFoundError,
    DeviceNotReadyError,
    InvalidReportError,
    DriverCommunicationError
)
from python.constants import NUM_TOUCH_ZONES, NUM_AIR_SENSORS


class TestChunithmUDEDevice(unittest.TestCase):
    """Test cases for ChunithmUDEDevice class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.touch_zones = [False] * NUM_TOUCH_ZONES
        self.air_sensors = [False] * NUM_AIR_SENSORS
    
    @patch('python.ude_device.ctypes.windll.kernel32')
    def test_device_open_success(self, mock_kernel32):
        """Test successful device opening."""
        mock_kernel32.CreateFileW.return_value = 123  # Valid handle
        
        device = ChunithmUDEDevice()
        self.assertIsNotNone(device.handle)
        self.assertEqual(device.handle, 123)
        
        device.close()
    
    @patch('python.ude_device.ctypes.windll.kernel32')
    def test_device_open_failure(self, mock_kernel32):
        """Test device open failure."""
        mock_kernel32.CreateFileW.return_value = -1  # INVALID_HANDLE_VALUE
        
        with self.assertRaises(DeviceNotFoundError):
            device = ChunithmUDEDevice()
    
    @patch('python.ude_device.ctypes.windll.kernel32')
    def test_send_report_success(self, mock_kernel32):
        """Test successful report sending."""
        mock_kernel32.CreateFileW.return_value = 123
        mock_kernel32.DeviceIoControl.return_value = True
        
        device = ChunithmUDEDevice()
        
        # Set some zones
        self.touch_zones[0] = True
        self.touch_zones[15] = True
        self.air_sensors[2] = True
        
        result = device.send_report(self.touch_zones, self.air_sensors)
        self.assertTrue(result)
        
        device.close()
    
    def test_send_report_invalid_touch_zones(self):
        """Test report with invalid number of touch zones."""
        with patch('python.ude_device.ctypes.windll.kernel32') as mock_kernel32:
            mock_kernel32.CreateFileW.return_value = 123
            
            device = ChunithmUDEDevice()
            
            # Wrong number of zones
            with self.assertRaises(InvalidReportError):
                device.send_report([False] * 10, self.air_sensors)
            
            device.close()
    
    def test_send_report_invalid_air_sensors(self):
        """Test report with invalid number of air sensors."""
        with patch('python.ude_device.ctypes.windll.kernel32') as mock_kernel32:
            mock_kernel32.CreateFileW.return_value = 123
            
            device = ChunithmUDEDevice()
            
            # Wrong number of sensors
            with self.assertRaises(InvalidReportError):
                device.send_report(self.touch_zones, [False] * 3)
            
            device.close()
    
    def test_send_report_device_not_ready(self):
        """Test sending report when device not opened."""
        device = ChunithmUDEDevice.__new__(ChunithmUDEDevice)
        device.handle = None
        
        with self.assertRaises(DeviceNotReadyError):
            device.send_report(self.touch_zones, self.air_sensors)
    
    @patch('python.ude_device.ctypes.windll.kernel32')
    def test_context_manager(self, mock_kernel32):
        """Test context manager usage."""
        mock_kernel32.CreateFileW.return_value = 123
        mock_kernel32.DeviceIoControl.return_value = True
        
        with ChunithmUDEDevice() as device:
            result = device.send_report(self.touch_zones, self.air_sensors)
            self.assertTrue(result)
        
        # Verify close was called
        mock_kernel32.CloseHandle.assert_called_once_with(123)
    
    @patch('python.ude_device.ctypes.windll.kernel32')
    def test_build_report_format(self, mock_kernel32):
        """Test HID report format correctness."""
        mock_kernel32.CreateFileW.return_value = 123
        
        device = ChunithmUDEDevice()
        
        # Set specific values
        self.touch_zones[0] = True   # Zone 1
        self.touch_zones[31] = True  # Zone 32
        self.air_sensors[0] = True   # Air 0
        self.air_sensors[5] = True   # Air 5
        
        report = device._build_report(self.touch_zones, self.air_sensors)
        
        # Verify report size
        self.assertEqual(len(report), 45)
        
        # Verify IR value (bits 0 and 5 set)
        self.assertEqual(report[0], 0b00100001)
        
        # Verify buttons
        self.assertEqual(report[1], 0x00)
        
        # Verify touch values
        self.assertEqual(report[2], 0x64)   # Zone 1 pressed
        self.assertEqual(report[33], 0x64)  # Zone 32 pressed
        self.assertEqual(report[3], 0x00)   # Zone 2 released
        
        # Verify card status
        self.assertEqual(report[34], 0x00)
        
        device.close()


class TestConstants(unittest.TestCase):
    """Test cases for constants module."""
    
    def test_touch_zone_count(self):
        """Test touch zone constant."""
        self.assertEqual(NUM_TOUCH_ZONES, 32)
    
    def test_air_sensor_count(self):
        """Test air sensor constant."""
        self.assertEqual(NUM_AIR_SENSORS, 6)


if __name__ == '__main__':
    unittest.main()
