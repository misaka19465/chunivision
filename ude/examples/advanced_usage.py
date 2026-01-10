"""
Advanced UDE device usage examples.
Demonstrates real-world integration patterns for ChunIVision.
"""

import time
import threading
from typing import List, Optional
from ude.python import ChunithmUDEDevice, DeviceNotFoundError


class UDEController:
    """Thread-safe UDE controller for continuous state updates."""

    def __init__(self):
        self.device: Optional[ChunithmUDEDevice] = None
        self.running = False
        self.lock = threading.Lock()
        self.current_touch_zones = [False] * 32
        self.current_air_sensors = [0] * 6

    def start(self) -> bool:
        """Initialize and start the UDE device."""
        try:
            self.device = ChunithmUDEDevice()
            self.device.open()
            self.running = True
            print("UDE device initialized successfully")
            return True
        except DeviceNotFoundError:
            print("Error: UDE driver not installed or device not ready")
            print("Please install the driver using: ude\\build\\install_driver.cmd")
            return False
        except Exception as e:
            print(f"Failed to initialize UDE device: {e}")
            return False

    def stop(self):
        """Stop and cleanup the UDE device."""
        self.running = False
        with self.lock:
            if self.device:
                self.device.close()
                self.device = None
        print("UDE device stopped")

    def update_state(self, touch_zones: List[bool], air_sensors: List[int]):
        """
        Update controller state.
        
        Args:
            touch_zones: List of 32 booleans (zone 1-32)
            air_sensors: List of 6 integers 0-255 (air sensor 1-6)
        """
        if not self.running or not self.device:
            return

        with self.lock:
            self.current_touch_zones = touch_zones.copy()
            self.current_air_sensors = air_sensors.copy()

        try:
            self.device.send_report(touch_zones, air_sensors)
        except Exception as e:
            print(f"Failed to send report: {e}")

    def get_status(self) -> dict:
        """Get current device status."""
        if not self.device:
            return {"connected": False}

        try:
            status = self.device.get_status()
            return {
                "connected": True,
                "device_ready": status.get("device_ready", False),
                "reports_sent": status.get("reports_sent", 0),
                "error_count": status.get("error_count", 0),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


class ChunIVisionIntegration:
    """Example integration with main ChunIVision pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.ude_controller = UDEController()
        self.enabled = config.get("output", {}).get("hid", {}).get("enabled", False)

    def initialize(self) -> bool:
        """Initialize output adapters."""
        if not self.enabled:
            print("HID/UDE output disabled in config")
            return True

        return self.ude_controller.start()

    def send_controller_state(
        self, touch_zones: List[bool], air_sensors: List[int]
    ):
        """
        Send controller state from vision processing.
        
        Called by vision pipeline for each frame.
        """
        if self.enabled:
            self.ude_controller.update_state(touch_zones, air_sensors)

    def shutdown(self):
        """Cleanup on exit."""
        if self.enabled:
            self.ude_controller.stop()


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic UDE Usage ===\n")

    # Initialize device
    device = ChunithmUDEDevice()
    try:
        device.open()
        print("Device opened successfully\n")
    except DeviceNotFoundError:
        print("Device not found - install driver first")
        return

    # Send initial state (all released)
    touch_zones = [False] * 32
    air_sensors = [0] * 6
    device.send_report(touch_zones, air_sensors)
    print("Sent initial state (all released)")

    # Simulate zone 1 touch
    time.sleep(0.1)
    touch_zones[0] = True  # Zone 1
    device.send_report(touch_zones, air_sensors)
    print("Zone 1 touched")

    # Simulate air sensor 1 activation
    time.sleep(0.1)
    air_sensors[0] = 150  # Height level
    device.send_report(touch_zones, air_sensors)
    print("Air sensor 1 activated (value: 150)")

    # Release all
    time.sleep(0.1)
    touch_zones = [False] * 32
    air_sensors = [0] * 6
    device.send_report(touch_zones, air_sensors)
    print("All released")

    # Get status
    status = device.get_status()
    print(f"\nDevice Status: {status}")

    # Cleanup
    device.close()
    print("\nDevice closed")


def example_continuous_updates():
    """Example of continuous state updates in a loop."""
    print("=== Continuous Updates Example ===\n")

    controller = UDEController()
    if not controller.start():
        return

    print("Sending continuous updates for 5 seconds...")
    print("Simulating sliding touch across zones\n")

    try:
        for i in range(50):  # 5 seconds at 100ms intervals
            touch_zones = [False] * 32
            air_sensors = [0] * 6

            # Simulate sliding touch
            zone_index = (i % 32)
            touch_zones[zone_index] = True

            # Simulate air sensor height
            air_value = int(128 + 64 * (i % 10) / 10)
            air_sensors[0] = air_value

            controller.update_state(touch_zones, air_sensors)

            if i % 10 == 0:
                status = controller.get_status()
                print(
                    f"Zone {zone_index + 1}, Air: {air_value}, "
                    f"Reports sent: {status.get('reports_sent', 0)}"
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        controller.stop()


def example_context_manager():
    """Example using context manager."""
    print("=== Context Manager Example ===\n")

    try:
        with ChunithmUDEDevice() as device:
            print("Device opened via context manager")

            # Send test pattern
            for zone in range(32):
                touch_zones = [False] * 32
                touch_zones[zone] = True
                device.send_report(touch_zones, [0] * 6)
                print(f"Activated zone {zone + 1}", end="\r")
                time.sleep(0.05)

            print("\nTest pattern completed")

    except DeviceNotFoundError:
        print("Device not found")
    # Device automatically closed


if __name__ == "__main__":
    print("ChunithmUDE Python Examples\n")
    print("=" * 50)
    print()

    # Run examples
    example_basic_usage()
    print("\n" + "=" * 50 + "\n")

    example_context_manager()
    print("\n" + "=" * 50 + "\n")

    # Uncomment to run continuous example
    # example_continuous_updates()
