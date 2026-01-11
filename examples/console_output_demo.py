"""
Demo script for ConsoleOutput.

Shows a live display of simulated touch and height sensor data.
Press Ctrl+C to exit.
"""

import time
import numpy as np

from chunivision.output.console_output import ConsoleOutput
from chunivision.vision.touch_detector import TouchState
from chunivision.vision.height_estimator import HeightState
from chunivision.utils.logger import Logger


def main():
    """Run console output demo."""
    # Configure logging to file only to avoid interfering with console display
    Logger.setup(level="INFO", log_file="console_demo.log")

    print("ChunIVision Console Output Demo")
    print("Press Ctrl+C to exit")
    print("(Logs are written to console_demo.log)\n")
    time.sleep(2)

    # Create console output with colors enabled
    config = {
        "update_interval": 0.05,  # 20 FPS
        "show_stats": True,
        "color_enabled": True,
    }

    with ConsoleOutput(config, "Demo") as output:
        try:
            frame = 0
            while True:
                # Simulate touch zones - create a sweeping pattern
                zones = np.zeros(32, dtype=bool)
                active_zones = [(frame + i) % 32 for i in range(3)]
                zones[active_zones] = True
                touch_state = TouchState(zones=zones)

                # Simulate height levels - create a wave pattern
                levels = np.zeros(6, dtype=bool)
                active_level = int((np.sin(frame * 0.1) + 1) * 2.5)
                if 0 <= active_level < 6:
                    levels[active_level] = True
                height_state = HeightState(levels=levels)

                # Send state to display
                output.send_state(touch_state, height_state)

                frame += 1
                time.sleep(0.05)  # 20 FPS

        except KeyboardInterrupt:
            print("\n\nDemo stopped by user")


if __name__ == "__main__":
    main()
