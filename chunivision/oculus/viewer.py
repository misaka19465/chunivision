"""Simple OpenCV viewer for the Oculus Rift CV1 camera."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .oculus_camera import OculusRiftCV1Camera
    from .triple_buffer import TripleBuffer  # type: ignore
except ImportError:
    from oculus_camera import OculusRiftCV1Camera
    from triple_buffer import TripleBuffer


class DistortionCalibration:
    """Lens distortion calibration for Oculus Rift CV1 camera.

    This class handles conversion between distorted (raw camera) and undistorted
    (corrected) pixel coordinates using the camera's calibration parameters.
    """

    def __init__(
        self,
        frame_size: tuple[int, int] = (1280, 960),
        center: tuple[float, float] = (655.052, 475.083),
        kappas: tuple[float, float, float] = (5.16403e-07, 2.44492e-13, 6.881e-19),
        rhos: tuple[float, float] = (-8.66716e-07, 8.37108e-07),
        max_r2: float = 0.0,
    ) -> None:
        """Initialize distortion calibration.

        Args:
            frame_size: Camera frame size (width, height)
            center: Optical center (cx, cy)
            kappas: Radial distortion coefficients
            rhos: Tangential distortion coefficients
            max_r2: Maximum squared radius for valid undistortion
        """
        self.frame_size = frame_size
        self.center = center
        self.kappas = kappas
        self.rhos = rhos
        self.max_r2 = max_r2

    def can_undistort(self, pixel: tuple[float, float]) -> bool:
        """Check if a pixel can be undistorted.

        Args:
            pixel: (x, y) coordinate tuple

        Returns:
            True if pixel is within valid undistortion range
        """
        if pixel is None or len(pixel) != 2:
            return False
        dx = pixel[0] - self.center[0]
        dy = pixel[1] - self.center[1]
        return (dx * dx + dy * dy) < self.max_r2

    def undistort(self, pixel: tuple[float, float]) -> tuple[float, float]:
        """Convert a distorted pixel coordinate to undistorted coordinate.

        Args:
            pixel: (x, y) coordinate in distorted (raw camera) space

        Returns:
            (x, y) coordinate in undistorted (corrected) space
        """
        if pixel is None or len(pixel) != 2:
            raise ValueError("Pixel must be a tuple of (x, y) coordinates")

        dx = pixel[0] - self.center[0]
        dy = pixel[1] - self.center[1]
        r2 = dx * dx + dy * dy

        # Compute radial distortion
        radial = 0.0
        for kappa in reversed(self.kappas):
            radial = (radial + kappa) * r2
        radial += 1.0

        # Apply radial and tangential distortion
        return (
            self.center[0]
            + dx * radial
            + 2.0 * self.rhos[0] * dx * dy
            + self.rhos[1] * (r2 + 2.0 * dx * dx),
            self.center[1]
            + dy * radial
            + self.rhos[0] * (r2 + 2.0 * dy * dy)
            + 2.0 * self.rhos[1] * dx * dy,
        )

    def distort(
        self,
        pixel: tuple[float, float],
        max_iterations: int = 20,
        tolerance: float = 1e-6,
    ) -> tuple[float, float]:
        """Convert an undistorted pixel coordinate to distorted coordinate (inverse of undistort).

        This is needed for creating remap tables with OpenCV, where for each output (undistorted)
        pixel we need to find which input (distorted) pixel to sample from.

        Args:
            pixel: (x, y) coordinate in undistorted (corrected) space
            max_iterations: Maximum number of Newton-Raphson iterations (must be > 0)
            tolerance: Convergence tolerance in pixels (must be > 0)

        Returns:
            (x, y) coordinate in distorted (raw camera) space
        """
        if pixel is None or len(pixel) != 2:
            raise ValueError("Pixel must be a tuple of (x, y) coordinates")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"Tolerance must be > 0, got {tolerance}")

        # Use fixed-point iteration to find the distorted point
        # Start with the undistorted point as initial guess
        distorted_x, distorted_y = pixel

        for _ in range(max_iterations):
            # Compute undistorted position from current distorted guess
            undistorted_x, undistorted_y = self.undistort((distorted_x, distorted_y))

            # Compute error
            error_x = undistorted_x - pixel[0]
            error_y = undistorted_y - pixel[1]

            # Check for convergence
            error = math.sqrt(error_x * error_x + error_y * error_y)
            if error < tolerance:
                break

            # Update distorted position (simple fixed-point iteration)
            # This works because the distortion is small
            distorted_x -= error_x
            distorted_y -= error_y

        # Clamp to valid image bounds to prevent out-of-bounds sampling
        distorted_x = max(0.0, min(float(self.frame_size[0] - 1), distorted_x))
        distorted_y = max(0.0, min(float(self.frame_size[1] - 1), distorted_y))

        return (distorted_x, distorted_y)


def run_viewer(
    device_index: int, undistort: bool = False, compare: bool = False
) -> int:
    camera = OculusRiftCV1Camera(device_index)
    frames = TripleBuffer[bytes]()
    save_next = False
    save_index = 0

    # Get calibration parameters from camera and create distortion calibration instance
    cal_params = camera.get_calibration_params()
    calibration = DistortionCalibration(
        frame_size=cal_params["frame_size"],
        center=(cal_params["cx"], cal_params["cy"]),
        max_r2=cal_params["max_r2"],
    )

    def on_frame(frame_bytes: bytes) -> None:
        nonlocal save_next, save_index
        frames.publish(frame_bytes)
        if save_next:
            filename = Path(f"SavedVideoFrame{save_index:04d}.pgm")
            save_index += 1
            with filename.open("wb") as fp:
                fp.write(
                    f"P5\n{camera.get_frame_width()} {camera.get_frame_height()}\n255\n".encode(
                        "ascii"
                    )
                )
                fp.write(frame_bytes)
            print(f"Saved frame to {filename}")
            save_next = False

    camera.set_flip(False, True)
    camera.start_streaming(on_frame)
    print("Streaming started!")
    print(
        "Controls: s=save, +=gain up, -=gain down, a=toggle auto exposure, q/ESC=quit"
    )

    window_name = "Oculus Rift CV1 Camera"
    window_name_undist = "Oculus Rift CV1 Camera (Undistorted)"
    window_name_dist = "Oculus Rift CV1 Camera (Distorted)"
    # Create windows depending on mode
    if compare:
        cv2.namedWindow(window_name_dist, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow(window_name_undist, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    else:
        # Single window will show either original or undistorted based on `undistort`
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    try:
        map_x = None
        map_y = None
        if undistort:
            # Precompute remap table for OpenCV's cv2.remap()
            # For each OUTPUT (undistorted) pixel, find which INPUT (distorted) pixel to sample from
            print("Computing undistortion map...")
            h, w = camera.get_frame_height(), camera.get_frame_width()
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)

            # Track statistics for debugging
            out_of_bounds = 0
            total_pixels = h * w

            for yy in range(h):
                for xx in range(w):
                    # For this output (undistorted) pixel, find the input (distorted) pixel
                    # Use distort() to go from undistorted -> distorted space
                    dx, dy = calibration.distort((float(xx), float(yy)))
                    map_x[yy, xx] = dx
                    map_y[yy, xx] = dy

                    # Track out-of-bounds mappings
                    if dx < 0 or dx >= w or dy < 0 or dy >= h:
                        out_of_bounds += 1

            print(
                f"Undistortion map computed. Out-of-bounds pixels: {out_of_bounds}/{total_pixels} ({100.0*out_of_bounds/total_pixels:.2f}%)"
            )

        while True:
            # Process USB events
            camera.handle_events()

            frame = frames.get_latest()
            if frame:
                mat = frame_to_mat(frame, camera)
                undistorted_mat = None
                if undistort and map_x is not None and map_y is not None:
                    # Use BORDER_REPLICATE to avoid black borders at edges
                    undistorted_mat = cv2.remap(
                        mat,
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

                if compare:
                    # Show both original (distorted) and undistorted for comparison
                    cv2.imshow(window_name_dist, mat)
                    # If undistortion wasn't requested/available, show the same image in the undistorted window
                    cv2.imshow(
                        window_name_undist,
                        undistorted_mat if undistorted_mat is not None else mat,
                    )
                else:
                    # Single window mode: prefer showing undistorted when requested
                    display_mat = (
                        undistorted_mat
                        if (undistorted_mat is not None and undistort)
                        else mat
                    )
                    cv2.imshow(window_name, display_mat)
            key = cv2.waitKey(10)
            if key in (ord("q"), 27):
                break
            if key in (ord("s"),):
                save_next = True
            elif key in (ord("+"), ord("=")):
                gain = min(camera.get_gain() + 16, 255)
                camera.set_gain(gain)
                print(f"Gain: {gain}")
            elif key in (ord("-"), ord("_")):
                gain = camera.get_gain()
                gain = gain - 16 if gain > 16 else 0
                camera.set_gain(gain)
                print(f"Gain: {gain}")
            elif key in (ord("a"), ord("A")):
                auto_exp = camera.get_auto_exposure()
                camera.set_auto_exposure(not auto_exp, True, True)
                print(f"Auto exposure: {'enabled' if not auto_exp else 'disabled'}")
    finally:
        cv2.destroyAllWindows()
        camera.stop_streaming()
    return 0


def frame_to_mat(frame: bytes, camera: OculusRiftCV1Camera) -> "cv2.Mat":
    height, width = camera.get_frame_height(), camera.get_frame_width()
    return np.frombuffer(frame, dtype=np.uint8).reshape((height, width))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Oculus Rift CV1 camera viewer (Python)"
    )
    parser.add_argument(
        "device_index",
        nargs="?",
        type=int,
        default=0,
        help="Camera index if multiple devices are present",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Enable lens distortion correction using camera calibration",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Open two windows: distorted and undistorted for side-by-side comparison",
    )
    args = parser.parse_args(argv)
    return run_viewer(args.device_index, undistort=args.undistort, compare=args.compare)


if __name__ == "__main__":
    sys.exit(main())
