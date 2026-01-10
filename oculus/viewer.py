"""Simple OpenCV viewer for the Oculus Rift CV1 camera (Python rewrite)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .camera import OculusRiftCV1Camera  # type: ignore
    from .triple_buffer import TripleBuffer  # type: ignore
except ImportError:
    from camera import OculusRiftCV1Camera
    from triple_buffer import TripleBuffer


def run_viewer(device_index: int, undistort: bool = False, compare: bool = False) -> int:
    camera = OculusRiftCV1Camera(device_index)
    frames = TripleBuffer[bytes]()
    save_next = False
    save_index = 0

    def on_frame(frame_bytes: bytes) -> None:
        nonlocal save_next, save_index
        frames.publish(frame_bytes)
        if save_next:
            filename = Path(f"SavedVideoFrame{save_index:04d}.pgm")
            save_index += 1
            with filename.open("wb") as fp:
                fp.write(f"P5\n{camera.get_frame_width()} {camera.get_frame_height()}\n255\n".encode("ascii"))
                fp.write(frame_bytes)
            print(f"Saved frame to {filename}")
            save_next = False

    camera.set_flip(False, True)
    camera.start_streaming(on_frame)
    print("Streaming started!")
    print("Controls: s=save, +=gain up, -=gain down, a=toggle auto exposure, q/ESC=quit")

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
                    dx, dy = camera.distort((float(xx), float(yy)))
                    map_x[yy, xx] = dx
                    map_y[yy, xx] = dy
                    
                    # Track out-of-bounds mappings
                    if dx < 0 or dx >= w or dy < 0 or dy >= h:
                        out_of_bounds += 1
            
            print(f"Undistortion map computed. Out-of-bounds pixels: {out_of_bounds}/{total_pixels} ({100.0*out_of_bounds/total_pixels:.2f}%)")


        while True:
            # Process USB events
            camera.handle_events()
            
            frame = frames.get_latest()
            if frame:
                mat = frame_to_mat(frame, camera)
                undistorted_mat = None
                if undistort and map_x is not None and map_y is not None:
                    # Use BORDER_REPLICATE to avoid black borders at edges
                    undistorted_mat = cv2.remap(mat, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                if compare:
                    # Show both original (distorted) and undistorted for comparison
                    cv2.imshow(window_name_dist, mat)
                    # If undistortion wasn't requested/available, show the same image in the undistorted window
                    cv2.imshow(window_name_undist, undistorted_mat if undistorted_mat is not None else mat)
                else:
                    # Single window mode: prefer showing undistorted when requested
                    display_mat = undistorted_mat if (undistorted_mat is not None and undistort) else mat
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
    parser = argparse.ArgumentParser(description="Oculus Rift CV1 camera viewer (Python)")
    parser.add_argument("device_index", nargs="?", type=int, default=0, help="Camera index if multiple devices are present")
    parser.add_argument("--undistort", action="store_true", help="Enable lens distortion correction using camera calibration")
    parser.add_argument("--compare", action="store_true", help="Open two windows: distorted and undistorted for side-by-side comparison")
    args = parser.parse_args(argv)
    return run_viewer(args.device_index, undistort=args.undistort, compare=args.compare)


if __name__ == "__main__":
    sys.exit(main())
