# Python rewrite

This `py/` directory contains a Python version of the standalone Oculus Rift CV1 camera driver and viewer. It mirrors the logic in `src/OculusRiftCV1Camera.cpp` but uses the `usb1` Python bindings and OpenCV for display.

## Dependencies

- Python 3.9+
- libusb-1.0 development library installed on the system
- Python packages: `libusb1`, `opencv-python`, `numpy`

Install packages into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r py/requirements.txt
```

## Running the viewer

```bash
python -m py.viewer 0
```

Controls match the C++ viewer: `s` saves a frame, `+`/`-` adjust gain, `a` toggles auto exposure, `q`/`ESC` exit. Frames are saved as `SavedVideoFrameXXXX.pgm` in the current directory.

## Notes

- You must have the Rift CV1 camera attached; the driver searches for vendor `0x2833` product `0x0211`.
- The code expects isochronous transfers (endpoint `0x81`, alt setting `2`).
- Run as root or with appropriate udev permissions to access the device.
