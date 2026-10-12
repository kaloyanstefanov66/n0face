"""Virtual webcam output, so Discord, Teams, Zoom, OBS etc. can pick up the processed feed."""

import numpy as np


class VirtualCamUnavailable(RuntimeError):
    pass


class VirtualCam:
    """Publishes frames through pyvirtualcam.

    On Windows and macOS this uses the OBS Virtual Camera driver, which is installed with OBS
    Studio (OBS itself doesn't need to be running). On Linux it uses v4l2loopback.
    """

    def __init__(self, width: int, height: int, fps: int):
        import pyvirtualcam

        try:
            self._cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR)
        except RuntimeError as e:
            raise VirtualCamUnavailable(
                "No virtual camera driver found. Install OBS Studio (Windows/macOS) or "
                f"v4l2loopback (Linux), then try again. ({e})"
            ) from e
        self.device = self._cam.device

    def send(self, frame: np.ndarray) -> None:
        self._cam.send(frame)

    def close(self) -> None:
        self._cam.close()
