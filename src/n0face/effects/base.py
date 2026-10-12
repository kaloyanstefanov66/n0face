"""Effect base class.

Privacy invariant: `render` never receives the camera frame. Whatever an effect draws inside
the mask is generated from landmark geometry and time only, so no face pixels can survive.
"""

import cv2
import numpy as np

FEATHER_PX = 9


class Effect:
    name = "base"
    # True only for effects that derive output from face pixels (e.g. pixelate). Those are
    # weaker by construction and are excluded from the input-independence guarantee.
    uses_camera_pixels = False
    # Mask-mode effects that can be drawn as a 3D-wrapped texture (see n0face.wrap). When the
    # wrap option is on, or the effect only exists as a texture, apply() draws a dark panel over
    # the mask and then the texture on each face mesh.
    wrap_design: str | None = None
    always_textured = False
    theme = None

    def render(self, h: int, w: int, faces: list[np.ndarray], t: float, rng: np.random.Generator) -> np.ndarray:
        """Return an (h, w, 3) BGR uint8 image. `faces` are in this region's coordinates."""
        raise NotImplementedError

    def apply(self, frame: np.ndarray, mask: np.ndarray, faces: list[np.ndarray], t: float) -> np.ndarray:
        if self.wrap_design and faces:
            from n0face import wrap

            if self.always_textured or wrap.config.enabled:
                return self._apply_textured(frame, mask, faces, t, wrap.config.enabled)
        return self._apply_fill(frame, mask, faces, t)

    def _apply_textured(self, frame, mask, faces, t, wrapped: bool) -> np.ndarray:
        from n0face import wrap

        # Flat stickers sit on a dark panel covering the whole mask region. A 3D-wrapped mask
        # is drawn on its own so it hugs the face; its texture is fully opaque, so no face
        # pixels show through wherever the mask covers.
        out = frame.copy() if wrapped else PANEL.apply(frame, mask, [], t)
        for lm in faces:
            wrap.draw_face(out, lm, self.wrap_design, self.theme, t, wrapped)
        return out

    def _apply_fill(self, frame: np.ndarray, mask: np.ndarray, faces: list[np.ndarray], t: float) -> np.ndarray:
        # Deterministic per frame, so output depends only on (mask, faces, t).
        rng = np.random.default_rng(int(t * 30))
        fh, fw = mask.shape
        bx, by, bw, bh = cv2.boundingRect(mask)
        if bw == 0:
            return frame.copy()
        if bw == fw and bh == fh and cv2.countNonZero(mask) == fw * fh:
            # Whole frame hidden (fail-closed): nothing of the camera frame is kept.
            return self.render(fh, fw, faces, t, rng)

        out = frame.copy()
        x0, y0 = max(bx - FEATHER_PX, 0), max(by - FEATHER_PX, 0)
        x1, y1 = min(bx + bw + FEATHER_PX, fw), min(by + bh + FEATHER_PX, fh)
        offset = np.array([x0, y0], dtype=np.float32)
        fill = self.render(y1 - y0, x1 - x0, [f - offset for f in faces], t, rng)

        # Soft edge outside the mask for looks; the mask interior is always fully opaque.
        core = mask[y0:y1, x0:x1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FEATHER_PX,) * 2)
        alpha = cv2.GaussianBlur(cv2.dilate(core, kernel), (0, 0), FEATHER_PX / 2)
        alpha = np.maximum(alpha, core).astype(np.float32) / 255
        roi = cv2.blendLinear(fill, out[y0:y1, x0:x1], alpha, 1 - alpha)
        cv2.copyTo(fill, core, roi)  # exact fill inside the mask, no rounding at alpha == 1
        out[y0:y1, x0:x1] = roi
        return out


def add_glow(base: np.ndarray, layer: np.ndarray, radius: float = 4.0, strength: float = 1.6) -> np.ndarray:
    """Composite a bright line/dot layer over `base` with a bloom halo."""
    halo = cv2.GaussianBlur(layer, (0, 0), radius)
    return cv2.addWeighted(cv2.add(base, layer), 1.0, halo, strength, 0)


def scanlines(img: np.ndarray, t: float, darkness: float = 0.35) -> np.ndarray:
    """Darken every other row, with a slow rolling bright band. Modifies and returns `img`."""
    h = img.shape[0]
    phase = int(t * 60) % 2
    img[phase::2] = cv2.convertScaleAbs(img[phase::2], alpha=1 - darkness)
    y = int((t * 120) % (h + 60)) - 30
    band = slice(max(y, 0), max(min(y + 30, h), 0))
    if band.stop > band.start:
        img[band] = cv2.convertScaleAbs(img[band], alpha=1.25)
    return img


class Panel(Effect):
    """Dark panel with faint scanlines: the base under textured masks."""

    name = "panel"

    def render(self, h, w, faces, t, rng):
        return scanlines(np.full((h, w, 3), (14, 12, 10), np.uint8), t, darkness=0.3)


PANEL = Panel()
