from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class JerseyOCR:
    """Lazy wrapper around the packaged PARSeq OCR implementation.

    This keeps `jersey_tracker` importable even when OCR-only dependencies are
    not installed. The actual OCR stack is loaded only when an instance is
    created.
    """

    def __init__(self, checkpoint: str = "pretrained=parseq", device: str = "cuda"):
        from strhub.jersey_ocr import JerseyOCR as _JerseyOCR

        self._impl = _JerseyOCR(checkpoint=checkpoint, device=device)

    def read(self, image: "Image.Image") -> tuple[str, float]:
        return self._impl.read(image)

    def read_batch(self, images: list["Image.Image"]) -> list[tuple[str, float]]:
        return self._impl.read_batch(images)


__all__ = ["JerseyOCR"]
