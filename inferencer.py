from __future__ import annotations

import numpy as np
from PIL import Image
from jersey_tracker.ocr import JerseyOCR
import torch


class JerseyOCRInferencer:
    def __init__(self, model_source: str = "pretrained=parseq", device: str = "cuda") -> None:
        self.device = device
        self.reader = JerseyOCR(checkpoint=model_source, device=device)

    @torch.inference_mode()
    def infer(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> list[tuple[str, float]]:
        if len(bboxes) == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        frame_rgb = frame[:, :, ::-1]
        images: list[Image.Image] = []
        valid_idx: list[int] = []
        for i, xyxy in enumerate(bboxes):
            x1 = max(0, int(np.floor(xyxy[0])))
            y1 = max(0, int(np.floor(xyxy[1])))
            x2 = min(frame_w, int(np.ceil(xyxy[2])))
            y2 = min(frame_h, int(np.ceil(xyxy[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_rgb[y1:y2, x1:x2]
            pil = Image.fromarray(np.ascontiguousarray(crop))
            images.append(pil)
            valid_idx.append(i)

        results: list[tuple[str, float]] = [("", 0.0)] * len(bboxes)
        if not images:
            return results

        predictions = self.reader.read_batch(images)
        for valid_i, (pred, confidence) in zip(valid_idx, predictions):
            pred_str = "".join(ch for ch in pred if ch.isdigit())
            if not pred_str:
                continue
            results[valid_i] = (pred_str, confidence)
        return results


__all__ = ["JerseyOCRInferencer"]
