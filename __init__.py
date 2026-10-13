from jersey_tracker.tracker import (
    NumberTracker,
    PlayerTracker,
    ResolvedTrack,
    Sighting,
    TeamSighting,
)
from jersey_tracker.ocr import JerseyOCR


def __getattr__(name: str):
    # Loaded on first use so `import jersey_tracker` stays light (the inferencer needs torch + PIL).
    if name == "JerseyOCRInferencer":
        from jersey_tracker.inferencer import JerseyOCRInferencer

        return JerseyOCRInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "JerseyOCR",
    "JerseyOCRInferencer",
    "PlayerTracker",
    "NumberTracker",
    "ResolvedTrack",
    "Sighting",
    "TeamSighting",
]
