from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Sighting:
    """A single OCR reading of a jersey number."""

    frame_id: int
    number: str
    confidence: float


@dataclass
class ResolvedTrack:
    """The voted result for a completed track."""

    track_id: int
    number: str
    confidence: float
    sighting_count: int
    first_frame: int
    last_frame: int


class NumberTracker:
    """Accumulates jersey number OCR sightings by ByteTrack ID and resolves
    them to final jersey numbers via confidence-weighted majority voting.

    Usage::

        tracker = NumberTracker()

        # Feed OCR results frame by frame
        tracker.update(frame_id=0, sightings=[
            (track_id_1, "23", 0.92),
            (track_id_2, "7", 0.85),
        ])
        # ... more frames ...

        # After processing, resolve remaining tracks and read results
        tracker.resolve_all()
        print(tracker.numbers_on_court())  # e.g. {"23": 0.92, "7": 0.85}
    """

    def __init__(self, stale_thresh: int = 30, min_sightings: int = 1):
        """
        Args:
            stale_thresh: Frames since last sighting before a track is
                considered complete and auto-resolved.
            min_sightings: Minimum sightings required to emit a result.
        """
        self._tracks: dict[int, list[Sighting]] = defaultdict(list)
        self._last_seen: dict[int, int] = {}
        self._resolved: dict[int, ResolvedTrack] = {}
        self.stale_thresh = stale_thresh
        self.min_sightings = min_sightings

    def update(self, frame_id: int, sightings: list[tuple[int, str, float]]) -> None:
        """Record OCR results for a single frame.

        Args:
            frame_id: Current frame number.
            sightings: List of ``(track_id, predicted_number, confidence)`` tuples
                produced by running the OCR model on each detected crop.
        """
        for track_id, number, confidence in sightings:
            self._tracks[track_id].append(Sighting(frame_id, number, confidence))
            self._last_seen[track_id] = frame_id
        self._resolve_stale(frame_id)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_all(self) -> None:
        """Force-resolve every unresolved track (call at end of video)."""
        for tid in list(self._tracks):
            if tid not in self._resolved:
                self._try_resolve(tid)

    def _resolve_stale(self, current_frame: int) -> None:
        for tid in list(self._last_seen):
            if tid in self._resolved:
                continue
            if current_frame - self._last_seen[tid] >= self.stale_thresh:
                self._try_resolve(tid)

    def _try_resolve(self, track_id: int) -> None:
        sightings = self._tracks[track_id]
        if len(sightings) < self.min_sightings:
            return
        number, confidence = self._vote(sightings)
        if not number:
            return
        self._resolved[track_id] = ResolvedTrack(
            track_id=track_id,
            number=number,
            confidence=confidence,
            sighting_count=len(sightings),
            first_frame=sightings[0].frame_id,
            last_frame=sightings[-1].frame_id,
        )

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    @staticmethod
    def _vote(sightings: list[Sighting]) -> tuple[str, float]:
        """Confidence-weighted majority vote.

        Each sighting casts a vote for its predicted number, weighted by the
        OCR confidence score.  The number with the highest total weight wins.

        Returns:
            ``(winning_number, mean_confidence_for_winner)``
        """
        weighted: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for s in sightings:
            if not s.number:
                continue
            weighted[s.number] += s.confidence
            counts[s.number] += 1
        if not weighted:
            return "", 0.0
        best = max(weighted, key=weighted.__getitem__)
        return best, weighted[best] / counts[best]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def numbers_on_court(self) -> dict[str, float]:
        """Return the set of resolved jersey numbers and their confidences.

        If multiple tracks resolved to the same number, the highest
        confidence is kept.

        Call :meth:`resolve_all` first to include tracks that are still active.

        Returns:
            Dict mapping jersey number string to its best confidence score.
        """
        result: dict[str, float] = {}
        for rt in self._resolved.values():
            if rt.number not in result or rt.confidence > result[rt.number]:
                result[rt.number] = rt.confidence
        return result

    def resolved_tracks(self) -> list[ResolvedTrack]:
        """Return all resolved tracks for inspection / debugging."""
        return list(self._resolved.values())
