from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, TypeVar


@dataclass
class Sighting:
    """A single OCR reading of a jersey number."""

    frame_id: int
    number: str
    confidence: float


@dataclass
class TeamSighting:
    """A single team-classifier prediction for a tracked player."""

    frame_id: int
    team: int
    confidence: float


@dataclass
class ResolvedTrack:
    """The voted result for a completed track.

    A resolved track may carry a jersey number, a team label, or both,
    depending on which sighting streams were fed in during the video.
    """

    track_id: int
    number: str
    confidence: float
    sighting_count: int
    first_frame: int
    last_frame: int
    team: int | None = None
    team_confidence: float | None = None
    team_sighting_count: int = 0


T = TypeVar("T")


class PlayerTracker:
    """Accumulates per-frame predictions for tracked players (keyed by ByteTrack ID)
    and resolves them via confidence-weighted majority voting.

    Two prediction streams are supported:

    * **Jersey numbers** — fed via :meth:`update_numbers` (or the legacy
      :meth:`update` alias) from an OCR model.
    * **Team labels** — fed via :meth:`update_teams` from a team classifier.

    Either stream alone is fine; they share track IDs and stale-tracking.

    Usage::

        tracker = PlayerTracker()

        tracker.update_numbers(frame_id=0, sightings=[(track_id, "23", 0.92)])
        tracker.update_teams(frame_id=0, sightings=[(track_id, 1, 0.8)])

        tracker.resolve_all()
        print(tracker.numbers_on_court())  # {"23": 0.92}
        print(tracker.teams_by_track())    # {track_id: 1}
    """

    def __init__(self, stale_thresh: int = 30, min_sightings: int = 1):
        """
        Args:
            stale_thresh: Frames since last sighting before a track is
                considered complete and auto-resolved.
            min_sightings: Minimum total sightings (number + team) required
                before a track is resolved.
        """
        self._tracks: dict[int, list[Sighting]] = defaultdict(list)
        self._team_tracks: dict[int, list[TeamSighting]] = defaultdict(list)
        self._last_seen: dict[int, int] = {}
        self._resolved: dict[int, ResolvedTrack] = {}
        self.stale_thresh = stale_thresh
        self.min_sightings = min_sightings

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_numbers(
        self, frame_id: int, sightings: list[tuple[int, str, float]]
    ) -> None:
        """Record jersey-number OCR results for a single frame.

        Args:
            frame_id: Current frame number.
            sightings: List of ``(track_id, predicted_number, confidence)`` tuples
                produced by running the OCR model on each detected crop.
        """
        for track_id, number, confidence in sightings:
            self._tracks[track_id].append(Sighting(frame_id, number, confidence))
            self._last_seen[track_id] = frame_id
        self._resolve_stale(frame_id)

    def update_teams(
        self, frame_id: int, sightings: list[tuple[int, int, float]]
    ) -> None:
        """Record team-classifier predictions for a single frame.

        Args:
            frame_id: Current frame number.
            sightings: List of ``(track_id, team_label, confidence)`` tuples
                produced by running the team classifier on each detected crop.
        """
        for track_id, team, confidence in sightings:
            self._team_tracks[track_id].append(
                TeamSighting(frame_id, int(team), confidence)
            )
            self._last_seen[track_id] = frame_id
        self._resolve_stale(frame_id)

    def update(self, frame_id: int, sightings: list[tuple[int, str, float]]) -> None:
        """Backwards-compatible alias for :meth:`update_numbers`."""
        self.update_numbers(frame_id, sightings)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_all(self) -> None:
        """Force-resolve every unresolved track (call at end of video)."""
        all_ids = set(self._tracks) | set(self._team_tracks)
        for tid in all_ids:
            if tid not in self._resolved:
                self._try_resolve(tid)

    def _resolve_stale(self, current_frame: int) -> None:
        for tid in list(self._last_seen):
            if tid in self._resolved:
                continue
            if current_frame - self._last_seen[tid] >= self.stale_thresh:
                self._try_resolve(tid)

    def _try_resolve(self, track_id: int) -> None:
        number_sightings = self._tracks.get(track_id, [])
        team_sightings = self._team_tracks.get(track_id, [])
        total = len(number_sightings) + len(team_sightings)
        if total < self.min_sightings:
            return

        number, number_conf = self._weighted_vote(
            number_sightings, lambda s: s.number, lambda s: s.confidence
        )
        team_label, team_conf = self._weighted_vote(
            team_sightings, lambda s: s.team, lambda s: s.confidence
        )

        if not number and team_label is None:
            return

        all_frames = [s.frame_id for s in number_sightings] + [
            s.frame_id for s in team_sightings
        ]
        self._resolved[track_id] = ResolvedTrack(
            track_id=track_id,
            number=number or "",
            confidence=number_conf,
            sighting_count=len(number_sightings),
            first_frame=min(all_frames),
            last_frame=max(all_frames),
            team=team_label if team_label is not None else None,
            team_confidence=team_conf if team_label is not None else None,
            team_sighting_count=len(team_sightings),
        )

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_vote(
        sightings: Iterable[T],
        get_label: Callable[[T], Hashable],
        get_conf: Callable[[T], float],
    ) -> tuple:
        """Confidence-weighted majority vote over a sighting stream.

        Each sighting casts a vote for its label, weighted by confidence.
        The label with the highest total weight wins.

        Returns:
            ``(winning_label, mean_confidence_for_winner)``. If no valid
            sightings exist, returns ``(None, 0.0)``.
        """
        weighted: dict = defaultdict(float)
        counts: dict = defaultdict(int)
        for s in sightings:
            label = get_label(s)
            if label is None or label == "":
                continue
            weighted[label] += get_conf(s)
            counts[label] += 1
        if not weighted:
            return None, 0.0
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
            if not rt.number:
                continue
            if rt.number not in result or rt.confidence > result[rt.number]:
                result[rt.number] = rt.confidence
        return result

    def teams_by_track(self) -> dict[int, int]:
        """Return resolved team assignments keyed by ByteTrack ID.

        Call :meth:`resolve_all` first to include tracks that are still active.
        """
        return {
            rt.track_id: rt.team
            for rt in self._resolved.values()
            if rt.team is not None
        }

    def resolved_tracks(self) -> list[ResolvedTrack]:
        """Return all resolved tracks for inspection / debugging."""
        return list(self._resolved.values())

    def number_sightings_for(self, track_id: int) -> list[Sighting]:
        """Return all jersey-number sightings accumulated for a track so far.

        Useful for callers that want to compute early lock-in decisions
        without waiting for :meth:`resolve_all` or stale auto-resolution.
        """
        return list(self._tracks.get(track_id, []))


# Backwards-compatible alias: prior code imported `NumberTracker`.
NumberTracker = PlayerTracker
