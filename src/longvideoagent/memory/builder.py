"""High-level builder: takes a list of Shots (from perception) and
populates a MemoryStore, then derives Events/Stories via lightweight
heuristics (mocked in v0.1; v0.2 swaps in BaSSL / LLM summarisation).

Open-source dependencies: numpy only.
Future v0.2 additions:
    • BaSSL  — boundary-aware scene segmentation for event grouping
    • LLM   — for Story summarisation
"""
from __future__ import annotations

from typing import Iterable

from ..types import Event, NarrativeMemory, Shot, Story
from .store import MemoryStore


def _group_events(shots: list[Shot], target_event_seconds: float = 30.0) -> list[Event]:
    """Cheap greedy grouping: keep accumulating shots until ~30s pass."""
    events: list[Event] = []
    bucket: list[Shot] = []
    bucket_start = 0.0
    for s in sorted(shots, key=lambda x: (x.source_video, x.start_time)):
        if not bucket:
            bucket_start = s.start_time
        bucket.append(s)
        if s.end_time - bucket_start >= target_event_seconds:
            events.append(_finalise_event(bucket, len(events)))
            bucket = []
    if bucket:
        events.append(_finalise_event(bucket, len(events)))
    return events


def _finalise_event(bucket: list[Shot], idx: int) -> Event:
    return Event(
        event_id=f"e{idx:04d}",
        shot_ids=[s.shot_id for s in bucket],
        summary=f"{len(bucket)} shots, " + " | ".join(s.caption[:32] for s in bucket[:3]),
        start_time=bucket[0].start_time,
        end_time=bucket[-1].end_time,
    )


def _build_stories(events: list[Event]) -> list[Story]:
    """v0.1 mock: lump everything into a single 'main' story arc."""
    if not events:
        return []
    return [Story(
        story_id="main",
        title="Main arc",
        event_ids=[e.event_id for e in events],
        summary=f"Auto-generated story spanning {len(events)} events.",
        arc_role="rising",
    )]


def build_memory_from_shots(
    shots: Iterable[Shot],
    store: MemoryStore,
) -> NarrativeMemory:
    shots_list = list(shots)
    for s in shots_list:
        store.add_shot(s)
    events = _group_events(shots_list)
    for e in events:
        store.add_event(e)
    stories = _build_stories(events)
    for st in stories:
        store.add_story(st)
    return store.load_full_memory(load_features=False)


__all__ = ["build_memory_from_shots"]
