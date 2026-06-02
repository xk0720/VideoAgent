"""GEST-style event-graph construction & validation.

Borrowed (cite): "Agentic Video Generation: From Text to Executable Event Graphs
via Tool-Constrained LLM Planning" (arXiv:2604.10383) — the Graph of Events in
Space and Time (GEST) and its *separation of concerns*: the LLM plans narrative
in natural language while a programmatic backend ENFORCES constraints via
validated construction, so a spec is executable "by construction" rather than
fixed post-hoc. We also borrow its "Relation Subagents" idea (fill the
logical/semantic edges procedural generation leaves blank).

OUR innovation / departure: the Event-Graph paper hands the GEST to a 3D game
engine and loses photorealism. Maestro instead uses the event graph as a
constraint + grounding layer for a NEURAL generator — it (1) drives the physics
sketch (entities/interactions/temporal order), and (2) seeds the semantic
checklist used by the self-improving critic loop. Pixels stay neural; only the
constraints come from the graph.
"""
from __future__ import annotations

from ..types import EventEdge, EventGraph, EventNode, ShotSpec

# base verbs (inflections handled by prefix/substring matching below)
_ACTION_VERBS = [
    "throw", "fall", "drop", "run", "jump", "hit", "crash", "pour",
    "bounce", "fly", "walk", "spin", "break", "splash", "kick",
    "抛", "落", "跑", "跳", "撞", "倒", "弹", "飞", "走", "旋转", "碎", "溅",
]
_OBJECT_CUES = [
    "ball", "car", "wall", "ground", "water", "bottle", "cup", "rock", "door",
    "city", "fence", "球", "车", "墙", "地", "水", "瓶", "杯", "石", "门",
]


def _tokens(text: str) -> list[str]:
    return [t for t in (text or "").lower().replace(",", " ").split() if t]


def _match_verb(tok: str) -> str | None:
    """Map an (inflected) token to its base verb. 'thrown'->throw, 'hits'->hit."""
    for v in _ACTION_VERBS:
        if v.isascii():
            if tok.startswith(v):     # throw/thrown/throws, hit/hits, bounce/bounces
                return v
        elif v in tok:                # CJK single-char verbs
            return v
    return None


def _match_objects(toks: list[str]) -> list[str]:
    found: list[str] = []
    for tok in toks:
        for o in _OBJECT_CUES:
            if (o.isascii() and tok.startswith(o)) or (not o.isascii() and o in tok):
                if o not in found:
                    found.append(o)
                break
    return found


def build_event_graph(spec: ShotSpec) -> EventGraph:
    """Heuristic single-/multi-event graph from a shot prompt.

    v0.1 deterministic parse; v0.2 a Director sub-agent emits the graph in JSON
    under tool-constrained validation (one validated tool call per node/edge).
    """
    toks = _tokens(spec.prompt)
    actions = [v for v in (_match_verb(t) for t in toks) if v is not None]
    objects = _match_objects(toks)
    actor = objects[0] if objects else "subject"

    nodes: list[EventNode] = []
    if not actions:
        nodes.append(
            EventNode(
                event_id=f"s{spec.shot_idx}_e0", action="appear",
                actors=[actor], objects=objects, start=0.0, end=spec.duration,
            )
        )
    else:
        n = len(actions)
        seg = spec.duration / n if n else spec.duration
        for i, act in enumerate(actions):
            nodes.append(
                EventNode(
                    event_id=f"s{spec.shot_idx}_e{i}", action=act,
                    actors=[actor],
                    objects=[o for o in objects if o != actor] or objects,
                    start=round(i * seg, 3), end=round((i + 1) * seg, 3),
                )
            )
    graph = EventGraph(nodes=nodes)
    fill_relations(graph)
    return graph


def fill_relations(graph: EventGraph) -> EventGraph:
    """Relation Subagent (GEST): connect consecutive events temporally/causally.

    Simple heuristic: each event precedes the next; an impact-type action caused
    by the prior motion is marked 'causes'. v0.2: an LLM sub-agent infers richer
    logical edges.
    """
    impact = {"hit", "crash", "bounce", "splash", "break", "撞", "弹", "溅", "碎"}
    graph.edges = []
    for a, b in zip(graph.nodes, graph.nodes[1:]):
        rel = "causes" if b.action in impact else "before"
        graph.edges.append(EventEdge(src=a.event_id, dst=b.event_id, relation=rel))
    return graph


def validate_event_graph(graph: EventGraph) -> tuple[bool, list[str]]:
    """Tool-constrained validation: a graph is only accepted if structurally sound.

    Mirrors the paper's 'executable by construction' principle: catch problems at
    build time, not after rendering.
    """
    issues: list[str] = []
    if not graph.nodes:
        issues.append("empty graph: no events")
    ids = {n.event_id for n in graph.nodes}
    for n in graph.nodes:
        if not n.actors:
            issues.append(f"{n.event_id}: no actors")
        if n.end < n.start:
            issues.append(f"{n.event_id}: end<start ({n.start}->{n.end})")
    for e in graph.edges:
        if e.src not in ids or e.dst not in ids:
            issues.append(f"edge {e.src}->{e.dst}: dangling reference")
    # temporal monotonicity along 'before'/'causes' edges
    pos = {n.event_id: i for i, n in enumerate(graph.nodes)}
    for e in graph.edges:
        if e.relation in ("before", "causes") and pos.get(e.src, 0) > pos.get(e.dst, 0):
            issues.append(f"edge {e.src}->{e.dst}: violates temporal order")
    return (len(issues) == 0, issues)
