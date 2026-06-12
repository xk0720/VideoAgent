"""EntityStore — Tier-4 entity persistence, now DUAL-REGISTER (v0.4).

v0.3 borrowed VideoMemory's (arXiv:2601.03655) Dynamic Memory Bank and
extended it cross-RUN (the legacy `find_or_create` path below is kept intact
for back-compat — understand.py and old JSONL files keep working).

v0.4 — OUR increment (survey_memory_2026_06.md §5 gap 2, Angles 1+2):
no prior generation-memory factorizes WHO an entity is from WHAT STATE it is
currently in. EntityMem (arXiv:2605.15199) freezes reference features — its
entities cannot evolve; VideoMemory (arXiv:2601.03655) and StoryMem
(arXiv:2512.19539) update descriptors freely — their entities drift. We split
every entity into:

  • an IMMUTABLE canonical identity register (`EntityIdentity`, frozen
    dataclass; this store exposes NO API to mutate one after `register`), and
  • a MUTABLE state register (`EntityState`) that changes ONLY through typed
    `StateTransition` entries in an append-only log.

State is never stored authoritatively: on load it is REPLAYED from the
committed/correction transitions, so every state version is traceable to a
log entry (A-MEM, arXiv:2502.12110, showed memory should carry structured
provenance; we make provenance the only write path).

Writes are verification-gated (Angle 1, write_gate.py): a transition proposed
at planning time is committed only after the gate confirms it in the ACCEPTED
rendered clip — VideoMemory writes unverified LLM descriptions and errors
compound; we commit only what rendered. Rejections stay in the log with the
discrepancy; `record_correction` is the explicit, auditable path for when the
render contradicts the proposal.

Persistence is JSONL (same style as skill_library.py: full atomic rewrite,
content-stable ids). One file holds three record kinds discriminated by a
"kind" key; legacy v0.3 lines have no "kind" and load as PersistentEntity.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..types import (
    EntityIdentity,
    EntityState,
    PersistentEntity,
    ShotSpec,
    StateTransition,
)


def _stable_entity_id(canonical_name: str) -> str:
    h = hashlib.md5(canonical_name.encode("utf-8")).hexdigest()
    return f"E{h[:12]}"


def make_identity(
    name: str,
    reference_paths: Optional[list[str]] = None,
    description: str = "",
) -> EntityIdentity:
    """Build an EntityIdentity with the content-stable id used since v0.3 —
    the same canonical name always maps to the same entity_id, so identity
    registration is idempotent across shots AND runs."""
    return EntityIdentity(
        entity_id=_stable_entity_id(name),
        name=name,
        reference_paths=list(reference_paths or []),
        description=description,
        created_ts=time.time(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transition authoring — mock stand-in for the Director agent.
#
# In the real system the Director AUTHORS transitions while planning ("the
# hero gets soaked in shot 3"). The mock pipeline has no such authoring step,
# so this tiny deterministic helper scans the shot prompt for a small cue
# list of state words and proposes the corresponding typed transitions for
# every REGISTERED entity named in the prompt. The cue list is deliberately
# small and visible — it is a stand-in, not a claim of NLU.
# ─────────────────────────────────────────────────────────────────────────────
_STATE_CUES: dict[str, tuple[str, str]] = {
    # cue word in prompt -> (state field, new value)
    "wet": ("condition", "wet"),
    "dry": ("condition", "dry"),
    "soaked": ("condition", "wet"),
    "broken": ("condition", "broken"),
    "shattered": ("condition", "broken"),
    "burning": ("condition", "burning"),
    "dirty": ("condition", "dirty"),
    "clean": ("condition", "clean"),
    "injured": ("condition", "injured"),
    "angry": ("emotion", "angry"),
    "happy": ("emotion", "happy"),
    "sad": ("emotion", "sad"),
    "scared": ("emotion", "scared"),
    "holding": ("holding", ""),       # value = word after "holding" in prompt
}


def _tokens(text: str) -> list[str]:
    return [w.strip(".,;:!?'\"()") for w in text.lower().split()]


def propose_transitions_from_spec(
    spec: ShotSpec, store: "EntityStore",
) -> list[StateTransition]:
    """Derive proposed transitions for registered entities named in the spec.

    Entities are matched by name against the prompt and the physics
    annotation's entity list; cues are matched as whole tokens (so "dry"
    never fires on "laundry"). A transition is only proposed when it would
    actually change the current state register, and never duplicated for the
    same (entity, shot, field, new) already in the log.
    """
    toks = _tokens(spec.prompt)
    tok_set = set(toks)
    annotated = {
        e.name.lower()
        for e in (spec.physics_annotation.entities if spec.physics_annotation else [])
    }
    out: list[StateTransition] = []
    for ident in store.identities.values():
        name = ident.name.lower()
        if name not in tok_set and name not in annotated:
            continue
        st = store.states.get(ident.entity_id)
        attrs = st.attributes if st else {}
        for cue, (fld, val) in _STATE_CUES.items():
            if cue not in tok_set:
                continue
            if cue == "holding":     # value = the word following the cue
                i = toks.index(cue)
                val = toks[i + 1] if i + 1 < len(toks) else "object"
            if attrs.get(fld) == val:
                continue             # no-op: state already says so
            if any(
                t.entity_id == ident.entity_id and t.shot_idx == spec.shot_idx
                and t.field == fld and t.new == val
                for t in store.log
            ):
                continue             # already proposed/decided for this shot
            out.append(StateTransition(
                entity_id=ident.entity_id, shot_idx=spec.shot_idx,
                field=fld, old=attrs.get(fld, ""), new=val,
                cause=f"prompt cue '{cue}' in shot {spec.shot_idx}",
            ))
    return out


class EntityStore:
    """Dual-register entity store + append-only transition log.

    Legacy v0.3 surface (`find_or_create` / `get` / `__len__`, cross-run
    embedding dedup) is preserved unchanged — understand.py still calls it.

    v0.4 surface:
      register(identity)            idempotent on entity_id; identities are
                                    frozen and this class has no mutator
      propose(transition)           append a "proposed" entry to the log
      commit_gated(clip,spec,gate)  gate each proposal for that shot →
                                    commit (apply to state) or reject
      record_correction(...)        explicit auditable write when the render
                                    contradicted the proposal
      current_state(entity_id)      composed identity+state view
      history(entity_id)            ordered transitions (the audit trail)
      reentry_context(entity_id)    EntityBench-style long-gap re-entry
                                    payload: identity refs + current attrs
    """

    MATCH_THRESHOLD = 0.85   # high bar to avoid false reuse (legacy dedup)

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        # Legacy v0.3 register (PersistentEntity) — kept for back-compat.
        self.entities: list[PersistentEntity] = []
        self._by_id: dict[str, PersistentEntity] = {}
        # v0.4 dual registers + append-only log.
        self.identities: dict[str, EntityIdentity] = {}
        self.states: dict[str, EntityState] = {}
        self.log: list[StateTransition] = []
        if self.path and self.path.exists():
            self._load()

    # ── persistence (skill_library style: JSONL, full atomic rewrite) ──────
    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            kind = d.get("kind")
            if kind == "identity":
                ident = EntityIdentity(
                    entity_id=d["entity_id"],
                    name=d["name"],
                    reference_paths=list(d.get("reference_paths", [])),
                    description=d.get("description", ""),
                    created_ts=float(d.get("created_ts", 0.0)),
                )
                self.identities[ident.entity_id] = ident
            elif kind == "transition":
                t = StateTransition(
                    entity_id=d["entity_id"],
                    shot_idx=int(d["shot_idx"]),
                    field=d["field"],
                    old=d.get("old", ""),
                    new=d.get("new", ""),
                    cause=d.get("cause", ""),
                    status=d.get("status", "proposed"),
                    evidence=d.get("evidence", ""),
                    ts=float(d.get("ts", 0.0)),
                )
                self.log.append(t)
                # State is REPLAYED from the log, never stored — every state
                # version stays traceable to exactly one transition entry.
                if t.status in ("committed", "correction"):
                    self._apply(t)
            else:
                # Legacy v0.3 record (no "kind" key) — PersistentEntity shape.
                ent = PersistentEntity(
                    entity_id=d["entity_id"],
                    canonical_name=d["canonical_name"],
                    embedding=embed_text(d["canonical_name"]),
                    source_paths=d.get("source_paths", []),
                    style_descriptors=d.get("style_descriptors", {}),
                    appearance_log=d.get("appearance_log", []),
                    physics_profile=d.get("physics_profile", {}),
                    first_seen_ts=float(d.get("first_seen_ts", 0.0)),
                    last_seen_ts=float(d.get("last_seen_ts", 0.0)),
                )
                self.entities.append(ent)
                self._by_id[ent.entity_id] = ent

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            # Legacy records first (kind-less lines, v0.3 shape).
            for e in self.entities:
                f.write(json.dumps({
                    "entity_id": e.entity_id,
                    "canonical_name": e.canonical_name,
                    "source_paths": e.source_paths,
                    "style_descriptors": e.style_descriptors,
                    "appearance_log": e.appearance_log,
                    "physics_profile": e.physics_profile,
                    "first_seen_ts": e.first_seen_ts,
                    "last_seen_ts": e.last_seen_ts,
                }, ensure_ascii=False) + "\n")
            for ident in self.identities.values():
                f.write(json.dumps({
                    "kind": "identity",
                    "entity_id": ident.entity_id,
                    "name": ident.name,
                    "reference_paths": ident.reference_paths,
                    "description": ident.description,
                    "created_ts": ident.created_ts,
                }, ensure_ascii=False) + "\n")
            for t in self.log:
                f.write(json.dumps({
                    "kind": "transition",
                    "entity_id": t.entity_id,
                    "shot_idx": t.shot_idx,
                    "field": t.field,
                    "old": t.old,
                    "new": t.new,
                    "cause": t.cause,
                    "status": t.status,
                    "evidence": t.evidence,
                    "ts": t.ts,
                }, ensure_ascii=False) + "\n")

    # ── identity register (immutable after registration) ──────────────────
    def register(self, identity: EntityIdentity) -> EntityIdentity:
        """Idempotent on entity_id. Re-registering an existing id returns the
        ORIGINAL register untouched — there is deliberately no update path
        (the identity register is the immutable half of the dual register)."""
        existing = self.identities.get(identity.entity_id)
        if existing is not None:
            return existing
        self.identities[identity.entity_id] = identity
        self._persist()
        return identity

    # ── state transitions (the only write path into EntityState) ──────────
    def _apply(self, t: StateTransition) -> None:
        st = self.states.setdefault(t.entity_id, EntityState())
        st.attributes[t.field] = t.new
        st.shot_idx = t.shot_idx
        st.version += 1

    def propose(self, transition: StateTransition) -> StateTransition:
        """Append a proposal to the log. Proposals do NOT touch the state
        register — only `commit_gated` / `record_correction` apply changes."""
        transition.status = "proposed"
        if not transition.ts:
            transition.ts = time.time()
        self.log.append(transition)
        self._persist()
        return transition

    def commit_gated(self, clip, spec: ShotSpec, gate) -> dict[str, int]:
        """Run the verification gate on every proposed transition for this
        shot; commit confirmed ones into the state register, mark the rest
        rejected. Either way the gate's evidence string is logged — the
        decision is auditable, not silent. Returns {"committed": n,
        "rejected": m}."""
        committed = rejected = 0
        for t in self.log:
            if t.status != "proposed" or t.shot_idx != spec.shot_idx:
                continue
            ok, evidence = gate.confirm(t, clip, spec)
            t.evidence = evidence
            if ok:
                t.status = "committed"
                self._apply(t)
                committed += 1
            else:
                t.status = "rejected"
                rejected += 1
        if committed or rejected:
            self._persist()
        return {"committed": committed, "rejected": rejected}

    def record_correction(
        self,
        entity_id: str,
        shot_idx: int,
        field: str,
        new: str,
        cause: str,
        evidence: str,
    ) -> StateTransition:
        """Explicit correction entry: the render contradicted the proposal,
        so memory follows the pixels — but through a logged, typed entry
        (status="correction"), never a silent overwrite."""
        st = self.states.get(entity_id)
        t = StateTransition(
            entity_id=entity_id, shot_idx=shot_idx, field=field,
            old=(st.attributes.get(field, "") if st else ""), new=new,
            cause=cause, status="correction", evidence=evidence,
            ts=time.time(),
        )
        self.log.append(t)
        self._apply(t)
        self._persist()
        return t

    # ── composed views ─────────────────────────────────────────────────────
    def current_state(self, entity_id: str) -> Optional[dict]:
        """Composed view: immutable identity + current mutable state."""
        ident = self.identities.get(entity_id)
        if ident is None:
            return None
        st = self.states.get(entity_id) or EntityState()
        return {
            "entity_id": ident.entity_id,
            "name": ident.name,
            "reference_paths": list(ident.reference_paths),
            "description": ident.description,
            "attributes": dict(st.attributes),
            "state_version": st.version,
            "last_state_shot_idx": st.shot_idx,
        }

    def history(self, entity_id: str) -> list[StateTransition]:
        """Ordered transition log for one entity — the continuity audit trail
        (the log list is append-only, so order == proposal order)."""
        return [t for t in self.log if t.entity_id == entity_id]

    def reentry_context(self, entity_id: str) -> Optional[dict]:
        """EntityBench-style long-gap re-entry payload: when an entity
        reappears shots later, condition the generator on its frozen identity
        refs PLUS its last committed state, with a ready-made prompt
        fragment. Returns None for unregistered ids."""
        view = self.current_state(entity_id)
        if view is None:
            return None
        attrs = ", ".join(f"{k}: {v}" for k, v in sorted(view["attributes"].items()))
        view["conditioning_text"] = (
            view["name"] + (f" ({attrs})" if attrs else "")
        )
        return view

    # ── legacy v0.3 surface (cross-run embedding dedup) — unchanged ───────
    def find_or_create(
        self,
        canonical_name: str,
        source_path: str = "",
        task_id: str = "",
        bbox: Optional[list[float]] = None,
    ) -> PersistentEntity:
        emb = embed_text(canonical_name)
        # First, look for a strong embedding match.
        for ent in self.entities:
            if ent.embedding is None:
                continue
            if cosine(emb, ent.embedding) >= self.MATCH_THRESHOLD:
                # Cross-run reuse: log the new appearance + update timestamps.
                if source_path and source_path not in ent.source_paths:
                    ent.source_paths.append(source_path)
                ent.appearance_log.append(
                    {"task_id": task_id, "source_path": source_path,
                     "bbox": bbox or [], "ts": time.time()}
                )
                ent.last_seen_ts = time.time()
                self._persist()
                return ent
        # Otherwise, create new.
        ent = PersistentEntity(
            entity_id=_stable_entity_id(canonical_name),
            canonical_name=canonical_name,
            embedding=emb,
            source_paths=[source_path] if source_path else [],
            appearance_log=[{"task_id": task_id, "source_path": source_path,
                             "bbox": bbox or [], "ts": time.time()}],
            first_seen_ts=time.time(),
            last_seen_ts=time.time(),
        )
        self.entities.append(ent)
        self._by_id[ent.entity_id] = ent
        self._persist()
        return ent

    def get(self, entity_id: str) -> Optional[PersistentEntity]:
        return self._by_id.get(entity_id)

    def __len__(self) -> int:
        return len(self.entities)
