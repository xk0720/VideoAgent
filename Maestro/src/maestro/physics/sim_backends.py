"""Genesis physics-simulation backend — sim as REPAIR CONDITIONING (v0.5).

Borrowed (cite): NEWTON — Agentic Planning for Physics-Following Video
Generation (arXiv:2605.18396). NEWTON's key move is NOT "simulation as a
verifier": it uses a Genesis simulation to produce a physics-CORRECT reference
video that CONDITIONS the generator (Seedance's reference_videos channel), and
verifies with a VLM instead (pre-generation condition judge + blind A/B).
Maestro adopts the conditioning half: when the MEASURED physics verifier
(law_verifier) localizes a motion violation, the brain can call
`simulate_reference` — write a structured scene_spec, simulate it here, and
regenerate the shot conditioned on the resulting reference clip. Our measured
verdicts tell the brain WHEN to reach for the simulator; NEWTON's planner has
to guess.

v1 scope: RIGID BODIES ONLY (sphere / box / cylinder / plane + gravity +
initial velocities). That covers the failure modes our law verifier actually
measures (gravity_inertia, collision, conservation). NEWTON's particle solvers
(SPH liquid, MPM sand/snow/elastic, PBD cloth) are NOT ported — a defect
needing them degrades honestly (ValueError), never a fake sim.

Training-free; Genesis is a heavy GPU dependency, so everything degrades
LOUDLY without it (same policy as CoTracker/GroundingDINO).

scene_spec (the brain writes this; documented in skills/brain_skills/orchestrator.md):
    {
      "objects": [
        {"id": "ball", "type": "sphere", "pos": [0,0,1.0], "radius": 0.1,
         "init_velocity": [0.5,0,0], "fixed": false}, ...
      ],
      "gravity": [0, 0, -9.81]          # optional, default shown
    }
A floor plane at z=0 is ALWAYS added implicitly. Units are SI (meters).
Timing is fixed (NEWTON's policy): 1.0 s of physics played back over 3.0 s at
24 fps — a 3x slow-motion clip with no dead tail, valid as a generator
reference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Fixed reference-clip timing (ported from NEWTON tools/simulator/core.py).
SIM_DURATION_S = 3.0        # playback length (seconds)
SIM_RENDER_FPS = 24.0       # playback frame rate
SIM_PHYS_DURATION_S = 1.0   # real physics time captured (then slowed 3x)

_RIGID_TYPES = {"sphere", "box", "cylinder", "plane"}


def describe_scene_spec(spec: dict) -> str:
    """Ground-truth one-line description of the sim scene from its spec
    (NEWTON's `_describe_scene_spec`): exact movable-object counts by type, so
    a judge/critic reads the TRUE count from the spec instead of (mis)counting
    the rendered clip. Ignores fixed supports and the floor."""
    from collections import Counter

    counts: Counter = Counter()
    for o in spec.get("objects", []):
        if o.get("fixed") or o.get("type") == "plane":
            continue
        counts[str(o.get("type", "object"))] += 1
    if not counts:
        return ""
    parts = [f"{n} {label}" + ("s" if n > 1 else "") for label, n in counts.items()]
    return "a physics simulation containing exactly " + ", ".join(parts)


def validate_scene_spec(spec: Any) -> list[dict]:
    """Validate a brain-written scene_spec; return its objects list or raise
    ValueError with a message the brain can act on next turn."""
    if not isinstance(spec, dict) or not isinstance(spec.get("objects"), list) \
            or not spec["objects"]:
        raise ValueError("scene_spec must be a dict with a non-empty 'objects' list")
    for o in spec["objects"]:
        if not isinstance(o, dict):
            raise ValueError(f"object entries must be dicts, got {type(o).__name__}")
        t = o.get("type")
        if t not in _RIGID_TYPES:
            raise ValueError(
                f"unknown object type {t!r}; v1 supports {sorted(_RIGID_TYPES)}")
        mat = o.get("material", "rigid")
        if mat != "rigid":
            raise ValueError(
                f"material {mat!r} not supported: v1 is RIGID-ONLY (NEWTON's "
                "SPH/MPM/PBD particle solvers are not ported); use rigid bodies "
                "or a different repair tool")
    return spec["objects"]


class GenesisSimClient:
    """Run a rigid-body Genesis simulation from a scene_spec.

    run(spec) -> {"video_path", "trajectory", "summary", "scene_desc"}.
    Loud RuntimeError when the `genesis` package is missing (GPU dependency —
    never silently mocked)."""

    def __init__(self, name: str = "genesis", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.backend = self.config.get("backend", "gpu")
        self.output_dir = Path(
            self.config.get("output_dir")
            or os.environ.get("MAESTRO_SIM_DIR", "outputs/sim")
        )
        self._gs = None

    def capabilities(self) -> set[str]:
        return {"sim"}

    def _ensure_gs(self):
        if self._gs is not None:
            return self._gs
        try:
            import genesis as gs  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "GenesisSimClient needs the `genesis-world` package (GPU): "
                "pip install genesis-world — or drop the simulate_reference "
                "tool (the brain menu hides it when no sim client is wired)."
            ) from exc
        gs.init(backend=gs.gpu if self.backend == "gpu" else gs.cpu)
        self._gs = gs
        return gs

    # ── helpers (condensed rigid-only port of NEWTON SimCore) ──
    @staticmethod
    def _half_extent(o: dict):
        import numpy as np

        return 0.5 * np.array(o.get("size", [2.0 * o.get("radius", 0.1)] * 3),
                              dtype=float)

    def _make_morph(self, gs, o: dict):
        t = o["type"]
        pos = tuple(o.get("pos", [0.0, 0.0, 0.0]))
        opts: dict = {}
        if o.get("euler") is not None:
            opts["euler"] = tuple(o["euler"])
        if o.get("fixed", False):
            opts["fixed"] = True
        if t == "sphere":
            return gs.morphs.Sphere(pos=pos, radius=float(o.get("radius", 0.1)), **opts)
        if t == "box":
            return gs.morphs.Box(pos=pos, size=tuple(o.get("size", [0.1, 0.1, 0.1])), **opts)
        if t == "cylinder":
            return gs.morphs.Cylinder(
                pos=pos, radius=float(o.get("radius", 0.1)),
                height=float(o.get("height", 0.1)), **opts)
        return gs.morphs.Plane()

    def _frame_camera(self, aabb, res=(1280, 720), fov=45.0, margin=0.7):
        """Side-and-above camera framing the box every body sweeps through
        (NEWTON's data-driven policy: frame the WHOLE motion, not the start)."""
        import math

        (xlo, ylo, zlo), (xhi, yhi, zhi) = aabb
        cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
        ex, ez = (xhi - xlo) / 2, (zhi - zlo) / 2
        vfov = math.radians(fov)
        hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * (res[0] / res[1]))
        dist = max(max(ez, 0.1) / math.tan(vfov / 2.0),
                   max(ex, 0.1) / math.tan(hfov / 2.0)) * margin
        cam_pos = (cx, cy - dist, (zlo + zhi) / 2 + 0.85 * dist)
        center = (cx, cy, zlo + 0.25 * (zhi - zlo))
        return cam_pos, center, fov

    def run(self, spec: dict) -> dict:
        import numpy as np

        objs = validate_scene_spec(spec)
        gs = self._ensure_gs()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # implicit floor (NEWTON policy: the brain never specifies it)
        if not any(o.get("type") == "plane" for o in objs):
            objs.append({"id": "floor", "type": "plane", "fixed": True})

        dt = float(spec.get("dt", 0.01))
        n_steps = max(1, int(round(SIM_PHYS_DURATION_S / dt)))
        n_frames = max(1, int(round(SIM_DURATION_S * SIM_RENDER_FPS)))
        render_steps = set(np.linspace(0, n_steps - 1, n_frames, dtype=int).tolist())

        def _build(with_camera_aabb=None):
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=dt, gravity=tuple(spec.get("gravity", [0.0, 0.0, -9.81]))),
                vis_options=gs.options.VisOptions(shadow=False, plane_reflection=False),
                show_viewer=False,
            )
            ents = []
            for i, o in enumerate(objs):
                ent = scene.add_entity(morph=self._make_morph(gs, o),
                                       material=gs.materials.Rigid())
                ents.append((o.get("id", f"{o['type']}_{i}"), o, ent))
            cam = None
            if with_camera_aabb is not None:
                cam_pos, center, fov = self._frame_camera(with_camera_aabb)
                cam = scene.add_camera(res=(1280, 720), pos=cam_pos,
                                       lookat=center, fov=fov, GUI=False)
            scene.build()
            for _oid, o, ent in ents:
                if o.get("type") == "plane" or o.get("fixed", False):
                    continue
                v = o.get("init_velocity")
                if v is not None:
                    w = o.get("init_angular", [0.0, 0.0, 0.0])
                    try:
                        ent.set_dofs_velocity(list(v) + list(w))
                    except Exception:
                        pass
            return scene, ents, cam

        def _step_and_sample(scene, ents, cam=None):
            traj: dict = {oid: [] for oid, o, _ in ents
                          if o.get("type") != "plane" and not o.get("fixed", False)}
            for step in range(n_steps):
                scene.step()
                for oid, o, ent in ents:
                    if oid not in traj:
                        continue
                    p = ent.get_pos()
                    p = np.asarray(p.cpu() if hasattr(p, "cpu") else p)
                    traj[oid].append(p.reshape(-1)[:3].tolist())
                if cam is not None and step in render_steps:
                    cam.render()
            return traj

        # physics-only pre-pass → the AABB the bodies sweep through → camera
        scene, ents, _ = _build()
        traj = _step_and_sample(scene, ents)
        pts = np.asarray([p for pp in traj.values() for p in pp], dtype=float)
        pad = max((float(np.max(self._half_extent(o))) for o in objs
                   if o.get("type") != "plane"), default=0.1) + 0.15
        lo = pts.min(axis=0) - pad
        hi = pts.max(axis=0) + pad
        lo[2] = min(lo[2], 0.0)
        aabb = (tuple(lo.tolist()), tuple(hi.tolist()))

        # real pass with the camera framed to the full motion
        scene, ents, cam = _build(with_camera_aabb=aabb)
        cam.start_recording()
        traj = _step_and_sample(scene, ents, cam)
        video_path = self.output_dir / f"sim_{abs(hash(str(spec))) % 10**8}.mp4"
        cam.stop_recording(save_to_filename=str(video_path),
                           fps=int(SIM_RENDER_FPS))

        summary = "; ".join(
            f"{oid}: moved {float(np.linalg.norm(np.array(pp[-1]) - np.array(pp[0]))):.2f} m, "
            f"z {pp[0][2]:.2f}->{pp[-1][2]:.2f}"
            for oid, pp in traj.items() if pp
        ) or "no movable objects sampled"
        return {"video_path": video_path, "trajectory": traj,
                "summary": summary, "scene_desc": describe_scene_spec(spec)}


def build_sim_client(spec: str | dict | None) -> Optional[GenesisSimClient]:
    """None / "" → None (no sim tool in the brain's menu — the honest default);
    "genesis" / {"name": "genesis", ...} → GenesisSimClient (loud without the
    genesis package at RUN time, not at build time)."""
    if spec is None or spec == "":
        return None
    name = spec.get("name", "") if isinstance(spec, dict) else str(spec)
    config = spec if isinstance(spec, dict) else {}
    if name.lower() == "genesis":
        return GenesisSimClient(config=config)
    raise ValueError(f"Unknown sim backend '{name}'. Known: genesis (or None)")
