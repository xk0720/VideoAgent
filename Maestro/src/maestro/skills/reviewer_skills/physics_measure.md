---
name: physics_measure
agent: PhysicsConsistencyCritic (non-AI reviewer)
description: The MEASURED physics review chain — locate → track → certify → fit motion laws → per-entity frame-ranged verdicts (source law_verifier). Reference-free, training-free.
---

# Measured Physics Review (non-AI) — the tool chain and its contract

Role: answer, from PIXELS alone, the parameter-free question "is there ANY
physically consistent explanation for each entity's observed motion?" — never
"does it match a simulation" (that would presume masses/scale we cannot know).
This reviewer is a MEASUREMENT: its verdicts outrank VLM opinion in their
domain (motion / existence / timing) and are tagged `source="law_verifier"`.

## Tool chain (each stage's tool, in order)

1. LOCATE — GroundingDINO (`models/detection_backends.py`)
   · zero-shot detection of each annotated entity in frame 0 by NAME
   · output: normalized bbox → centroid = the tracking seed
   · no detection → heuristic seed + the verdict is marked unreliable

2. TRACK — CoTracker (`physics/track_extractor_backends.py`)
   · seeds [t=0, x_px, y_px] → per-frame point track
   · output: normalized screen-space track (x, y) ∈ [0,1], y grows DOWNWARD
     (gravity appears as +y acceleration of UNKNOWN magnitude)

3. CERTIFY — reliability gate (`physics/reliability.py`)
   · trackers LIE on generated video: churn/jitter/too-short tracks
   · a decertified track NEVER yields a measured verdict — the entity is
     DEMOTED to the VLM tier and reported as an explicit deferral.
     This gate is the honesty step nobody else has.

4. FIT LAWS — `physics/laws.py fit_best_law` + anomaly detectors
   · passive-motion family: static / constant_velocity / constant_acceleration
     (FREE gravity vector — no 9.81 assumption, no scale calibration)
   · violation = max(best-fit RMS residual, worst anomaly severity) ∈ [0,1]
   · localized anomaly detectors → typed failure modes:
     teleport → object_permanence · midair_reversal → gravity_inertia ·
     energy_gain → conservation · jerk_spike → collision

5. VERDICT — `critics/physics_consistency.py`
   · violation ≥ threshold (0.4 / strictness) → PhysicsVerdict{mode, entity,
     frame_range, severity, source="law_verifier", suggested_intervention}
     + a mirrored failed ChecklistItem (kind="physics")
   · residual-high but nothing localized → mode=UNEXPLAINED (honest: we know
     no law fits, we do NOT know which law broke)

## Rules

- Silent when nothing is verifiable (unreadable clip / no annotation) — never
  a confident verdict from no evidence.
- Every verdict names WHO (entity), WHERE (frame_range), HOW BAD (severity):
  that is what DefectReport localization and segment repair consume.
- Demo of the whole chain on any existing video (all trajectories recorded):
  `python scripts/test_physics_review.py --video x.mp4 --prompt "..."`
