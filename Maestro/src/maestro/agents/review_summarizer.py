"""ReviewSummarizerAgent — the review 整理员 between the critics and the brain.

Maestro's reviewers are HETEROGENEOUS: MLLM critics judge semantics/physics by
watching frames (OPINION), the pixel-track law verifier MEASURES violations
(entity + frame range + severity, source="law_verifier"), and the metric tool
emits scalars. This agent consolidates all of it into ONE prioritized brief the
brain (OrchestratorAgent) reads first. Rules are evidence-backed
(docs/research/survey_review_summarizer_2026_07.md):

  1. ORGANIZE + PRIORITIZE + LOCALIZE. Suggest fix CLASSES (non-binding hints
     like "localized_regen"), never concrete tool calls — tool choice belongs
     to the brain and its skill file (judge/planner separation; Kamoi TACL'24,
     Agentless FSE'25).
  2. Every issue carries PROVENANCE (measured | opinion). Measurement outranks
     opinion in its competence domain (Tyen ACL'24; Meta repair-agent ablation).
  3. Same entity + overlapping span from BOTH source types merges into ONE
     issue with an agreement/confidence boost (cross-type agreement beats
     same-type consensus — bias-amplification, arXiv 2505.19477).
  4. Conflicts are SURFACED as first-class entries, never silently deduped;
     resolution defaults to measured precedence.
  5. Turn-over-turn progress (fixed/new/regressed/unchanged, by stable issue
     keys) + a do_not_repeat ledger from the verifier history (Reflexion-style
     verbal failure memory) so the brain is TOLD what already failed.

Signal honesty: the structured brief is 100% deterministic and every issue is
traceable to an actual reviewer output — nothing invented. An optional LLM may
REPHRASE the natural-language summary only (facts fixed, template fallback on
any error); a Mock LLM is never consulted.
"""
from __future__ import annotations

from typing import Optional

# fix modality (from DefectReport) → NON-BINDING repair-class hints. These are
# class labels the brain's skill file groups tools under — not tool names.
_FIX_CLASSES: dict[str, list[str]] = {
    "motion": ["localized_regen", "edit_in_place"],
    "presence": ["extend", "edit_in_place"],
    "content": ["keyframe_edit", "localized_regen"],
    "background": ["depth_edit"],
    "foreground": ["depth_edit"],
    "style": ["style_edit"],
}

_MEASURED_SOURCES = {"law_verifier"}


def _overlap(a: tuple, b: tuple) -> bool:
    return min(int(a[1]), int(b[1])) > max(int(a[0]), int(b[0]))


class ReviewSummarizerAgent:
    """Consolidate one reviewed clip's critiques into a prioritized brief."""

    def __init__(self, llm=None):
        # Only a REAL LLM polishes prose; mocks would inject canned nonsense
        # into the brain's prompt, so they are skipped (template stays).
        if llm is not None and type(llm).__name__.lower().startswith("mock"):
            llm = None
        self.llm = llm

    # ── evidence: trace a defect back to the reviewer outputs it came from ──
    @staticmethod
    def _evidence_for(defect, clip) -> list[dict]:
        ev: list[dict] = []
        if defect.kind == "physics":
            for v in clip.physics_verdicts:
                fr = v.frame_range or defect.frame_range
                if (v.entity or "") == defect.entity and _overlap(fr, defect.frame_range):
                    ev.append({
                        "type": "measured" if v.source in _MEASURED_SOURCES
                                else "opinion",
                        "reviewer": v.source,
                        "severity": round(float(v.severity), 3),
                        "detail": v.mode.value
                                  + (f" — {v.suggested_intervention}"
                                     if v.suggested_intervention else ""),
                    })
        else:
            ev.append({"type": "opinion", "reviewer": "semantic_critic",
                       "detail": defect.note})
        return ev or [{"type": "opinion", "reviewer": "review",
                       "detail": defect.note}]

    @staticmethod
    def _key(defect, n_frames: int) -> str:
        """Stable cross-turn identity: kind|entity|modality|clip-quarter bucket.
        Ranges shift a little after a repair; the coarse bucket keeps the same
        underlying problem matched across turns."""
        bucket = (int(defect.frame_range[0]) * 4) // max(1, int(n_frames))
        return f"{defect.kind}|{defect.entity or 'clip'}|{defect.fix_modality}|q{bucket}"

    # ── the consolidation ──
    def summarize(self, clip, spec, defect_report, history=None,
                  prev_issues: Optional[list] = None) -> dict:
        """DefectReport (localized) + raw review (provenance) + verifier history
        (ledger) + previous brief (progress) → the brain's review_brief dict."""
        history = history or []
        n_frames = getattr(defect_report, "n_frames", 0) or 1

        # 1. merge defects: same entity + overlapping span → one issue.
        issues: list[dict] = []
        for d in defect_report.sorted_by_severity():
            ev = self._evidence_for(d, clip)
            merged = None
            for iss in issues:
                if (iss["entity"] == (d.entity or "")
                        and iss["_kind"] == d.kind
                        and _overlap(iss["frame_range"], d.frame_range)):
                    merged = iss
                    break
            if merged is not None:
                merged["frame_range"] = [
                    min(merged["frame_range"][0], int(d.frame_range[0])),
                    max(merged["frame_range"][1], int(d.frame_range[1])),
                ]
                merged["evidence"].extend(ev)
                merged["severity"] = max(merged["severity"], float(d.severity))
            else:
                issues.append({
                    "_kind": d.kind,
                    "title": d.note or f"{d.kind} defect",
                    "entity": d.entity or "",
                    "frame_range": [int(d.frame_range[0]), int(d.frame_range[1])],
                    "severity": float(d.severity),
                    "fix_classes": _FIX_CLASSES.get(d.fix_modality, ["regenerate"]),
                    "fix_modality": d.fix_modality,
                    "evidence": ev,
                })

        # 2. agreement / confidence / conflicts (measured outranks opinion).
        conflicts: list[dict] = []
        for iss in issues:
            types = {e["type"] for e in iss["evidence"]}
            cross = "measured" in types and "opinion" in types
            iss["agreement"] = ("cross_type_confirmed" if cross
                                else "multi" if len(iss["evidence"]) > 1
                                else "single")
            base = 0.9 if "measured" in types else 0.6
            iss["confidence"] = round(min(0.95, base + (0.05 if cross else 0.0)), 2)
            if cross:
                sev_m = max((float(e.get("severity", 0.0))
                             for e in iss["evidence"] if e["type"] == "measured"),
                            default=0.0)
                if sev_m and abs(sev_m - iss["severity"]) > 0.4:
                    conflicts.append({
                        "entity": iss["entity"],
                        "frame_range": iss["frame_range"],
                        "note": "measured and opinion severities disagree",
                        "resolution": "measured_precedence",
                    })
                    iss["severity"] = sev_m   # measurement wins in its domain

        # 3. stable keys + progress vs the previous turn's brief.
        prev_by_key = {i.get("key"): i for i in (prev_issues or [])}

        class _KeyView:      # tiny adapter so _key reads issue dicts too
            def __init__(self, iss):
                self.kind = iss["_kind"]
                self.entity = iss["entity"]
                self.fix_modality = iss["fix_modality"]
                self.frame_range = iss["frame_range"]

        for iss in issues:
            iss["key"] = self._key(_KeyView(iss), n_frames)
            prev = prev_by_key.get(iss["key"])
            if prev is None:
                iss["status"] = "new" if prev_issues is not None else "initial"
            elif iss["severity"] > float(prev.get("severity", 0.0)) + 0.05:
                iss["status"] = "regressed"
            else:
                iss["status"] = "unchanged"
        now_keys = {i["key"] for i in issues}
        progress = {
            "fixed": [k for k in prev_by_key if k not in now_keys],
            "new": [i["key"] for i in issues if i["status"] == "new"],
            "regressed": [i["key"] for i in issues if i["status"] == "regressed"],
            "unchanged": [i["key"] for i in issues if i["status"] == "unchanged"],
        }

        # 4. rank: regressed > measured-backed > severity×confidence.
        issues.sort(key=lambda i: (
            i["status"] == "regressed",
            any(e["type"] == "measured" for e in i["evidence"]),
            i["severity"] * i["confidence"],
        ), reverse=True)
        for rank, iss in enumerate(issues, 1):
            iss["id"] = f"I-{rank}"
            iss.pop("_kind", None)

        # 5. do-not-repeat ledger from the verifier history (rejected actions).
        do_not_repeat = [
            {"tool": d.get("tool"), "args": d.get("args", {}),
             "why": f"verifier {outcome} (total stayed {total})"}
            for (d, outcome, total) in history
            if isinstance(d, dict) and outcome in ("rejected", "accept_overridden")
        ]

        headline = ""
        if issues:
            top = issues[0]
            lo, hi = top["frame_range"]
            headline = (f"Worst: {top['title']} on "
                        f"'{top['entity'] or 'the clip'}' frames {lo}-{hi} "
                        f"(severity {top['severity']:.2f}, {top['agreement']})")

        brief = {
            "headline": headline,
            "issues": issues,
            "conflicts": conflicts,
            "progress": progress,
            "do_not_repeat": do_not_repeat,
            "brief_nl": self._render_nl(headline, issues, progress, do_not_repeat),
            "brief_nl_source": "template",
        }
        self._polish(brief)
        return brief

    # ── prose rendering (deterministic template; optional LLM rephrase) ──
    @staticmethod
    def _render_nl(headline, issues, progress, do_not_repeat) -> str:
        if not issues:
            return "No open defects — the review passed everything it checked."
        lines = [headline + "."]
        for iss in issues[1:3]:
            lo, hi = iss["frame_range"]
            lines.append(f"Also: {iss['title']} on '{iss['entity'] or 'the clip'}'"
                         f" frames {lo}-{hi} (severity {iss['severity']:.2f}).")
        if progress["fixed"]:
            lines.append(f"Fixed since last turn: {len(progress['fixed'])}.")
        if progress["regressed"]:
            lines.append(f"REGRESSED: {len(progress['regressed'])} — fix first.")
        if do_not_repeat:
            worst = do_not_repeat[-1]
            lines.append(f"Do NOT repeat {worst['tool']} on the same target "
                         f"({worst['why']}).")
        return " ".join(lines)

    def _polish(self, brief: dict) -> None:
        """Ask a REAL LLM to rephrase brief_nl for clarity — facts fixed, no new
        claims. Any failure keeps the deterministic template (never blocks)."""
        if self.llm is None or not brief["issues"]:
            return
        import json
        try:
            reply = self.llm.complete(
                "Rewrite this review brief as 2-4 clear sentences for a repair "
                "planner. Use ONLY the facts given — no new issues, numbers or "
                "guesses. Keep entity names and frame ranges verbatim.\n"
                + json.dumps({"issues": brief["issues"][:3],
                              "progress": brief["progress"],
                              "do_not_repeat": brief["do_not_repeat"]},
                             ensure_ascii=False)
            )
            reply = (reply or "").strip()
            if 0 < len(reply) < 1200:
                brief["brief_nl"] = reply
                brief["brief_nl_source"] = "llm"
        except Exception:
            pass   # template already in place
