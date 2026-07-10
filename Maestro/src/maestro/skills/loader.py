"""Predefined per-agent skill files (NEWTON planner_skills-style).

Every agent's operating knowledge lives in an editable markdown file with YAML
frontmatter (name / agent / description), grouped by role:

    skills/
      brain_skills/       orchestrator.md      (the brain — LOADED into its prompt)
      reviewer_skills/    semantic_critic.md   (VLM semantics/character/object)
                          physics_critic.md    (VLM physics opinion)
                          physics_measure.md   (non-AI measured physics chain)
      summarizer_skills/  review_summarizer.md (the review 整理员 — polish prompt)
      verifier_skills/    verifier.md          (the monotonic gate)

Loading (borrowed from NEWTON loop/run_loop.py `load_skill_catalog`, without
the read_skill indirection — our catalog is small enough to load directly):
frontmatter is stripped; `body` is what an agent puts in its prompt. LEARNED
skills (memory/skill_library.py) are a different thing: those are distilled
at runtime and retrieved by signature; THESE are the hand-written operating
manuals each agent starts with.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SKILLS_ROOT = Path(__file__).resolve().parent


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """(frontmatter dict, body) for a markdown file with `---` frontmatter.
    No frontmatter → ({}, whole text)."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm: dict = {}
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            return fm, text[end + 4:].lstrip("\n")
    return {}, text


def load_skill_catalog(root: Optional[Path] = None) -> dict[str, dict]:
    """name → {description, agent, body, path} for every skills/**/*.md."""
    root = Path(root) if root else SKILLS_ROOT
    catalog: dict[str, dict] = {}
    if not root.is_dir():
        return catalog
    for md in sorted(root.rglob("*.md")):
        if md.name == "README.md":
            continue
        text = md.read_text(encoding="utf-8")
        if not text.strip():
            continue    # empty placeholder (a skill being drafted) — not a skill yet
        fm, body = parse_frontmatter(text)
        name = fm.get("name") or md.stem
        catalog[name] = {
            "description": fm.get("description", ""),
            "agent": fm.get("agent", ""),
            "body": body,
            "path": str(md),
        }
    return catalog


def load_skill(name: str, root: Optional[Path] = None) -> Optional[dict]:
    """One skill by frontmatter name (or file stem); None if absent."""
    return load_skill_catalog(root).get(name)


def render_skill_index(catalog: Optional[dict] = None) -> str:
    """One line per skill (name — agent — description), for docs/logs."""
    catalog = catalog if catalog is not None else load_skill_catalog()
    lines = ["Predefined agent skills:"]
    for name, meta in catalog.items():
        lines.append(f"- {name} [{meta['agent']}]: {meta['description']}")
    return "\n".join(lines)
