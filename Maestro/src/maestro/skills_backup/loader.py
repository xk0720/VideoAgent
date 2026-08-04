"""Predefined per-agent skills — standard folder + files form (Claude/ChatGPT
skill-framework convention, per the user's ruling):

    skills/
      brain_skills/
        orchestrator/SKILL.md       (repair brain — loaded whole into its prompt)
        window_generation/SKILL.md  (window brain — strategy selection)
        scene_write/SKILL.md        (playwriting: prompt → ordered shot list)
        video_retrieval/SKILL.md    (asset retrieval for keyframes/replacement)
      reviewer_skills/
        semantic_critic/SKILL.md    (VLM semantics/character/object dims)
        physics_critic/SKILL.md     (VLM physics opinion)
        physics_measure/SKILL.md    (non-AI measured physics chain)
      summarizer_skills/review_summarizer/SKILL.md
      verifier_skills/verifier/SKILL.md

Each skill = a FOLDER named after the skill containing SKILL.md (frontmatter:
name / agent / description) plus optional scripts/ or reference files. The
skill name resolves from frontmatter `name:`, else the FOLDER name (for
SKILL.md), else the file stem (legacy flat .md still loads). Empty SKILL.md
files are drafts, skipped. LEARNED skills (memory/skill_library.py) are a
different thing: distilled at runtime, retrieved by signature; THESE are the
hand-written operating manuals each agent starts with.
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
        # name: frontmatter > folder name (standard <skill>/SKILL.md form) > stem
        default = md.parent.name if md.name == "SKILL.md" else md.stem
        name = fm.get("name") or default
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
