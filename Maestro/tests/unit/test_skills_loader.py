"""Predefined per-agent skill files (skills/) + loader + consumer wiring."""
from maestro.skills.loader import (
    load_skill,
    load_skill_catalog,
    parse_frontmatter,
    render_skill_index,
)


def test_catalog_contains_every_agent_skill():
    cat = load_skill_catalog()
    assert {"orchestrator", "semantic_critic", "physics_critic",
            "physics_measure", "review_summarizer", "verifier"} <= set(cat)
    for name, meta in cat.items():
        assert meta["description"], name          # frontmatter present
        assert meta["agent"], name
        assert meta["body"].strip(), name         # non-empty operating manual


def test_frontmatter_parsing_and_index():
    fm, body = parse_frontmatter("---\nname: x\ndescription: d\n---\n\nBODY")
    assert fm == {"name": "x", "description": "d"}
    assert body == "BODY"
    fm2, body2 = parse_frontmatter("no frontmatter here")
    assert fm2 == {} and body2 == "no frontmatter here"
    assert "orchestrator" in render_skill_index()


def test_brain_loads_its_skill_from_the_skills_folder():
    """The orchestrator's prompt body now comes from
    skills/brain_skills/orchestrator.md (frontmatter stripped)."""
    from maestro.agents.orchestrator import OrchestratorAgent

    orch = OrchestratorAgent()
    assert "Tool catalog" in orch._skill_prompt
    assert "simulate_reference" in orch._skill_prompt
    assert not orch._skill_prompt.startswith("---")   # frontmatter stripped


def test_summarizer_polish_instruction_comes_from_skill_file():
    from maestro.agents.review_summarizer import ReviewSummarizerAgent

    instr = ReviewSummarizerAgent()._polish_instruction()
    assert "Use ONLY the facts given" in instr
    assert "measured_precedence" in instr    # the skill body, not the fallback


def test_measured_physics_skill_documents_the_real_chain():
    body = load_skill("physics_measure")["body"]
    for stage in ("GroundingDINO", "CoTracker", "CERTIFY", "fit_best_law",
                  "law_verifier"):
        assert stage in body, stage


def test_folder_skill_form_and_user_drafts():
    """标准 folder+SKILL.md 形态:名字从 frontmatter/文件夹名解析;用户草稿
    (scene_write / video_retrieval)已填充并入目录。"""
    from maestro.skills.loader import load_skill_catalog

    cat = load_skill_catalog()
    for name in ("scene_write", "video_retrieval", "window_generation"):
        assert name in cat, name
        assert cat[name]["path"].endswith(f"{name}/SKILL.md")
