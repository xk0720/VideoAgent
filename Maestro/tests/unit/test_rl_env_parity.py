"""训练=生产同构锁(2026-08-19 用户令:agent loop 训练和生产必须完全
一样)。rl/env 是 src/maestro 生成路径的逐字移植 —— 本文件在 CI 里锁
两边不漂移:任何一边单独改了,这里立刻红。

两类锁:
① 行为锁:同输入 → 同输出(prompt 拼装、清洗闸、槽位清单、菜单、
   对白链、正典闸、引用闸 —— 直接调两边函数比对);
② 源文锁:整文件移植件(storyboard/ref_slots/space_bible/
   junction_stitcher/logging_utils/language)去掉 import 行与移植头注
   后逐字节一致。
"""
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rl"))
sys.path.insert(0, str(REPO / "src"))

import env.window_core as W                                   # noqa: E402
import maestro.pipeline.window_loop as P                      # noqa: E402


# ── ① 行为锁 ────────────────────────────────────────────────────────
_CAST = {"小明": "static: 蓝夹克短发男子; dynamic: none",
         "阿芳": "static: 红围裙长发女子; dynamic: none"}


class _VG:
    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_images",
                "first_frame_plus_refs", "t2i"}

    @staticmethod
    def ref_token(n):
        return f"<<<image_{n}>>>"


class _Entry:
    shot_idx = 2
    label = "shot 3"
    description = "Shot 3: <小明>把饭团递给<阿芳>"
    end_state = "两人在收银台前,镜头静止"
    variation = "medium"
    opening_frame = ""
    dialogue = "就这个吧"
    dialogue_speaker = "小明"
    images: list = []
    keyframe_path = None
    junction_meta: dict = {}


class _Prev:
    video_path = "/tmp/prev.mp4"
    end_state = "小明拿着饭团"


def test_decision_prompt_and_skills_identical():
    for kind in ("generation-condition", "image-plan"):
        assert W._skill_body(kind) == P._skill_body(kind)
    for name in ("scene_write", "screenplay", "character_extract",
                 "scene_image", "junction_stitch", "window_generation",
                 "image_plan"):
        assert W._skill_body_named(name) == P._skill_body_named(name), name
    menu = [{"name": "ref2v", "description": "x"}]
    for ctx in ({"prompt_language": "zh", "shot": {"label": "s"}},
                {"prompt_language": "en"}):
        assert W.decision_prompt("SKILL", menu, ctx) == \
            P.decision_prompt("SKILL", menu, ctx)
    # trainer 用的 env.skills.decision_prompt 也必须与生产一致
    from env.skills import decision_prompt as trainer_dp
    assert trainer_dp("SKILL", menu, {"prompt_language": "zh"}) == \
        P.decision_prompt("SKILL", menu, {"prompt_language": "zh"})


def test_condition_menu_and_slots_identical():
    e, pv, vg = _Entry(), _Prev(), _VG()
    ports = {"小明": "/tmp/p1.png", "阿芳": "/tmp/p2.png"}
    assert W._condition_menu(e, pv, vg, portraits=ports) == \
        P._condition_menu(e, pv, vg, portraits=ports)
    for strat in ("ref2v", "i2v_first", "t2v"):
        s1 = W._slot_manifest(strat, e, pv, use_prev_tail=True,
                              source_videos=[], portraits=ports,
                              video_gen=vg)
        s2 = P._slot_manifest(strat, e, pv, use_prev_tail=True,
                              source_videos=[], portraits=ports,
                              video_gen=vg)
        assert s1 == s2, strat


def test_text_gates_identical():
    txt = ("Shot 3: <小明>把饭团递给<阿芳>,旁白:\"深夜\"。"
           "static: 蓝夹克短发男子; dynamic: none 音效:雨声。")
    assert W._strip_markers(txt) == P._strip_markers(txt)
    assert W._scrub_cast_labels(txt, _CAST) == \
        P._scrub_cast_labels(txt, _CAST)
    assert W._scene_text_for_prompt(txt) == P._scene_text_for_prompt(txt)
    assert W._static_half(_CAST["小明"]) == P._static_half(_CAST["小明"])
    assert W._cast_in_shot(_Entry.description, _CAST) == \
        P._cast_in_shot(_Entry.description, _CAST)
    ns = {"小明": "<<<image_2>>>", "阿芳": "<<<image_3>>>"}
    body = '小明看着阿芳说:"就这个吧"。'
    assert W._names_to_tokens(body, ns) == P._names_to_tokens(body, ns)
    assert W._with_dialogue(body, _Entry(), _CAST, ns) == \
        P._with_dialogue(body, _Entry(), _CAST, ns)
    p, n1 = W._enforce_cast_canon("小明走过", {"小明": _CAST["小明"]},
                                  _CAST)
    q, n2 = P._enforce_cast_canon("小明走过", {"小明": _CAST["小明"]},
                                  _CAST)
    assert p == q
    slots = [{"slot": "<<<image_1>>>", "content": "c",
              "referenceable": True}]
    assert W.validate_references("看<<<image_1>>>", slots) == \
        P.validate_references("看<<<image_1>>>", slots)
    assert W.validate_references("看<<<image_9>>>", slots)[1]["ok"] \
        is False
    assert W._CONDITION_PRIORITY == P._CONDITION_PRIORITY
    assert W._PLAN_PRIORITY == P._PLAN_PRIORITY
    assert W._PLAN_ROLES == P._PLAN_ROLES
    assert W._ANCHORED_STRATEGIES == P._ANCHORED_STRATEGIES
    assert W._PIN_SENTENCE == P._PIN_SENTENCE
    assert W._portrait_slot_content("小明") == \
        P._portrait_slot_content("小明")


def test_scene_write_prompt_source_identical():
    """分镜任务 prompt 的拼装源文一致(冻结 agent 的输入分布不漂):
    直接比对两边 _write_outline 的函数源码(去空白)——它内嵌了完整
    的 STRICT JSON 契约与 HANDOFF LAW 原文。"""
    a = re.sub(r"\s+", " ", inspect.getsource(W._write_outline))
    b = re.sub(r"\s+", " ", inspect.getsource(P._write_outline))
    assert a == b


def test_generation_ladder_source_identical():
    """条件执行降级梯与派生缝合的源码一致(去空白;env 侧允许的唯一
    差别 = import shim,函数体内不含 import,应逐字符相同)。"""
    def _src(fn_obj) -> str:
        # 遮蔽惰性 import 行(env 侧唯一许可的重写点),其余逐字比
        lines = [("<IMPORT>" if re.match(r"\s*(from|import)\s", ln)
                  or re.match(r"\s*(_frame_after_cut|_spaced_retry|"
                              r"frame_review_ok)[,)]?\s*$", ln)
                  else ln)
                 for ln in inspect.getsource(fn_obj).splitlines()]
        return re.sub(r"(<IMPORT>\s*)+", "<IMPORT> ",
                      re.sub(r"\s+", " ", "\n".join(lines)))

    for fn in ("_generate_with_condition", "_derive_junction_frame",
               "_junction_state", "_judge_junction_cast",
               "_map_tail_report", "_map_markers", "_parse_tail_report",
               "_execute_image_plan", "_image_plan_menu",
               "_make_keyframe", "_brain_pick", "_decide",
               "_write_screenplay", "_extract_characters",
               "_write_bg_prompts", "_ensure_cast_portraits",
               "_with_dialogue", "_scrub_setting_sentence",
               "_slot_manifest", "_condition_menu", "_final_cut"):
        a = _src(getattr(W, fn))
        b = _src(getattr(P, fn))
        assert a == b, f"{fn} 与生产漂移"


def test_wholefile_ports_identical():
    """②源文锁:整文件移植件与生产原件逐字节一致(剥移植头注 +
    import 行重写)。"""
    pairs = [("rl/env/storyboard.py", "src/maestro/memory/storyboard.py"),
             ("rl/env/ref_slots.py", "src/maestro/pipeline/ref_slots.py"),
             ("rl/env/space_bible.py",
              "src/maestro/pipeline/space_bible.py"),
             ("rl/env/junction_stitcher.py",
              "src/maestro/agents/junction_stitcher.py"),
             ("rl/env/logging_utils.py", "src/maestro/logging_utils.py"),
             ("rl/env/language.py", "src/maestro/language.py")]

    def _norm(text: str) -> str:
        out = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# 2026-08-19 用户令【训练=生产完全同构"):
                continue
            if s.startswith("# src/maestro/") \
                    or s.startswith("# 改生产原件必须同步改这里"):
                continue
            if s.startswith(("from ..", "from .", "from env.")) \
                    or re.match(r"\s*from (env|maestro)", line):
                out.append("<IMPORT>")
                continue
            out.append(line)
        return "\n".join(out)

    for env_p, src_p in pairs:
        a = _norm((REPO / env_p).read_text())
        b = _norm((REPO / src_p).read_text())
        assert a == b, f"{env_p} 与 {src_p} 漂移"


def test_junction_notes_verbatim_in_production():
    """driver 内联的 junction 提示语必须逐字存在于生产源文件里
    (loop.py 是重写件,靠字符串锁)。"""
    prod = (REPO / "src/maestro/pipeline/window_loop.py").read_text()
    loop_src = (REPO / "rl/env/loop.py").read_text()
    for needle in ("硬切换场:背景已变,本镜是全新构图",
                   "缝合策略:本镜首帧已由派生帧给定",
                   "HARD CUT: the background changed",
                   "STITCH strategy: the opening frame is given by a",
                   "the OFFICIAL look of background",
                   "画面从", "所示的首帧继续。"):
        assert needle in prod, f"生产缺失: {needle[:20]}"
        assert needle in loop_src, f"rl/env/loop.py 缺失: {needle[:20]}"
