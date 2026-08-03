"""S0 · RL 数据管道回归(2026-07-18,用户批准方案后落地):
decision_id 贯通(brain_log 自发 id 并返回 → 决策 dict → 结局记录);
build_dataset 标签规则(保守归因 + 排除必留原因)与成对样本挖掘;
eval_replay 确定性打分。全部离线,不碰网络。"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "rl" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── decision_id 贯通 ─────────────────────────────────────────────────

def test_brain_log_embeds_and_returns_decision_id(tmp_path):
    from maestro.logging_utils import brain_log, set_brain_log

    logf = tmp_path / "b.jsonl"
    set_brain_log(logf)
    try:
        did = brain_log("window/test", {"usable": True, "parsed": {}})
    finally:
        set_brain_log(None)
    assert isinstance(did, str) and len(did) == 16
    rec = json.loads(logf.read_text().splitlines()[0])
    assert rec["decision_id"] == did


def test_decide_llm_path_carries_decision_id(tmp_path):
    from maestro.logging_utils import set_brain_log
    from maestro.pipeline.window_loop import _decide

    class _LLM:
        def complete(self, prompt, **kw):
            return json.dumps({"strategy": "t2v", "reason": "r"})

    set_brain_log(tmp_path / "b.jsonl")
    try:
        d = _decide(_LLM(), "generation-condition", [{"name": "t2v"}],
                    {"shot": {"label": "scene 1 shot 1"}},
                    replay_hint=None, priority=["t2v"])
    finally:
        set_brain_log(None)
    assert d["via"] == "llm" and d.get("decision_id")


# ── build_dataset:合成 run 端到端 ────────────────────────────────────

def _write_run(tmp_path):
    run = tmp_path / "attemptX"
    run.mkdir()
    ctx = {"shot": {"label": "scene 1 shot 1"}, "slots_by_strategy": {}}
    recs = [
        # 条件决策:可用 + 零修复收敛 → 好
        {"stage": "window/generation-condition", "label": "scene 1 shot 1",
         "decision_id": "d1", "usable": True, "via": "llm",
         "menu": ["t2v", "extend_prev"], "context": ctx,
         "raw": "{}", "parsed": {"strategy": "extend_prev", "reason": "x"},
         "skill_chars": 1},
        # 条件决策:解析失败 → 坏
        {"stage": "window/generation-condition", "label": "scene 1 shot 2",
         "decision_id": "d2", "usable": False, "menu": ["t2v"],
         "context": ctx, "raw": "garbage", "parsed": None, "skill_chars": 1},
        # episode 重放 → 排除
        {"stage": "window/generation-condition", "label": "scene 1 shot 3",
         "decision_id": "d3", "usable": True, "via": "episode",
         "menu": ["t2v"], "context": ctx, "parsed": {"strategy": "t2v"}},
        # enhancer:attempt0 引用未知槽位被拒 → 坏 + 成对(与 attempt1)
        {"stage": "window/prompt_enhance", "label": "scene 1 shot 1",
         "decision_id": "d4", "usable": False, "attempt": 0,
         "ref_audit": {"ok": False, "unknown": ["@Image9"]},
         "context": {"shot_description": "s", "conditions": []},
         "raw": "bad @Image9 prompt", "skill_chars": 1},
        {"stage": "window/prompt_enhance", "label": "scene 1 shot 1",
         "decision_id": "d5", "usable": True, "attempt": 1,
         "ref_audit": {"ok": True, "unknown": [], "appended": []},
         "context": {"shot_description": "s", "conditions": []},
         "raw": "good prompt", "parsed": {"video_prompt": "good prompt"},
         "skill_chars": 1},
        # 修复决策:t 轮被拒、t+1 轮被收 → 一好一坏 + 修复对
        {"stage": "repair/decide", "shot_idx": 1, "decision_id": "d6",
         "usable": True, "via": "llm", "menu": ["regenerate"],
         "context": {"k": 1}, "tools_menu": [{"name": "regenerate"}],
         "raw": json.dumps({"tool": "regenerate_segment"}),
         "parsed": {"tool": "regenerate_segment", "via": "llm"},
         "skill_chars": 1},
        {"stage": "repair/outcome", "decision_id": "d6", "shot_idx": 1,
         "outcome": "rejected"},
        {"stage": "repair/decide", "shot_idx": 1, "decision_id": "d7",
         "usable": True, "via": "llm", "menu": ["regenerate"],
         "context": {"k": 2}, "tools_menu": [{"name": "regenerate"}],
         "raw": json.dumps({"tool": "regenerate"}),
         "parsed": {"tool": "regenerate", "via": "llm"}, "skill_chars": 1},
        {"stage": "repair/outcome", "decision_id": "d7", "shot_idx": 1,
         "outcome": "accepted"},
        # 每镜结局
        {"stage": "window/shot_outcome", "label": "scene 1 shot 1",
         "shot_idx": 0, "converged": True, "repair_turns": 0},
        {"stage": "window/shot_outcome", "label": "scene 1 shot 2",
         "shot_idx": 1, "converged": False, "repair_turns": 2,
         "stop_reason": "turns_exhausted"},
        {"stage": "window/shot_outcome", "label": "scene 1 shot 3",
         "shot_idx": 2, "converged": True, "repair_turns": 0},
    ]
    with (run / "brain_calls.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (run / "storyboard.json").write_text(json.dumps({"entries": []}))
    return run


def test_build_dataset_labels_pairs_exclusions(tmp_path):
    bd = _load_script("build_dataset")
    run = _write_run(tmp_path)
    got = bd.build_run(run)
    by = {}
    for s in got["samples"]:
        by.setdefault((s["meta"]["stage"], s["label"]), []).append(s)

    # 条件:好(零修复收敛)与坏(解析失败)各一
    assert len(by.get(("condition", True), [])) == 1
    assert len(by.get(("condition", False), [])) == 1
    good = by[("condition", True)][0]
    assert good["meta"]["why"] == "shot converged with zero repairs"
    # completion 是规范化 JSON 且不带 decision_id/via
    comp = json.loads(good["completion"][0]["content"])
    assert comp == {"strategy": "extend_prev", "reason": "x"}
    # prompt 由生产同源函数重建,含菜单与上下文
    ptxt = good["prompt"][0]["content"]
    assert "THIS TURN (JSON)" in ptxt and "extend_prev" in ptxt

    # enhancer:拒 → 坏;attempt1 好(零修复收敛镜)
    assert len(by.get(("enhance", False), [])) == 1
    assert "@Image9" in by[("enhance", False)][0]["meta"]["why"]
    assert len(by.get(("enhance", True), [])) == 1

    # 修复:accepted → 好,rejected → 坏
    labels = {s["meta"]["decision_id"]: s["label"]
              for s in got["samples"] if s["meta"]["stage"] == "repair"}
    assert labels == {"d6": False, "d7": True}

    # episode 重放被排除且写明原因
    excl = {e["decision_id"]: e["why"] for e in got["excluded"]
            if e.get("decision_id")}
    assert "via=episode" in excl["d3"]

    # 成对:只剩 enhancer 重试对(2026-08-02 用户指正:修复相邻两轮之间
    # 执行过真生成、上下文已变 —— 拒-收对是分布外假同题,已废除;被拒/
    # 被采纳的修复决策各按自己的题干走 KTO 单条,上面 d6/d7 正是)。
    kinds = sorted(p["meta"]["kind"] for p in got["pairs"])
    assert kinds == ["enhancer_retry"]


def test_build_dataset_cli_holdout_split(tmp_path, capsys):
    bd = _load_script("build_dataset")
    run = _write_run(tmp_path)
    out = tmp_path / "data"
    assert bd.main([str(run), "--out", str(out),
                    "--holdout", run.name]) == 0
    hold = (out / "eval_holdout.jsonl").read_text().strip().splitlines()
    assert len(hold) > 0
    assert (out / "kto.jsonl").read_text().strip() == ""   # 全部划走
    assert (out / "excluded.jsonl").read_text().strip() != ""


def test_legacy_run_without_outcome_records_excludes_honestly(tmp_path):
    bd = _load_script("build_dataset")
    run = tmp_path / "legacy"
    run.mkdir()
    rec = {"stage": "window/generation-condition", "label": "scene 1 shot 1",
           "usable": True, "via": "llm", "menu": ["t2v"],
           "context": {"shot": {}}, "raw": "{}",
           "parsed": {"strategy": "t2v"}, "skill_chars": 1}
    (run / "brain_calls.jsonl").write_text(json.dumps(rec) + "\n")
    got = bd.build_run(run)
    assert got["samples"] == []
    assert "no shot outcome record" in got["excluded"][0]["why"]


# ── eval_replay:确定性打分 ──────────────────────────────────────────

def test_eval_replay_scores_with_stub_model(tmp_path):
    bd = _load_script("build_dataset")
    er = _load_script("eval_replay")
    run = _write_run(tmp_path)
    samples = [s for s in bd.build_run(run)["samples"]
               if s["meta"]["stage"] == "condition" and s["label"]]

    good_reply = json.dumps({"strategy": "extend_prev", "reason": "r"})
    res = er.run_eval(samples, lambda p: good_reply, k=3)
    st = res["per_stage"]["condition"]
    assert st["parse_ok"] == 1.0 and st["in_menu"] == 1.0
    assert st["agree"] == 1.0 and st["pass_k"] == 1.0

    res_bad = er.run_eval(samples, lambda p: "not json", k=2)
    st_bad = res_bad["per_stage"]["condition"]
    assert st_bad["parse_ok"] == 0.0 and st_bad["pass_k"] == 0.0


def test_synthetic_runs_feed_builder(tmp_path):
    """合成日志生成器与数据构建器闭环:非空 sft/kto/dpo,格式齐。"""
    import random

    msr = _load_script("make_synthetic_runs")
    bd = _load_script("build_dataset")
    rng = random.Random(1)
    n = msr.make_run(tmp_path / "run_00", rng, n_shots=4)
    assert n > 10
    got = bd.build_run(tmp_path / "run_00")
    labels = [s["label"] for s in got["samples"]]
    assert True in labels and False in labels          # 好坏都有
    kinds = {p["meta"]["kind"] for p in got["pairs"]}
    assert kinds == {"enhancer_retry"}
    s0 = got["samples"][0]
    assert s0["prompt"][0]["role"] == "user"
    assert s0["completion"][0]["role"] == "assistant"
