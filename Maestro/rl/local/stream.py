"""一条 rollout 流(2026-08-21 用户裁决:本地推理 + 多流并行)。

一条流 = 一个进程 = 一张卡:持一份基座 + 现役 LoRA,只做采样;
组写进队列给训练器,adapter 只在【镜与镜之间】重载。

它把三个钩子交给 rl/env/loop.py 的 driver —— driver 本身一行未改其
既有行为(不给钩子就走原来的 vLLM 路径)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .broadcast import AdapterSubscriber, GroupQueue
from .config import LIVE_ADAPTER, RL, REPO

sys.path.insert(0, str(RL))

from env.logging_utils import brain_log                        # noqa: E402
from env.skills import extract_json                            # noqa: E402


# ── 决策解析:与 window_core._brain_pick 的纪律逐条对齐 ────────────────
# (菜单越界即无效、语义字段轻校验透传、机械字段丢弃;单测锁死一致性)
def parse_decision(raw: str, menu: list, priority: list) -> dict:
    valid = {m["name"] for m in menu}
    data = extract_json(raw)
    if isinstance(data, dict) and str(data.get("strategy", "")) in valid:
        out = {"strategy": str(data["strategy"]),
               "reason": str(data.get("reason", "")), "via": "llm"}
        vp = data.get("video_prompt")
        if isinstance(vp, str) and vp.strip():
            out["video_prompt"] = vp.strip()
        if isinstance(data.get("use_prev_tail_video"), bool):
            out["use_prev_tail_video"] = data["use_prev_tail_video"]
        if isinstance(data.get("images"), list):
            imgs = [{"source": str(im.get("source", "")),
                     "description": str(im.get("description", ""))}
                    for im in data["images"][:2] if isinstance(im, dict)]
            if imgs:
                out["images"] = imgs
        return out
    # 兜底:确定性优先级(菜单里一个都不匹配时,与生产同样落 t2v 守卫)
    name = next((n for n in priority if n in valid),
                sorted(valid)[0] if valid else "t2v")
    return {"strategy": name, "via": "fallback",
            "reason": "deterministic priority (brain reply unusable)"}


def make_group_sampler(policy, hp, on_group=None):
    """→ 交给 driver 的 group_sampler(prompt, menu, k, temperature)。

    关键:兜底样本记录的是【模型真实产出的 token】,不是兜底 JSON ——
    低 format 分正好教它别再乱写。旧路径把罐头字符串当训练目标,梯度
    精确为 0,是纯粹的空转。"""
    import env.window_core as W

    def sampler(prompt, menu, k, temperature):
        samples = policy.sample_group(prompt, k=k,
                                      temp_main=hp.temp_main,
                                      temp_branch=temperature)
        variants = []
        for s in samples:
            d = parse_decision(s.text, menu, W._CONDITION_PRIORITY)
            d["decision_id"] = brain_log("window/generation-condition", {
                "label": None, "menu": sorted(m["name"] for m in menu),
                "raw": s.text, "parsed": dict(d),
                "usable": d["via"] == "llm",
                "skill": "window_generation",
                "policy_version": policy.version,
                "n_prompt_tokens": len(s.prompt_ids),
                "n_response_tokens": len(s.response_ids)})
            d["_raw"] = s.text
            d["_prompt_ids"] = s.prompt_ids
            d["_response_ids"] = s.response_ids
            d["_logp_old"] = s.logp_old
            d["_temperature"] = s.temperature
            variants.append(d)
        return variants

    return sampler


def make_shot_boundary(policy, sub: AdapterSubscriber, queue: GroupQueue,
                       hp):
    """镜间安全点:① 组入队列 ② 换脑(只在这里)。"""

    def on_boundary(record, entry, context, menu):
        samples = []
        for s in record:
            if not s.get("response_ids"):
                continue
            samples.append({
                "prompt_ids": s["prompt_ids"],
                "response_ids": s["response_ids"],
                "logp_old": s["logp_old"],
                "sample_temperature": s.get("sample_temperature"),
                "reward": s.get("reward"),
                "r_format": s.get("r_format"),
                "r_text": s.get("r_text"),
                "r_video": s.get("r_video"),
                "via": s.get("via"), "chosen": s.get("chosen")})
        if len(samples) >= 2:
            queue.put({"label": getattr(entry, "label", "?"),
                       "shot_idx": getattr(entry, "shot_idx", -1),
                       "policy_version": policy.version,
                       "worker": hp.worker_id,
                       "ts": time.time(), "samples": samples})
        v = sub.maybe_reload(policy)
        if v is not None:
            print(f"[stream{hp.worker_id}] 换脑 → adapter v{v}", flush=True)

    return on_boundary


# ── 任务池:按 worker 错开,多流不撞车 ────────────────────────────────
def pick_task(pool_path: str, it: int, workers: int, worker_id: int) -> dict:
    import yaml
    pool = yaml.safe_load(open(pool_path))
    mix = pool.get("mix", {})
    sw = int(mix.get("screenplay_weight", 3))
    iw = int(mix.get("idea_weight", 2))
    sps, ideas = pool.get("screenplays", []), pool.get("ideas", [])
    cycle = ["s"] * sw + ["i"] * iw
    # 全局序号 = 本流的第 it 次 × 流数 + 本流编号 —— N 条流互不重号
    gid = it * max(1, workers) + worker_id
    pos = gid % len(cycle)
    kind = cycle[pos]
    if kind == "i" and not ideas:
        kind = "s"
    if kind == "s" and not sps:
        kind = "i"
    n_cyc = gid // len(cycle)
    if kind == "s":
        e = sps[(n_cyc * sw + cycle[:pos].count("s")) % len(sps)]
        return {"mode": "screenplay", "file": e["file"],
                "prompt": e.get("prompt", "")}
    return {"mode": "idea",
            "prompt": ideas[(n_cyc * iw + cycle[:pos].count("i"))
                            % len(ideas)]}


def run_stream(hp, wandb_on: bool = False) -> int:
    from env.clients import CallLog
    from env.loop import build_judges, run_episode

    from .config import build_externals
    from .policy import LocalPolicy

    sub = AdapterSubscriber(LIVE_ADAPTER)
    queue = GroupQueue()
    # 起手等训练器发布 v0(它落盘后我们才有脑可用)
    for _ in range(120):
        if sub.live_version() >= 0 and (LIVE_ADAPTER / "VERSION").exists():
            break
        time.sleep(5)
    v0 = sub.live_version()
    adapter = LIVE_ADAPTER / f"v{v0}"
    policy = LocalPolicy.load(hp, device="cuda:0",
                              adapter_path=adapter if adapter.exists()
                              else None)
    policy.version = v0
    print(f"[stream{hp.worker_id}] 起手 adapter v{v0};"
          f"thinking={hp.enable_thinking} top_p={hp.top_p} "
          f"K={hp.group}", flush=True)

    log_dir = RL / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    it = 0
    while True:
        task = pick_task(hp.task_pool, it, hp.workers, hp.worker_id)
        # run 目录带 worker 后缀 —— 多流同秒启动也不会撞名
        run_dir = (Path(hp.out_root)
                   / f"{time.strftime('movie_%Y%m%d_%H%M%S')}"
                     f"_w{hp.worker_id}")
        call_log = CallLog(run_dir / "env_calls.jsonl")
        frozen, video_gen, image_edit, mllm, models = build_externals(
            hp.env_config, call_log)
        judges = build_judges(models, log_dir / "judge_calls.jsonl")

        screenplay = None
        if task["mode"] == "screenplay":
            p = REPO / task["file"]
            screenplay = str(json.loads(p.read_text()).get("content", "")
                             ).strip() or None
        print(f"[stream{hp.worker_id}] rollout #{it} {task['mode']} "
              f"{task['prompt'][:32]!r} → {run_dir.name} "
              f"(policy v{policy.version})", flush=True)
        try:
            res = run_episode(
                task_text=task["prompt"], screenplay=screenplay,
                run_dir=run_dir, frozen_llm=frozen, policy=policy,
                video_gen=video_gen, image_edit=image_edit, mllm=mllm,
                judges=judges, group=hp.group,
                rl_temperature=hp.temp_branch,
                # ── 三个钩子:本地推理路线的全部接入点 ──────────────
                group_sampler=make_group_sampler(policy, hp),
                ref_llm=_RefLLM(policy),          # 图计划 / 空间视图挑图
                on_shot_boundary=make_shot_boundary(policy, sub, queue, hp))
            print(f"[stream{hp.worker_id}] rollout #{it} 完成 {res}",
                  flush=True)
        except Exception as exc:
            print(f"[stream{hp.worker_id}] rollout #{it} 失败:"
                  f"{str(exc)[:300]}", flush=True)
        it += 1


class _RefLLM:
    """θ_ref 的 complete 门面:图计划与空间视图挑图吃它。
    它们是环境组件、不产生训练信号 —— 钉住它们 = 环境静止。"""

    def __init__(self, policy):
        self.policy = policy

    def complete(self, prompt, temperature=None, max_tokens=None) -> str:
        return self.policy.ref_complete(prompt, temperature=temperature,
                                        max_tokens=max_tokens)
