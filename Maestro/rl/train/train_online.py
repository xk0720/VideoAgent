"""在线 GRPO trainer(LoRA + vLLM 热载)。

方法(诚实定位):组内 baseline 的 policy gradient ——
  loss = −mean_k( advantage_k · logprob(completion_k | prompt_k) )
advantage_k = r_k − mean(r_group)(GRPO 的组相对优势);completion 来自
rollout(vLLM 上同一策略),陈旧度由 policy_version 过滤控制(≤K 版),
不做 ratio clip(单步陈旧度受限时 REINFORCE 偏差可接受;超版丢弃)。

prompt 重建 = 生产同源:decision_prompt + window_generation 技能全文
(与 rollout 逐字符一致 —— 单一事实源)。

循环:tail groups_online.jsonl → 攒 batch_groups 个新组 → 一步更新 →
每 save_every 步存 LoRA adapter → POST vLLM /v1/load_lora_adapter 热载
→ bump rl/state/policy_version.txt(rollout 侧下一次调用换新 adapter 名,
并把版本写进组记录)。

依赖(训练机):pip install torch transformers peft requests
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
DATA = REPO / "rl/data/groups_online.jsonl"
STATE_DIR = REPO / "rl/state"
CKPT_DIR = REPO / "rl/ckpt"


def build_prompt(group: dict) -> str:
    from maestro.pipeline.window_loop import (_skill_body_named,
                                              decision_prompt)
    skill = _skill_body_named("window_generation")
    return decision_prompt(skill, group.get("menu") or [],
                           group.get("context") or {})


def load_new_groups(offset: int, staleness_max: int,
                    current_version: int) -> tuple[list, int]:
    if not DATA.exists():
        return [], offset
    lines = DATA.read_text(errors="replace").splitlines()
    out = []
    for line in lines[offset:]:
        try:
            g = json.loads(line)
        except Exception:
            continue
        try:
            pv = int(g.get("policy_version") or 0)
        except ValueError:
            pv = 0
        if current_version - pv > staleness_max:
            continue                       # 超陈丢弃(诚实降数据不降质)
        samples = [s for s in g.get("samples") or []
                   if s.get("raw") and s.get("reward") is not None]
        if len(samples) >= 2 and g.get("context"):
            mean = sum(s["reward"] for s in samples) / len(samples)
            for s in samples:
                s["advantage"] = s["reward"] - mean
            if any(abs(s["advantage"]) > 1e-6 for s in samples):
                out.append({**g, "samples": samples})
    return out, len(lines)


def batch_metrics(batch: list) -> dict:
    """一步更新的监测指标(2026-08-14 用户令:reward 各分项都要看)。
    None 分量按缺失剔除;advantage 统计衡量组内信号强度 —— std 塌到
    0 附近 = 判官分不出好坏,训练在空转,这是最该盯的一条曲线。"""
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    samples = [s for g in batch for s in g["samples"]]
    advs = [s.get("advantage", 0.0) for s in samples]
    m = {"reward/mean": _mean([s.get("reward") for s in samples]),
         "reward/format": _mean([s.get("r_format") for s in samples]),
         "reward/text": _mean([s.get("r_text") for s in samples]),
         "reward/video": _mean([s.get("r_video") for s in samples]),
         "advantage/std": round(
             (sum(a * a for a in advs) / max(1, len(advs))) ** 0.5, 4),
         "batch/groups": len(batch),
         "batch/samples": len(samples),
         "batch/judged_text_rate": round(
             sum(1 for s in samples if s.get("r_text") is not None)
             / max(1, len(samples)), 3),
         "batch/judged_video_rate": round(
             sum(1 for s in samples if s.get("r_video") is not None)
             / max(1, len(samples)), 3)}
    for dim in ("action", "physics", "camera", "consistency"):
        m[f"video/{dim}"] = _mean(
            [(s.get("video_detail") or {}).get(dim) for s in samples])
    return m


def wandb_init(args, enabled: bool):
    """wandb 可选依赖 + 离线优先(2026-08-14:训练服务器出网受限,
    默认 WANDB_MODE=offline 落本地,之后 wandb sync 补传;装了才用,
    没装诚实降级为纯打印,绝不因监控库缺失挡训练)。"""
    if not enabled:
        return None
    try:
        import os
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config={"model": args.model, "lr": args.lr,
                           "batch_groups": args.batch_groups,
                           "staleness_max": args.staleness_max})
        return wandb
    except Exception as exc:
        print(f"[trainer] wandb unavailable ({exc}) — metrics will "
              "only be printed", flush=True)
        return None


def hot_reload_vllm(vllm_url: str, adapter_name: str,
                    adapter_path: Path) -> bool:
    import requests
    try:
        r = requests.post(f"{vllm_url}/v1/load_lora_adapter",
                          json={"lora_name": adapter_name,
                                "lora_path": str(adapter_path)},
                          timeout=60)
        return r.status_code < 400
    except Exception as exc:
        print(f"[trainer] vLLM hot-reload failed: {exc}", flush=True)
        return False


def main() -> int:
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--batch-groups", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--staleness-max", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载模型:验证数据流/分组/advantage")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--wandb", action="store_true",
                    help="开 wandb 监控(默认 offline 模式落本地)")
    ap.add_argument("--wandb-project", default="maestro-brain-rl")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()
    DATA = Path(args.data)

    wb = wandb_init(args, args.wandb)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ver_file = STATE_DIR / "policy_version.txt"
    version = int(ver_file.read_text()) if ver_file.exists() else 0

    if args.dry_run:
        groups, _ = load_new_groups(0, 10**9, version)
        print(f"[dry-run] usable groups={len(groups)}")
        for g in groups[:3]:
            print(" ", g["run"], g["label"],
                  [round(s["advantage"], 3) for s in g["samples"]])
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    offset, step, pending = 0, 0, []
    while True:
        new, offset = load_new_groups(offset, args.staleness_max, version)
        pending.extend(new)
        if len(pending) < args.batch_groups:
            time.sleep(args.poll)
            continue
        batch, pending = pending[:args.batch_groups], \
            pending[args.batch_groups:]
        opt.zero_grad()
        total, n = 0.0, 0
        for g in batch:
            prompt = build_prompt(g)
            for s in g["samples"]:
                ids_p = tok(prompt, return_tensors="pt",
                            truncation=True, max_length=12288
                            ).input_ids.to(model.device)
                ids_c = tok(str(s["raw"]), return_tensors="pt",
                            truncation=True, max_length=1024
                            ).input_ids.to(model.device)
                ids = torch.cat([ids_p, ids_c], dim=1)
                out = model(ids).logits[:, ids_p.shape[1] - 1:-1]
                logp = torch.log_softmax(out, dim=-1).gather(
                    -1, ids_c.unsqueeze(-1)).squeeze(-1).sum()
                loss = -float(s["advantage"]) * logp \
                    / max(1, ids_c.shape[1])
                (loss / args.batch_groups).backward()
                total += float(loss.detach())
                n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        met = batch_metrics(batch)
        met["train/loss"] = round(total / max(1, n), 4)
        met["train/policy_version"] = version
        print(f"[trainer] step={step} loss={met['train/loss']} "
              f"reward={met['reward/mean']} "
              f"(fmt={met['reward/format']} text={met['reward/text']} "
              f"video={met['reward/video']}) "
              f"adv_std={met['advantage/std']} "
              f"video_dims=[act={met['video/action']} "
              f"phy={met['video/physics']} cam={met['video/camera']} "
              f"con={met['video/consistency']}]", flush=True)
        if wb is not None:
            wb.log(met, step=step)
        if step % args.save_every == 0:
            version += 1
            adapter = CKPT_DIR / f"adapter_v{version}"
            model.save_pretrained(adapter)
            ver_file.write_text(str(version))
            ok = hot_reload_vllm(args.vllm_url, f"brain-v{version}",
                                 adapter)
            (STATE_DIR / "active_adapter.txt").write_text(
                f"brain-v{version}" if ok else "")
            print(f"[trainer] saved {adapter} hot_reload={ok}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
