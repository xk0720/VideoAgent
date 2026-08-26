"""本地 GRPO 训练器(2026-08-21 用户裁决)。

相对旧训练器(rl/train/train_online.py)补齐的四件事,每一件都对应一个
实测过的病灶:

  ① 训练与采样是【同一串 token】—— 组里带着 prompt_ids / response_ids,
     不再拿裸文本重新 tokenize(旧路径连 chat 模板都没套);
  ② 重要性比值 + 非对称裁剪 —— 旧的是纯 REINFORCE,单步更新没有上限,
     多流并行后更是连"采样策略=训练策略"的前提都不成立;
  ③ KL 到 θ_ref —— 关掉 adapter 就是参考策略,精确且免费;旧路径没有
     任何拉回力,策略可以一路漂到吐乱码;
  ④ 优势除以组内标准差 —— 旧的只减均值,尺度随组方差乱跳。

外加两处显存/正确性守则:
  · logits 只算 completion 位置(num_logits_to_keep),9.2GB → 0.7GB;
  · 零优势组直接跳过 —— 四个候选打平时梯度精确为 0,做 optimizer step
    毫无意义还会累积数值噪声。
"""
from __future__ import annotations

import time

from .broadcast import AdapterPublisher, GroupQueue


def group_advantage(rewards: list) -> list:
    """组内相对优势:(r − mean) / (std + 1e-6)。

    std 用无偏(n−1)估计,与 verl 的 GRPO 实现一致;不做留一法
    (样本自己也在自己的基线里)。返回长度与 rewards 相同。"""
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]                       # 单样本无基线,优势定义为 0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    std = var ** 0.5
    return [(r - mean) / (std + 1e-6) for r in rewards]


def train_one_group(policy, opt, group: dict, hp) -> dict:
    """一个组 = 一镜 4 个候选 = 一次 optimizer step。"""
    import torch

    samples = [s for s in (group.get("samples") or [])
               if s.get("response_ids") and s.get("reward") is not None]
    if len(samples) < 2:
        return {"skipped": "too_few_samples"}

    adv = group_advantage([float(s["reward"]) for s in samples])
    if max(abs(a) for a in adv) < 1e-6:
        return {"skipped": "zero_advantage"}   # 打平 → 梯度恒为 0

    total_tok = sum(len(s["response_ids"]) for s in samples)

    # 参考策略先算完(无梯度、关 adapter),避免反复切换 adapter 状态
    ref_logps = []
    with policy.ref_context():
        for s in samples:
            ref_logps.append(policy.seq_logprob(
                s["prompt_ids"], s["response_ids"],
                float(s.get("sample_temperature") or hp.temp_main),
                grad=False).detach())

    opt.zero_grad()
    m = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "ratio": 0.0,
         "clipfrac": 0.0, "n_tok": total_tok}
    for s, a, logp_ref in zip(samples, adv, ref_logps):
        temp = float(s.get("sample_temperature") or hp.temp_main)
        logp_new = policy.seq_logprob(s["prompt_ids"], s["response_ids"],
                                      temp, grad=True)
        logp_old = torch.tensor([float(x) for x in s["logp_old"]],
                                device=logp_new.device)

        # ── PPO 比值 + 非对称裁剪(clip-higher)──────────────────
        ratio = torch.exp(torch.clamp(logp_new - logp_old, -20, 20))
        pg1 = -a * ratio
        pg2 = -a * torch.clamp(ratio, 1 - hp.clip_low, 1 + hp.clip_high)
        pg = torch.maximum(pg1, pg2)

        # ── KL 到 θ_ref(k3 低方差估计量,作损失项不进奖励)────────
        delta = torch.clamp(logp_ref - logp_new, -20, 20)
        kl = torch.clamp(torch.exp(delta) - delta - 1, -10, 10)

        loss_tok = pg + hp.kl_coef * kl
        (loss_tok.sum() / max(1, total_tok)).backward()   # token-mean 累积

        with torch.no_grad():
            m["loss"] += float(loss_tok.sum())
            m["pg"] += float(pg.sum())
            m["kl"] += float(kl.sum())
            m["ratio"] += float(ratio.sum())
            m["clipfrac"] += float((pg2 > pg1).float().sum())

    gn = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),
                                        hp.grad_clip)
    opt.step()
    for k in ("loss", "pg", "kl", "ratio", "clipfrac"):
        m[k] = round(m[k] / max(1, total_tok), 5)
    # 组内奖励的【原始】标准差 —— 归一化前的那个,才有信息量:它衡量
    # "判官拉得开四个候选吗"。归一化后的优势 RMS 是常数 √((K−1)/K)
    # (K=4 时恒 0.866),之前打那个纯属同义反复,已换掉。
    rs = [float(s["reward"]) for s in samples]
    mr = sum(rs) / len(rs)
    m.update({"grad_norm": round(float(gn), 4),
              "reward_std": round((sum((r - mr) ** 2 for r in rs)
                                   / (len(rs) - 1)) ** 0.5, 4),
              "group_size": len(samples),
              "reward_mean": round(mr, 4)})
    return m


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_trainer(hp, wandb_on: bool = False) -> int:
    """训练器主循环:认领组 → 陈旧过滤 → 训练 → 按节拍广播 adapter。"""
    import json
    from collections import deque

    import torch
    from .policy import LocalPolicy

    policy = LocalPolicy.load(hp, device="cuda:0", train=True)
    opt = torch.optim.AdamW(policy.trainable_parameters(), lr=hp.lr)

    # 开跑前先验一次梯度真的在回传 —— PEFT + 梯度检查点少一步设置就会
    # 静默失效,loss 照跑、梯度恒零,整轮训练白费(2026-08 事故)
    gn0 = policy.selftest_grad()
    if not (gn0 > 0):
        raise RuntimeError(
            f"梯度自检失败:范数={gn0} —— 反向没有回传到 LoRA。"
            "多半是梯度检查点缺 enable_input_require_grads,"
            "或 target_modules 没命中任何层。")
    n_train = sum(p.numel() for p in policy.trainable_parameters())
    print(f"[trainer] 梯度自检通过(|g|={gn0:.4e});"
          f"可训练参数 {n_train/1e6:.1f}M", flush=True)

    queue = GroupQueue()
    pub = AdapterPublisher(hp)
    wb = _wandb_init(hp) if wandb_on else None

    recovered = queue.recover_stale_claims()
    if recovered:
        print(f"[trainer] 回收了 {recovered} 个未完成认领", flush=True)
    # 起手先发一版,让流有脑可用(v0 = LoRA 零初始化 ≈ 基座)
    pub.publish(policy, 0)
    print(f"[trainer] 起点 adapter 已发布 v0;lr={hp.lr} "
          f"rank={hp.rank} clip=[{1 - hp.clip_low:.2f},"
          f"{1 + hp.clip_high:.2f}] kl={hp.kl_coef} "
          f"broadcast_every={hp.broadcast_every}", flush=True)

    # 每版 adapter 的近期奖励/KL 落一行账 —— 这是"按峰值回滚"时找峰值
    # 的依据;只看终端日志的话,重启一次历史就没了
    recent_r, recent_kl = deque(maxlen=20), deque(maxlen=20)
    hist_path = pub.root / "reward_history.jsonl"

    step, skipped = 0, 0
    while True:
        group, path = queue.claim()
        if group is None:
            time.sleep(hp.poll_s)
            continue
        gv = int(group.get("policy_version", 0))
        if pub.published - gv > hp.staleness_max:
            print(f"[trainer] 丢弃陈旧组 v{gv}(当前 v{pub.published})",
                  flush=True)
            queue.done(path)
            continue

        t0 = time.time()
        m = train_one_group(policy, opt, group, hp)
        queue.done(path)
        if m.get("skipped"):
            skipped += 1
            continue

        step += 1
        m["step_s"] = round(time.time() - t0, 1)
        # 显存实况:OOM 排查全靠它,别再靠估算(单位 GB;多卡时逐卡列出)
        if torch.cuda.is_available():
            peaks = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                     for i in range(torch.cuda.device_count())]
            m["mem_peak"] = (peaks[0] if len(peaks) == 1
                             else "+".join(str(p) for p in peaks))
            m["n_tok_max"] = max(
                (len(s.get("prompt_ids") or []) + len(s.get("response_ids")
                                                      or [])
                 for s in group.get("samples") or []), default=0)
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        m["queue_depth"] = queue.depth()
        m["policy_version"] = pub.published
        m["staleness"] = pub.published - gv
        print(f"[trainer] step={step} ({m['step_s']}s) loss={m['loss']} "
              f"pg={m['pg']} kl={m['kl']} ratio={m['ratio']} "
              f"clipfrac={m['clipfrac']} r_std={m['reward_std']} "
              f"reward={m['reward_mean']} |g|={m['grad_norm']} "
              f"queue={m['queue_depth']} stale={m['staleness']} "
              f"mem={m.get('mem_peak', '-')}G tok={m.get('n_tok_max', '-')} "
              f"(skipped={skipped})", flush=True)
        if wb is not None:
            wb.log({f"train/{k}": v for k, v in m.items()
                    if isinstance(v, (int, float))}, step=step)

        recent_r.append(m["reward_mean"])
        recent_kl.append(m["kl"])
        v = pub.maybe_publish(policy, step)
        if v is not None:
            rec = {"v": v, "step": step,
                   "reward_recent": round(sum(recent_r) / len(recent_r), 4),
                   "kl_recent": round(sum(recent_kl) / len(recent_kl), 5)}
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            pub.annotate_archive(v, rec)   # 归档版内嵌同一份 reward 信息
            print(f"[trainer] 广播 adapter v{v}(step {step},"
                  f"近期奖励 {rec['reward_recent']})", flush=True)
            if pub.maybe_save_best(v, rec["reward_recent"]):
                print(f"[trainer] 🏆 新最优:v{v} 已存入 best/"
                      f"(近期奖励 {rec['reward_recent']})", flush=True)


def _wandb_init(hp):
    try:
        import os

        import wandb
        return wandb.init(
            project=os.getenv("WANDB_PROJECT", "maestro-brain-rl"),
            entity=os.getenv("WANDB_ENTITY") or None,
            config={k: getattr(hp, k) for k in
                    ("lr", "rank", "alpha", "group", "clip_low",
                     "clip_high", "kl_coef", "broadcast_every",
                     "temp_main", "temp_branch")})
    except Exception as exc:      # 诚实降级:监控挂了不影响训练
        print(f"[trainer] wandb 不可用({str(exc)[:100]})— 仅打终端",
              flush=True)
        return None
