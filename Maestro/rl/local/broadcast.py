"""LoRA 广播与组队列(2026-08-21 用户裁决:文件通道)。

两条通道都走文件系统 —— 进程各占一张卡、共享不了 Python 对象;
文件的好处是崩溃可恢复、可肉眼检查、不用额外守护进程(AReaL / ART /
SkyRL 用的是同一机制)。代价是必须自己处理三件事,下面逐条落实:

  ① 原子性     先写 .tmp 再 os.replace(POSIX 原子)—— 绝不会读到半截
  ② 一次性消费 训练器先 rename 进 claimed/ 再加载 —— 崩了也不丢
  ③ 磁盘控制   adapter 只留 N 代、队列消费即删、积压设上限
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

from .config import CLAIMED_DIR, LIVE_ADAPTER, QUEUE_DIR


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)            # 原子换指针


# ── LoRA 广播 ────────────────────────────────────────────────────────
class AdapterPublisher:
    """训练器侧:每 broadcast_every 次 optimizer step 发布一版。"""

    def __init__(self, hp, root: Path = LIVE_ADAPTER):
        self.hp = hp
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.published = 0

    def maybe_publish(self, policy, step: int) -> int | None:
        if self.hp.broadcast_every <= 0 or step % self.hp.broadcast_every:
            return None
        return self.publish(policy, step)

    def publish(self, policy, step: int) -> int:
        v = step // max(1, self.hp.broadcast_every)
        dst = self.root / f"v{v}"
        stage = self.root / f".staging_v{v}"
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        policy.model.save_pretrained(str(stage))     # 只存 LoRA
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        os.replace(stage, dst)                       # 目录整体原子换入
        # VERSION 必须落在【本实例的 root】下,不能写全局常量 ——
        # 否则覆写 root 时(测试/多实验并存)指针与目录会分家
        _atomic_write_text(self.root / "VERSION", str(v))
        self.published = v
        self._prune()
        return v

    def _prune(self) -> None:
        vs = sorted((int(p.name[1:]) for p in self.root.glob("v*")
                     if p.name[1:].isdigit()), reverse=True)
        for old in vs[self.hp.keep_adapters:]:
            shutil.rmtree(self.root / f"v{old}", ignore_errors=True)


class AdapterSubscriber:
    """采样副本侧。maybe_reload 只允许在【镜与镜之间】调用 ——
    插在组采样中途会让一个组横跨两个策略版本,组内相对比较被污染,
    而且日志上看不出任何异常。"""

    def __init__(self, root: Path = LIVE_ADAPTER):
        self.root = Path(root)

    def live_version(self) -> int:
        try:
            return int((self.root / "VERSION").read_text().strip())
        except Exception:
            return 0

    def maybe_reload(self, policy) -> int | None:
        v = self.live_version()
        if v <= getattr(policy, "version", 0):
            return None
        path = self.root / f"v{v}"
        if not (path / "adapter_config.json").exists():
            return None                  # 还没落全,下一镜再来
        policy.reload_adapter(path, v)
        return v


# ── 组队列 ───────────────────────────────────────────────────────────
class GroupQueue:
    """流写、训练器读。组含 token ids(每组约 160KB),不塞 JSON。"""

    def __init__(self, qdir: Path = QUEUE_DIR, claimed: Path = CLAIMED_DIR,
                 backlog_max: int = 512):
        self.qdir = Path(qdir)
        self.claimed = Path(claimed)
        self.qdir.mkdir(parents=True, exist_ok=True)
        self.claimed.mkdir(parents=True, exist_ok=True)
        self.backlog_max = backlog_max

    # 流侧
    def put(self, group: dict) -> Path | None:
        import torch
        if len(list(self.qdir.glob("*.pt"))) >= self.backlog_max:
            return None                  # 积压封顶:训练跟不上就别再堆
        name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.pt"
        tmp = self.qdir / (name + ".tmp")
        torch.save(group, tmp)
        os.replace(tmp, self.qdir / name)     # 原子出现
        return self.qdir / name

    # 训练器侧
    def claim(self):
        """→ (group, claimed_path) 或 (None, None)。先认领再加载。"""
        import torch
        for p in sorted(self.qdir.glob("*.pt")):
            dst = self.claimed / p.name
            try:
                os.replace(p, dst)            # 认领成功 = 独占
            except OSError:
                continue                      # 被别人抢走了
            try:
                return torch.load(dst, map_location="cpu",
                                  weights_only=False), dst
            except Exception:
                dst.unlink(missing_ok=True)   # 坏文件直接丢,不卡住队列
                continue
        return None, None

    @staticmethod
    def done(path: Path) -> None:
        Path(path).unlink(missing_ok=True)

    def recover_stale_claims(self, older_than_s: float = 1800) -> int:
        """训练器崩溃重启:把久未处理的认领放回队列。"""
        n = 0
        now = time.time()
        for p in self.claimed.glob("*.pt"):
            if now - p.stat().st_mtime > older_than_s:
                os.replace(p, self.qdir / p.name)
                n += 1
        return n

    def depth(self) -> int:
        return len(list(self.qdir.glob("*.pt")))
