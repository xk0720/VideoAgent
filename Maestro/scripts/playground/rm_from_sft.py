#!/usr/bin/env python3
"""SFT 模型 → 奖励模型(RM)的最小完整示例(2026-08-27,面试备课用)。

核心改造一句话:扔掉 SFT 模型顶端的词表头(lm_head),换一个
hidden_size → 1 的线性层;读完 (idea + 剧本) 全文后,取【最后一个
有效 token】的隐藏向量过这个头,输出一个标量 = 预测满意度。

两种训法(本文件都实现):
  · 回归 MSE      —— 拟合归一化分数;简单,但吃"张三手松李四手紧"的
                     尺度噪声(MSE 把绝对值当真);
  · Bradley–Terry —— 只训"同 idea 下 A ≻ B"的序:L = -log σ(r_w - r_l),
                     损失只含分差,任何按人整体平移的打分尺度自动抵消。

验收 = 留出集成对判对率(RM 把人类偏好那篇打更高的比例;随机 50%)。
65–75% 即可用:人类互评一致率也就这个量级,是标签自身的噪声天花板。

直接运行:python scripts/playground/rm_from_sft.py
  —— 用一个随机初始化的微型 Llama 当"SFT 底座"(离线、CPU、秒级),
  在合成偏好数据上训几步,演示 BT 损失下降与判对率上升的全流程。
真实使用:RewardModel.from_sft("path/to/sft_ckpt") 替换微型底座即可;
  等价捷径是 HF 的 AutoModelForSequenceClassification(num_labels=1),
  多数 RLHF 框架(TRL 等)内部就是这么做的。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RewardModel(nn.Module):
    """SFT 底座 + 标量头。

    结构:AutoModel 加载 SFT 权重时【自动丢掉 lm_head】(它属于
    AutoModelForCausalLM),得到裸 Transformer 主体;标量头随机初始化,
    与主体一同微调(全参或 LoRA 皆可)。
    """

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        # bias=False:BT 损失只看分差,偏置本来就会抵消,留着只添参数
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    @classmethod
    def from_sft(cls, sft_path: str) -> "RewardModel":
        """真实入口:从 SFT checkpoint 起一个 RM。"""
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(sft_path)
        backbone = AutoModel.from_pretrained(          # 不含 lm_head
            sft_path, torch_dtype=torch.bfloat16)
        return cls(backbone, cfg.hidden_size)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B,) 标量分。

        取【最后一个有效 token】的隐藏向量:因果注意力下只有末位
        看得到全文;用 attention_mask 定位它,右侧 padding 不会混入。
        """
        h = self.backbone(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1               # (B,) 末位下标
        pooled = h[torch.arange(h.size(0)), last]          # (B, hidden)
        return self.score_head(pooled).squeeze(-1)         # (B,)


# ── 两种损失 ──────────────────────────────────────────────────────────
def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor
            ) -> torch.Tensor:
    """Bradley–Terry:P(A≻B)=σ(r_A−r_B),极大似然 = −log σ(分差)。
    只含分差 → 对逐用户的打分尺度平移免疫("只学序不学值")。"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def mse_loss(r_pred: torch.Tensor, score_norm: torch.Tensor
             ) -> torch.Tensor:
    """回归训法:score_norm 必须是【用户内 z-score】后的分数——
    即便如此,MSE 仍要拟合绝对值,残余尺度噪声全被当成信号。"""
    return F.mse_loss(r_pred, score_norm)


@torch.no_grad()
def pairwise_accuracy(rm: RewardModel, pairs: list[dict]) -> float:
    """验收指标:留出偏好对里,RM 给人类偏好方更高分的比例。
    随机基线 50%;65–75% 即可投产(≈人类互评一致率,标签噪声天花板)。"""
    hit = 0
    for p in pairs:
        rw = rm(p["chosen_ids"], p["chosen_mask"])
        rl = rm(p["rejected_ids"], p["rejected_mask"])
        hit += int((rw > rl).item())
    return hit / max(1, len(pairs))


# ── 可离线运行的端到端演示(微型随机底座 + 合成数据)────────────────────
def _demo() -> None:
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=128,
                      vocab_size=256)
    rm = RewardModel(LlamaModel(cfg), cfg.hidden_size)

    # 合成偏好对:chosen 序列偏向"高分 token 区"(id≥128),rejected 反之
    # —— 制造一个可学的规律,模拟"好剧本有可辨识的特征"
    def make(seq_bias: int, n: int = 24, L: int = 16):
        ids = torch.randint(0, 128, (n, L)) + seq_bias
        return ids, torch.ones_like(ids)

    train, heldout = [], []
    for bucket, k in ((train, 32), (heldout, 16)):
        for _ in range(k):
            cw, cm = make(128, n=1)
            rj, rjm = make(0, n=1)
            bucket.append({"chosen_ids": cw, "chosen_mask": cm,
                           "rejected_ids": rj, "rejected_mask": rjm})

    print(f"训练前 留出判对率 = {pairwise_accuracy(rm, heldout):.2f}"
          "(≈0.5,随机)")
    opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    for step in range(30):
        batch = train[(step * 8) % len(train):][:8] or train[:8]
        rw = torch.cat([rm(p["chosen_ids"], p["chosen_mask"])
                        for p in batch])
        rl = torch.cat([rm(p["rejected_ids"], p["rejected_mask"])
                        for p in batch])
        loss = bt_loss(rw, rl)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            print(f"step {step:2d}  BT loss = {loss.item():.4f}")
    print(f"训练后 留出判对率 = {pairwise_accuracy(rm, heldout):.2f}")


if __name__ == "__main__":
    _demo()
