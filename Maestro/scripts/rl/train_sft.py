#!/usr/bin/env python
"""S1-① SFT 温启动:只喂 label=true 的样本(sft.jsonl),行为克隆
"闸门通过且结局良好"的决策 —— 先让开源底座会说我们的 JSON 方言。

token mask:数据是 {prompt, completion} 对 → completion_only_loss=True,
题干(skill+上下文,2-4k token)整体不计损失,只训决策 JSON 本身。

用法(训练机):
    pip install -r scripts/rl/requirements-rl.txt
    python scripts/rl/train_sft.py [--config scripts/rl/train_config.yaml]
"""
import argparse

from train_common import (REPO, build_model_and_tokenizer, load_cfg,
                          load_jsonl_dataset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    from trl import SFTConfig, SFTTrainer

    model, tok, peft_cfg = build_model_and_tokenizer(cfg)
    ds = load_jsonl_dataset(cfg, "sft").remove_columns(["label"])
    seq, s = cfg["seq"], cfg["sft"]
    train_cfg = SFTConfig(
        output_dir=str(REPO / cfg["output_dir"] / "sft"),
        max_length=int(seq["max_prompt_length"])
        + int(seq["max_completion_length"]),
        completion_only_loss=True,       # ← 题干不计损失(核心 mask 设定)
        packing=False,                   # 不打包:样本边界即 mask 边界
        learning_rate=float(s["learning_rate"]),
        num_train_epochs=float(s["num_train_epochs"]),
        per_device_train_batch_size=int(s["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(s["gradient_accumulation_steps"]),
        gradient_checkpointing=True, bf16=cfg["model"].get("bf16", True),
        logging_steps=5, save_strategy="epoch", seed=int(cfg["seed"]),
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=train_cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model()
    print(f"[done] SFT adapter → {train_cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
