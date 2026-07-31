#!/usr/bin/env python
"""S1-③ DPO 精修:只吃真正成对的数据(dpo_pairs.jsonl:enhancer 拒/过
重试对 confidence 1.0、修复拒→收相邻轮对 0.7)。接在 KTO 产物之后。

token mask:DPO 只在 chosen/rejected 两条 completion 上算 logprob,
题干共享且不计损失;keep_end 从左截超长题干。
防坍缩:rpo_alpha=1.0 在 chosen 上叠 NLL —— 我们的决策是短结构化
JSON,margin 拉大同时双双掉概率的坍缩来得特别快,必须带这项
(训练中盯 logps/chosen 曲线,趋降即调大)。

用法:
    python scripts/rl/train_dpo.py [--adapter outputs/rl_adapters/kto]
"""
import argparse

from train_common import (REPO, build_model_and_tokenizer, load_cfg,
                          load_jsonl_dataset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", default="", help="继续训的 LoRA(如 KTO 产物)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    from trl import DPOConfig, DPOTrainer

    model, tok, peft_cfg = build_model_and_tokenizer(cfg)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter,
                                          is_trainable=True)
        peft_cfg = None
    ds = load_jsonl_dataset(cfg, "dpo_pairs")
    seq, d = cfg["seq"], cfg["dpo"]
    train_cfg = DPOConfig(
        output_dir=str(REPO / cfg["output_dir"] / "dpo"),
        beta=float(d["beta"]),
        rpo_alpha=float(d["rpo_alpha"]),           # chosen 上的 NLL 项
        label_smoothing=float(d["label_smoothing"]),
        max_prompt_length=int(seq["max_prompt_length"]),
        max_completion_length=int(seq["max_completion_length"]),
        truncation_mode=str(seq["truncation_mode"]),
        learning_rate=float(d["learning_rate"]),
        num_train_epochs=float(d["num_train_epochs"]),
        per_device_train_batch_size=int(d["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(d["gradient_accumulation_steps"]),
        gradient_checkpointing=True, bf16=cfg["model"].get("bf16", True),
        logging_steps=5, save_strategy="epoch", seed=int(cfg["seed"]),
        report_to="none",
    )
    trainer = DPOTrainer(model=model, args=train_cfg, train_dataset=ds,
                         processing_class=tok,
                         **({"peft_config": peft_cfg} if peft_cfg else {}))
    trainer.train()
    trainer.save_model()
    print(f"[done] DPO adapter → {train_cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
