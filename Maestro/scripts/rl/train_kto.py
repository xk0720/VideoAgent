#!/usr/bin/env python
"""S1-② KTO 主训:全量不成对样本(kto.jsonl,{prompt, completion,
label}),好样本推高、坏样本压低,不需要成对。建议接在 SFT 产物之后
(--adapter 指向 SFT 输出目录继续训)。

token mask:KTO 天生只在 completion 上算 logprob(题干只当条件),
无需手写;题干超长按 keep_end 从左截,保住 THIS TURN JSON。
类不平衡:脚本按数据比例自动配平 desirable/undesirable 权重(官方
建议加权积比 1:1 ~ 4:3)。

用法:
    python scripts/rl/train_kto.py [--adapter outputs/rl_adapters/sft]
"""
import argparse

from train_common import (REPO, build_model_and_tokenizer, load_cfg,
                          load_jsonl_dataset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", default="", help="继续训的 LoRA(如 SFT 产物)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    from trl import KTOConfig, KTOTrainer

    model, tok, peft_cfg = build_model_and_tokenizer(cfg)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter,
                                          is_trainable=True)
        peft_cfg = None                  # 已带 adapter,不再新建
    ds = load_jsonl_dataset(cfg, "kto")
    n_good = sum(1 for x in ds["label"] if x)
    n_bad = len(ds) - n_good
    # 官方口径:desirable_weight*n_good : undesirable_weight*n_bad ∈ 1:1~4:3
    des_w, und_w = 1.0, 1.0
    if n_good and n_bad:
        if n_good > n_bad:
            und_w = min(1.33, max(1.0, n_good / n_bad))
        else:
            des_w = min(1.33, max(1.0, n_bad / n_good))
    print(f"[kto] good={n_good} bad={n_bad} → desirable_weight={des_w:.2f} "
          f"undesirable_weight={und_w:.2f}")
    seq, k = cfg["seq"], cfg["kto"]
    train_cfg = KTOConfig(
        output_dir=str(REPO / cfg["output_dir"] / "kto"),
        beta=float(k["beta"]),
        desirable_weight=des_w, undesirable_weight=und_w,
        max_prompt_length=int(seq["max_prompt_length"]),
        max_completion_length=int(seq["max_completion_length"]),
        truncation_mode=str(seq["truncation_mode"]),   # keep_end:从左截题干
        learning_rate=float(k["learning_rate"]),
        num_train_epochs=float(k["num_train_epochs"]),
        per_device_train_batch_size=int(k["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(k["gradient_accumulation_steps"]),
        gradient_checkpointing=True, bf16=cfg["model"].get("bf16", True),
        logging_steps=5, save_strategy="epoch", seed=int(cfg["seed"]),
        report_to="none",
    )
    trainer = KTOTrainer(model=model, args=train_cfg, train_dataset=ds,
                         processing_class=tok,
                         **({"peft_config": peft_cfg} if peft_cfg else {}))
    trainer.train()
    trainer.save_model()
    print(f"[done] KTO adapter → {train_cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
