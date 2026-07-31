"""三个训练脚本的共用件:配置加载、模型+LoRA 装配、数据装载。
只在训练机上运行(依赖 requirements-rl.txt);仓库测试不 import 本文件。"""
from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_cfg(path: str | None = None) -> dict:
    p = Path(path) if path else HERE / "train_config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def build_model_and_tokenizer(cfg: dict):
    """底座 + LoRA(全线性层)+ 可选 4bit。返回 (model, tokenizer,
    peft_config) —— peft_config 交给 TRL trainer,让它自己套(PEFT 模式
    下 KTO/DPO 无需单独参考模型:关掉 adapter 的底座即参考)。"""
    import torch
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    m = cfg["model"]
    kw: dict = {"torch_dtype": (torch.bfloat16 if m.get("bf16")
                                else torch.float16),
                "device_map": "auto"}
    if m.get("load_in_4bit"):
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(m["base"], **kw)
    tok = AutoTokenizer.from_pretrained(m["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lo = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lo["r"]), lora_alpha=int(lo["alpha"]),
        lora_dropout=float(lo["dropout"]),
        target_modules=lo["target_modules"], task_type="CAUSAL_LM")
    return model, tok, peft_cfg


def load_jsonl_dataset(cfg: dict, name: str):
    from datasets import load_dataset

    path = REPO / cfg["data_dir"] / f"{name}.jsonl"
    ds = load_dataset("json", data_files=str(path), split="train")
    # meta 列只做审计,不参与训练;label 列 KTO 要留
    drop = [c for c in ("meta",) if c in ds.column_names]
    return ds.remove_columns(drop) if drop else ds
