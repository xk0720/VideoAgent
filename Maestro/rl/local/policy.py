"""LocalPolicy —— 本地策略模型(2026-08-21 用户裁决:弃用 vLLM)。

一个对象同时承担三件事,而且三件事共用同一份 tokenizer 与同一个
logprob 函数 —— 这正是它存在的理由:

  ① 采样      sample_group():一次前缀填充出 K 个候选
  ② 参考策略  ref_context():关掉 adapter 就是 θ_ref,精确且免费
  ③ 打分      seq_logprob():采样端算 old、训练端算 new/ref,同一实现

为什么这样能治我们的三个老病:
  · chat 模板漂移 —— encode_prompt() 是【唯一】的 token 化入口,采样与
    训练拿到的是同一串 id,不可能对不齐(旧路径是训练时拿裸文本重新
    tokenize,模型从未见过那串 token);
  · 温度对不上 —— seq_logprob() 强制按采样温度缩放 logits;
  · logits 显存 —— num_logits_to_keep 只算 completion 位置(9.2GB→0.7GB)。

模型与 tokenizer 可注入(测试用假件,零 GPU)。
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field


@dataclass
class Sample:
    """一个候选:文本 + 权威 token ids + 行为策略下的逐 token logprob。"""
    text: str
    prompt_ids: list          # 1-D int 列表(权威:apply_chat_template 产物)
    response_ids: list
    temperature: float
    logp_old: list = field(default_factory=list)   # 逐 token,长度 = 回复长


class LocalPolicy:
    def __init__(self, model, tokenizer, hp, device="cuda:0"):
        self.model = model
        self.tok = tokenizer
        self.hp = hp
        self.device = device
        self.version = 0          # 当前 LoRA 版本(广播订阅后递增)

    # ── 装配(真实运行路径;测试走 __init__ 注入假件)───────────────
    @classmethod
    def load(cls, hp, device="cuda:0", adapter_path=None):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hp.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            hp.base_model, torch_dtype=torch.bfloat16,
            device_map={"": device})
        if adapter_path:                    # 采样副本:载入现役 adapter
            model = PeftModel.from_pretrained(base, str(adapter_path),
                                              is_trainable=False)
        else:                               # 训练器:全新 LoRA(B 零初始化)
            model = get_peft_model(base, LoraConfig(
                r=hp.rank, lora_alpha=hp.alpha,
                target_modules=list(hp.target_modules),
                task_type="CAUSAL_LM"))
        return cls(model, tok, hp, device)

    # ── ① 权威 token 化:采样与训练的唯一入口 ─────────────────────
    def encode_prompt(self, prompt: str) -> list:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
            enable_thinking=self.hp.enable_thinking)
        return list(ids[0]) if hasattr(ids, "shape") and len(
            getattr(ids, "shape", ())) == 2 else list(ids)

    # ── ② 逐 token logprob(三处共用)──────────────────────────────
    def seq_logprob(self, prompt_ids, response_ids, temperature,
                    grad: bool = False):
        """→ Tensor[R]:回复每个 token 在【温度缩放后】分布下的 log 概率。

        num_logits_to_keep=R+1:只算末尾 R+1 个位置(要预测第 t 个回复
        token,需要位置 P-1+t 的 logits),再丢掉最后一个。这是 batch
        上不去的那堵显存墙的解法。"""
        import torch
        ids = torch.tensor([list(prompt_ids) + list(response_ids)],
                           device=self.device)
        R = len(response_ids)
        ctx = nullcontext() if grad else torch.no_grad()
        with ctx:
            out = self.model(ids, num_logits_to_keep=R + 1)
            logits = out.logits[:, :-1].float() / max(1e-6, temperature)
            logp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor([list(response_ids)], device=logits.device)
            return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)

    # ── ③ 组采样:两次调用、两次前缀填充,出 K 个候选 ────────────────
    def sample_group(self, prompt: str, k: int = None,
                     temp_main: float = None,
                     temp_branch: float = None) -> list:
        import torch
        hp = self.hp
        k = k or hp.group
        temp_main = hp.temp_main if temp_main is None else temp_main
        temp_branch = hp.temp_branch if temp_branch is None else temp_branch
        pids = self.encode_prompt(prompt)
        inp = torch.tensor([pids], device=self.device)

        out: list = []
        # v0 用主干温度、v1..K-1 用分支温度;两组各自只做一次前缀填充
        for temp, n in ((temp_main, 1), (temp_branch, max(0, k - 1))):
            if n == 0:
                continue
            with torch.no_grad():
                gen = self.model.generate(
                    inp, num_return_sequences=n, do_sample=True,
                    temperature=temp,
                    top_p=hp.top_p, top_k=hp.top_k,   # 1.0 / 0 = 不截断
                    max_new_tokens=hp.max_new_tokens,
                    pad_token_id=(self.tok.pad_token_id
                                  or self.tok.eos_token_id))
            for row in gen:
                rids = [int(x) for x in row[len(pids):]]
                out.append(Sample(text=self.tok.decode(
                    rids, skip_special_tokens=True),
                    prompt_ids=list(pids), response_ids=rids,
                    temperature=temp))
        # 行为策略下的 logprob 统一补算:与训练端同一个函数,量纲必然一致
        for s in out:
            if s.response_ids:
                s.logp_old = [float(x) for x in self.seq_logprob(
                    s.prompt_ids, s.response_ids, s.temperature)]
        return out

    # ── ④ θ_ref:关掉 adapter ─────────────────────────────────────
    @contextmanager
    def ref_context(self):
        dis = getattr(self.model, "disable_adapter", None)
        if dis is None:
            yield                        # 没挂 adapter 时本身就是 θ_ref
        else:
            with dis():
                yield

    def ref_complete(self, prompt: str, **kw) -> str:
        with self.ref_context():
            return self.complete(prompt, **kw)

    # ── ⑤ 与 env.clients.TextLLM 同签名 —— 可直接顶替 ──────────────
    def complete(self, prompt: str, temperature: float = None,
                 max_tokens: int = None) -> str:
        import torch
        pids = self.encode_prompt(prompt)
        with torch.no_grad():
            gen = self.model.generate(
                torch.tensor([pids], device=self.device),
                do_sample=(temperature or 0) > 0,
                temperature=temperature or self.hp.temp_main,
                top_p=self.hp.top_p, top_k=self.hp.top_k,
                max_new_tokens=max_tokens or self.hp.max_new_tokens,
                pad_token_id=(self.tok.pad_token_id
                              or self.tok.eos_token_id))
        return self.tok.decode([int(x) for x in gen[0][len(pids):]],
                               skip_special_tokens=True)

    # ── ⑥ 换脑(只允许在镜间安全点调用)────────────────────────────
    def reload_adapter(self, path, version: int) -> None:
        self.model.load_adapter(str(path), adapter_name="default",
                                is_trainable=False)
        try:
            self.model.set_adapter("default")
        except Exception:
            pass
        self.version = int(version)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
