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
        self._logits_kw = "?"     # "?" = 未探测;None = 探测失败(退全量)

    # ── 装配(真实运行路径;测试走 __init__ 注入假件)───────────────
    @classmethod
    def load(cls, hp, device="cuda:0", adapter_path=None, train=False):
        """train=True 时开梯度检查点 —— 这是 14B/40 层能不能塞进一张卡的
        分水岭:7000 token 的反向激活约 50GB,开了之后只存层边界,~4GB。
        代价是反向多算约 30%,而训练器本来就在空等视频生成,近乎白送。"""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hp.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            hp.base_model, torch_dtype=torch.bfloat16,
            device_map={"": device})
        if train:
            base.config.use_cache = False       # 与检查点互斥,必须关
            # PEFT + 梯度检查点的静默陷阱:检查点段的输入若不 require_grad,
            # 反向【无声无息地不回传】,梯度恒为零 —— 我们栽过一次,不能再栽
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        if adapter_path:                    # 采样副本:载入现役 adapter
            model = PeftModel.from_pretrained(base, str(adapter_path),
                                              is_trainable=False)
        else:                               # 训练器:全新 LoRA(B 零初始化)
            model = get_peft_model(base, LoraConfig(
                r=hp.rank, lora_alpha=hp.alpha,
                target_modules=list(hp.target_modules),
                task_type="CAUSAL_LM"))
        if train and getattr(hp, "grad_checkpoint", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[policy] 梯度检查点已开(use_reentrant=False)", flush=True)
        return cls(model, tok, hp, device)

    # ── ① 权威 token 化:采样与训练的唯一入口 ─────────────────────
    def encode_prompt(self, prompt: str) -> list:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
            enable_thinking=self.hp.enable_thinking)
        return list(ids[0]) if hasattr(ids, "shape") and len(
            getattr(ids, "shape", ())) == 2 else list(ids)

    # ── ② logits 截断:实测参数名,不靠版本号猜 ─────────────────────
    def logits_kwarg(self):
        """哪个参数名能真的把 logits 截短?—— 用一次小探针【看输出形状】
        判定,而不是查版本号。

        起因是一个静默事故:参数在 transformers 4.49 前后由
        num_logits_to_keep 改名为 logits_to_keep,而 forward 带 **kwargs,
        旧名字被【无声吞掉】—— 不报错、不告警,只是默默把全序列 logits
        都算出来。7000 token × 151936 词表,光这一项就白烧 15GB。
        形状探测对未来任何一次改名都免疫。"""
        if self._logits_kw != "?":
            return self._logits_kw
        import torch
        pad = self.tok.pad_token_id or self.tok.eos_token_id or 0
        probe = torch.tensor([[pad] * 4], device=self.device)
        self._logits_kw = None
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                with torch.no_grad():
                    out = self.model(probe, **{kw: 2})
                if int(out.logits.shape[1]) == 2:
                    self._logits_kw = kw
                    break
            except Exception:
                continue
        print(f"[policy] logits 截断参数 = {self._logits_kw}"
              + ("" if self._logits_kw else
                 "  ⚠️ 两个名字都不生效 —— 将计算全序列 logits,显存会高"),
              flush=True)
        return self._logits_kw

    # ── ③ 逐 token logprob(三处共用)──────────────────────────────
    def seq_logprob(self, prompt_ids, response_ids, temperature,
                    grad: bool = False):
        """→ Tensor[R]:回复每个 token 在【温度缩放后】分布下的 log 概率。

        两处显存讲究:
          · 只算末尾 R+1 个位置的 logits(要预测第 t 个回复 token,需要
            位置 P-1+t 的 logits);切片用【负索引】写,这样即便截断参数
            失效,取到的窗口依然正确 —— 正确性不依赖那个 kwarg;
          · 用 logsumexp 而非 log_softmax:后者会额外物化一整份
            (R, 词表) 张量并在反向再留一份缓冲。"""
        import torch
        ids = torch.tensor([list(prompt_ids) + list(response_ids)],
                           device=self.device)
        R = len(response_ids)
        kw = self.logits_kwarg()
        call = {kw: R + 1} if kw else {}
        ctx = nullcontext() if grad else torch.no_grad()
        with ctx:
            out = self.model(ids, use_cache=False, **call)
            # 负索引窗口:截断生效时等价于 [:, :-1],失效时也仍然对
            logits = out.logits[:, -(R + 1):-1].float() / max(1e-6,
                                                             temperature)
            tgt = torch.tensor([list(response_ids)],
                               device=logits.device).unsqueeze(-1)
            picked = logits.gather(-1, tgt).squeeze(-1)
            return (picked - logits.logsumexp(-1)).squeeze(0)

    # ── ③′ 梯度自检:防"梯度恒为零"复发 ────────────────────────────
    def selftest_grad(self) -> float:
        """跑一次极小的前向-反向,返回可训练参数的梯度范数。

        存在的理由:PEFT + 梯度检查点若少了 enable_input_require_grads,
        反向会【静默】不回传,训练看着在跑、loss 在动,梯度却恒为 0。
        这个坑我们踩过一次,代价是整轮训练作废,所以开跑前必须验一次。"""
        pad = self.tok.pad_token_id or self.tok.eos_token_id or 0
        lp = self.seq_logprob([pad] * 8, [pad] * 4, 1.0, grad=True)
        lp.sum().backward()
        tot = 0.0
        for p in self.trainable_parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().float().pow(2).sum())
            p.grad = None
        return tot ** 0.5

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
