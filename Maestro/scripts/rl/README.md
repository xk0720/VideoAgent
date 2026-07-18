# RL 数据管道(S0)—— 用法与飞轮

> 方案全文:`docs/RL_TOOLCALLING_RESEARCH_2026_07_18.md`。本目录是 S0
> 落地:把运行日志变成训练数据 + 零成本评估。训练脚本(SFT/KTO/DPO)
> 属于 S1,在 CUDA 机上用 TRL 跑,不进本仓库依赖。

## 飞轮一圈

```
① 跑片(正常使用,无需任何额外操作)
   outputs/attemptN/ 里自动落:brain_calls.jsonl(每个决策一条,自带
   decision_id;修复判决 repair/outcome、每镜结局 window/shot_outcome
   都在同一个文件里,靠 id/label 连接)+ storyboard.json

② 建数据
   python scripts/rl/build_dataset.py outputs/attempt2 outputs/attempt3 \
       --out data/rl --holdout attempt3
   → sft.jsonl / kto.jsonl / dpo_pairs.jsonl / eval_holdout.jsonl
     / excluded.jsonl(每条被排除的决策都写明原因 —— 审计用)

③ 训练(S1,另一台机器,TRL)
   SFT 温启动 → KTOTrainer(beta=0.1, lr 5e-6, 类权重配平)→
   DPOTrainer(loss=["sigmoid","sft"], VLM 对加 label_smoothing=0.1)
   底座 Qwen3-8B + LoRA(r16, 全线性层)。

④ 评估(不花生成费;三条基线:原始底座 / 仅 SFT / 仅格式奖励)
   python scripts/rl/eval_replay.py data/rl/eval_holdout.jsonl \
       --base-url http://localhost:8000/v1 --model <被测模型> -k 4

⑤ 接回 Maestro(管线零改动)
   vllm serve <merged-model> --port 8000
   config 里 models.llm 指到它(OpenAICompatLLM 原生支持):
     models:
       llm:
         name: openai-compat
         base_url: "http://localhost:8000/v1"
         model: "<merged-model>"
   然后回到 ① —— 新日志已经是新策略的分布。
```

## 数据形状(一道题 = 一条数据)

多轮修复不会拼成长对话:每一轮单独一条,**上一轮的结局写在这一轮的
题干里**(生产时执行器就是这么出题的,训练与实战逐字符同分布;条件/
图计划的题干走 `window_loop.decision_prompt` 同源函数重建)。

标签规则(v1 保守,"不怪它的失败不进它的坏样本")与成对样本的挖法
见 `build_dataset.py` 模块 docstring。
