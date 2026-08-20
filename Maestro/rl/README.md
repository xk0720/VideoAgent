# rl/ — brain(generation-condition)semi-online GRPO

**自包含纪律(2026-08-19 用户令)**:rl/ 不调用文件夹以外的包 ——
训练所需的一切(agent loop、客户端、判官、prompt 拼装)全部在本目录;
主管线 src/maestro 里没有任何 RL 挂点。唯一的跨目录读取是【数据文件】:
brain 技能文本(src/maestro/skills/brain_skills/*/SKILL.md)按文件原位
读取,保证策略的 prompt 分布与生产逐字符一致、永不漂移。

**一行命令**:
```
bash rl/run_grpo.sh          # GPU 机:vLLM(Qwen3+LoRA 热载)+ rl/env
                             # rollout 农场 + 收集 + 训练
bash rl/run_grpo.sh --fresh  # 真·全新开跑(清台账/书签/ckpt,落开跑标记)
bash rl/run_grpo.sh --stop   # 一键收摊(农场→rollout→vLLM→GPU 残留)
bash rl/run_grpo.sh --smoke  # 本机自检:mock rollout→收集→分组→advantage
```

**同构纪律(2026-08-19 用户令)**:训练 loop 与生产 loop 完全一样。
rl/env 的生成路径是 src/maestro 的【逐字移植】——window_core.py(窗口
函数原文)、storyboard/space_bible/ref_slots/junction_stitcher/
logging_utils/language(整文件拷贝,仅 import 行改 shim);
tests/unit/test_rl_env_parity.py 在 CI 锁两边不漂移(行为锁+源文锁)。
与生产的差异仅限用户明令三点:①每镜 K 组采样;②skill 判官择主干
(评审板/锦标赛/修复不进 RL);③enhancer/episode/BGM/转场关。

- DESIGN.md                 设计全文(v1 + v2 增补)
- env/rollout.py            RL rollout 入口(一次调用 = 一条轨迹)
- env/loop.py               driver:生产 generate_movie_windowed 的移植
                            (§A0→空间圣经→三叉分诊 derive/cut/continue
                            →image plan→K 组采样→判官择主干→§E 拼接)
- env/window_core.py        生产 window_loop 生成路径函数的逐字移植件
- env/storyboard.py 等      整文件移植件(见同构纪律)
- env/skills.py             策略 prompt 唯一事实源(env 采样与 trainer
                            重建共用;decision_prompt = 生产同款拼法)
- env/clients.py            生产接口面客户端(冻结 LLM/策略 vLLM/可灵/
                            wavespeed t2i+图像编辑/EnvVLM 图注)
- env/config.py             迷你 dotenv/yaml
- reward/judges.py          skill 判官(文本 4 维 + 三路视频排名 +
                            一致性对照;JudgeLog 全量留痕)
- reward/skills/            判官技能(英文;只进 RL,绝不进生产)
- collect/watch_online.py   收集器 = 纯聚合(rl_steps → groups_online)
- train/train_online.py     在线 trainer(组相对优势 PG + LoRA + 热载)
- data_gen/build_trainset.py 训练集构建器(100 条,ViMax benchgen 法)
- configs/server_bailian_qwen.yaml  服务器 RL 配置(唯一被引用的配置)

要点:评审在【采样端】——每镜 K=4 个候选由 skill 判官(文本判官 +
action/physics/camera 排名 + 一致性对照)打分,argmax 即主干,reward
写进组记录;无修复(生产的评审板/critic/修复循环不进 RL 环境);
enhancer 关(brain 学写终稿);非训练 agent 全部冻结在 models.llm。

## 训练服务器约束(2026-08-19 现状)
- 服务器 GPT/Google 直连不可达:冻结 agent + 文本判官 = qwen3.8-max
  (专属 MaaS 网关,enable_thinking:false 关思考);视频判官 =
  gemini-3.1-pro-preview(idealab 内部网关,原生 video_url);
  视频 = 百炼可灵;t2i = wavespeed。
- key 只认 Maestro/.env(脚本先 unset 壳残留再 source,不互补):
  DASHSCOPE_API_KEY / QWEN_API_KEY / WAVESPEED_API_KEY / IDEALAB_API_KEY。
- 本地策略权重:`BASE_MODEL=/data/models/Qwen3-32B bash rl/run_grpo.sh`
  —— 一个旋钮同时喂 vLLM 与 trainer;GPU 划分:VLLM_GPUS=0,1,2,3
  TRAIN_GPUS=4,5,6,7(或直接 bash rl/run_grpo_8gpu.sh)。
  LoRA adapter 产出在 rl/ckpt/adapter_vN(热载给 vLLM,幅面几十 MB)。
