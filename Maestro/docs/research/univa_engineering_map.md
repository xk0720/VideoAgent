# UniVA 工程图谱（本地 `VideoAgent/univa/`，2026-06-11 阅读）

> 目的：Maestro "链路跑通"参考。UniVA = agno (Agent 框架) + FastMCP stdio 工具服务
> + FastAPI/SSE 服务端 + Next.js 前端；生成全部走 WaveSpeed REST API。

## 1. 控制流（核心连通模式）

- 入口：CLI `univa/univa_agent.py:508 main()`；HTTP `univa_server.py` `POST/GET /chat/stream` → SSE 流。
- `PlanActSystem`（univa_agent.py:329）持有一个 `MultiMCPTools`（所有工具服务的统一函数调用面）+ `SqliteDb`。
- **PlanAgent**：无工具绑定，只靠 `prompts/plan.txt` 里手抄的工具描述产出 ```json``` 计划：
  `{task_analysis, execution_plan:{total_steps, steps:[{step_number, action_description, tool:{name,purpose,input_requirements}, dependencies, status, output}]}}`。
- **ActAgent**：绑定全部 MCP 工具，`tool_call_limit=15`；逐步执行 —— 每步的 prompt = 用户请求 + 整个计划
  + 已完成步骤结果 dict + 当前步骤；LLM 自己决定调用哪个 MCP 工具；其末尾 JSON
  `{success,message,content,output_path}` 被正则抽取，`update_plan` 把 `output_path` 写回计划。
- **关键连通模式**：计划是 Python 里的 JSON dict；**工件 = 本地文件路径字符串**，通过
  步骤结果 JSON 在 LLM 上下文里传递（无对象存储、无 ID 系统）。步骤严格顺序，
  `dependencies` 字段只是描述、代码不强制。

## 2. 工具层

6 个 FastMCP stdio 服务（`mcp_tools/*.py`，`@mcp.tool()`，`mcp_configs.json` 注册）：
video_gen（t2v/storyboard 工作流/i2v/续帧/首尾帧/拼接）、image_gen（t2i/i2i/序列一致编辑）、
video_editing（深度替换/姿态参考/风格化/重绘，runway 或本地 VACE）、video_understanding
（帧 base64 → GPT 描述问答）、video_tracking（本地 Sa2VA 指代分割，每次调用加载模型）、
audio_gen（mmaudio-v2 配音、minimax TTS）。
所有工具返回 `ToolResponse{success,message,content,output_path}`（mcp_tools/base.py:7）。

## 3. 记忆（README 宣称 3 层，代码只有 2 个持久化点）

1. agno SqliteDb：Plan agent 的对话历史（近 10 条）+ `session_state["execution_history"]`
   （每任务结束 append {plan, execution_results}，下次规划时注入 prompt）。
2. 用户/鉴权 JSON（access code 计数）。前端 Postgres 只存 auth。
**没有创作记忆、没有实体/偏好/技能记忆。**

## 4. 模型后端

- Plan/Act LLM：`utils/model_factory.py` → agno 各家模型类（OpenAI/Claude/DeepSeek/Gemini…），
  `.env` 的 `PLAN_MODEL_*` / `ACT_MODEL_*`。
- 工具侧 LLM：`utils/query_llm.py` 原生 requests；多模态 = decord 抽帧 base64。
- **生成全部 WaveSpeed API**（`utils/wavespeed_api.py`，单一 `WAVESPEED_API_KEY`）：
  flux-kontext-pro、seedream-v4、seedance-v1-pro t2v/i2v 480p、wan-2.1-vace、hailuo-02、
  runway gen4-aleph、mmaudio-v2、minimax speech。模式：POST 提交 → 轮询 → 下载到 `results/`。
- 本地 GPU 可选（Sa2VA、VACE 子进程，作者机器硬编码路径）。

## 5. 最小端到端（text→video，无 GPU）

`pip install -r requirements_simple.txt && pip install -e .`；`.env` 配
`PLAN_MODEL_API_KEY`、`ACT_MODEL_API_KEY`、`LLM_OPENAI_API_KEY`、`WAVESPEED_API_KEY`、
`AUTH_ENABLED=False`。t2v 走 seedance/WaveSpeed，**不需要本地模型**。

## 6. 它没有的东西（= Maestro 创新空间，逐条对应我们的方案）

- **无输出验证**：`success` = "API 返回了 URL"，没有任何机制看过生成的视频好不好。
  （aspirational 的 `director_check.txt` prompt 存在但代码从未引用。）
- **无任务内重试/重规划**：首个失败步骤直接 abort 整个任务；"动态调整计划"只能在
  *下一个用户轮次*发生（execution_history 注入）。
- **无技能习得**：工具静态注册；workflow（如 storyvideo_gen 的分镜→角色→关键帧→i2v）
  是硬编码 Python 函数，不是可蒸馏/可进化的技能。
- **无创作记忆**：跨任务只有"上次的计划+结果"文本注入。

## 7. Maestro 借鉴清单（连通性，不是创新）

1. **工件总线**：步骤结果里的本地文件路径字符串 + 结构化 step result —— Maestro 的
   plan→act→critic 链采用同样约定（`ToolResponse` 等价物已有 BaseTool 契约）。
2. **云 API 生成后端**：给 `models/video_gen_backends.py` 加 WaveSpeed 风格的 API 后端
   （提交→轮询→下载），让"真实链"不依赖本地 GPU —— D7 最小真实链的最快路径。
3. **结果注入式跨轮记忆**：execution_history 注入 prompt 的轻量做法可以做 baseline，
   我们的 lesson/skill/memory 与之对比。
4. 防踩坑：success 字符串化（"True" vs True）、正则抽 JSON 的脆弱性、每调用加载大模型、
   硬编码路径 —— Maestro 全部用类型化契约避免。
