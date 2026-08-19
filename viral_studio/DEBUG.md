# Debug 手册 — Planner

所有命令都在 `viral_studio/` 下跑。key 已放在 `viral_studio/.env`（已 gitignore），
脚本会自动读，**不用再 export**。

```bash
cd /Users/kevin/Desktop/Kevin/repositories/VideoAgent/viral_studio
```

## 1. Planner：出分镜脚本（要联网、花几次 qwen 调用，无视频生成费）

```bash
python3 run_planner.py --product examples/product_pink_tee.yaml            # 3 张人物图
python3 run_planner.py --product examples/product_pink_tee.yaml --hooks 2  # 只用前 2 张
python3 run_planner.py --product examples/product_pink_tee.yaml --hooks 1  # 只用 1 张(收尾段应跳过)
python3 run_planner.py --product examples/product_pink_tee.yaml --out outputs/tmp_sb
```
产物：`outputs/sb_<时间戳>/{storyboard.json, validation.json}`
退出码：0 = 校验通过，1 = 有阻断项。

## 2. 解码：看每段真正会送进模型的 prompt（离线，不花钱）

```bash
python3 tools/decode_storyboard.py outputs/sb_20260819_074507/storyboard.json
python3 tools/decode_storyboard.py $(ls -td outputs/sb_* | head -1)/storyboard.json   # 最近一次
```

## 3. 审 prompt：导出喂给 Planner 的真实 system+user（离线，不花钱）

```bash
python3 tools/dump_planner_io.py examples/product_pink_tee.yaml
python3 tools/dump_planner_io.py examples/product_pink_tee.yaml > /tmp/planner_io.txt
```

## 4. Skill 库自检（离线）

```bash
python3 -c "
import sys;sys.path.insert(0,'.')
from studio.skill_store import SkillStore
s=SkillStore()
print('候选(服装,3人,body):',[c['skill_id'] for c in s.candidates('服装',3,'body')])
print('候选(服装,1人,ending):',[c['skill_id'] for c in s.candidates('服装',1,'ending')])
"
python3 tools/check_memory.py          # 记忆库 YAML 体检
python3 -c "
import yaml,pathlib
for p in sorted(pathlib.Path('skills').rglob('*.yaml')):
    yaml.safe_load(p.read_text(encoding='utf-8')); print('✓',p)
"                                       # skill 卡 YAML 体检
```

## 5. 只跑某一段的填空（定位填空问题）

```bash
python3 - <<'PY'
import sys, json, yaml; sys.path.insert(0,'.')
from pathlib import Path
from studio.config import load_dotenv; load_dotenv()
from studio.skill_store import SkillStore
from studio.agents.storyboard_planner import StoryboardPlanner
brief = yaml.safe_load(Path("examples/product_pink_tee.yaml").read_text(encoding="utf-8"))
store = SkillStore(); p = StoryboardPlanner(store)
seg = {"seg_id":"seg02","part":"body","skill_id":"outdoor_narration","hook_index":2}
slots, reason = p._fill(seg, store.get("outdoor_narration"), brief, 3, prior=[])
print(json.dumps(slots, ensure_ascii=False, indent=2))
PY
```

## 6. 换模型 / 换端点

```bash
VS_LLM_MODEL=qwen-plus python3 run_planner.py --product examples/product_pink_tee.yaml
VS_LLM_MODEL=qwen-max  python3 run_planner.py --product examples/product_pink_tee.yaml   # 默认
```

## 常见现象与位置

| 现象 | 看哪里 |
|---|---|
| `slots` 被写成字符串 | `studio/skill_store.py::digest` — 展示格式会被模型当成输出格式照抄 |
| 配色/场景填错位 | skill 卡的 `auto_from` 字段；由 `storyboard_planner._auto_slots` 程序注入 |
| 中文字数不达标 | skill 卡 `slots.*.min_chars/max_chars`；重试见 `_fill`（会回传上一版内容） |
| 选了不该选的卡 | `skill_store.candidates()` 的筛选条件 + `storyboard_select.md` 的规则 |
| 连接被掐(RemoteDisconnected) | `studio/llm.py::chat_json` 已带网络异常重试；重跑即可 |

## 日志

默认 INFO。想看每轮 LLM 交互细节：
```bash
python3 -c "
import logging;logging.basicConfig(level=logging.DEBUG)
" # 或在 run_planner.py 里把 level 改成 logging.DEBUG
```
