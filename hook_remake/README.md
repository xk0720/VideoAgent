# hook_remake — 爆款视频复刻·极简测试链路

> 一句话:把爆款视频**切镜**,按 person_hook 数量**平均分**给几位目标人物,逐镜调百炼
> `wan2.2-animate-move` 换人,再按原时间轴**拼回去**。
> 这是 [`../HookRemakeAgent_DESIGN.md`](../HookRemakeAgent_DESIGN.md) 的最小可跑子集,
> 用来先打通"切镜 → 调模型 → 拼回"这条主动脉;视频理解(镜头卡/身份聚类/take 重组)
> 团队定框架后再接入。

## 打个比方

原片是一支排练好的舞,我们不研究"谁在跳"(跳过视频理解),直接把舞切成一段段,
按顺序发给几位新演员:前三分之一给 hook_1、中间给 hook_2……每位演员**在自己照片
里的场景里**把分到的动作跳一遍(所以背景=hook 图背景),最后按原来的剪法拼回去,
配回原 BGM。

## 为什么是 animate-move 而不是 animate-mix

| | 人物 | 背景 | 用途 |
|---|---|---|---|
| `wan2.2-animate-move` 图生动作 | 来自 hook 图 | **来自 hook 图** | ✅ 本链路(需求:背景与 hook 图一致) |
| `wan2.2-animate-mix` 视频换人 | 来自 hook 图 | 来自原片 | 设计文档里 MIX 路线,本版不用 |

调用协议(上传 getPolicy→OSS→`oss://`、异步提交、轮询、直连绕代理)照搬本仓已实测的
M0 契约(`Maestro/scripts/playground/bailian_kling_probe.py`,2026-08-03 全 SUCCEEDED)。

## 快速开始

```bash
cd hook_remake
pip install -r requirements.txt        # 另需系统 ffmpeg/ffprobe
export DASHSCOPE_API_KEY=sk-xxx        # 或写进仓库根 .env

cp hooks.example.json hooks.json       # 填入你的 person_hook(建议与原片同比例的全身照)

# 第一步永远先 dry-run:只切镜+分配+排产,落 manifest.json,不花一分钱
python run_test.py --video viral.mp4 --hooks hooks.json --dry-run

# 真跑:默认只生成前 3 个镜头(控费),其余镜头用原片段占位拼出完整成片
python run_test.py --video viral.mp4 --hooks hooks.json

# 满意后全量
python run_test.py --video viral.mp4 --hooks hooks.json --limit 0 --yes
```

产物在 `outputs/run_<时间戳>/`:

```
manifest.json      # 全量台账:切镜表、分配表、逐镜任务状态、预计计费秒数
shots/             # 切出的原始镜头
padded/            # <2.1s 镜头的回文补齐驱动片段
gen/               # 模型输出
conform/           # 对齐到原时长/分辨率/fps 的片段
remake.mp4         # 成片(失败镜头自动回退原片段,永远能拼出来)
```

## 链路里每一步在干什么

1. **切镜** `splitter.py` — PySceneDetect ContentDetector 找切点(ViMax 本身没有切镜
   模块,它是生成侧框架;这里用标准方案,未安装时自动退化为 4s 等间隔切)。
   超过 28s 的镜头等分切块;**短于 2.1s 的镜头做"回文补齐"**(正放+倒放循环到 2.1s,
   因为 API 硬性要求输入 ≥2s;生成后只取与原镜头对应的前一截,动作逐帧对得上)。
2. **平均分** `assigner.py` — `sequential` 连续均分(成片=分段换人)或 `round_robin`
   逐镜轮转;超长镜头的切块跟原镜头保持同一人。
3. **生成** `bailian_animate.py` — 逐镜提交 `wan2.2-animate-move`(hook 图 + 驱动片段),
   并行 2 路,轮询 15s,结果 URL 24h 失效所以即下即存。
4. **拼回** `splitter.py::conform_clip/concat_and_mux` — 每段掐回原镜头时长、缩放
   补边到原片分辨率、统一 fps,按原顺序拼接,铺回原片音轨(BGM 卡点感就是这么来的)。

## 已知取舍(测试链路的"故意偷懒",设计文档里都有正解)

| 取舍 | 后果 | 正解(设计文档章节) |
|---|---|---|
| 不做身份聚类,按数量平均分 | 换人位置与原片人物无关,同帧多人也只换主体 | 选角表 §2.2 |
| 逐镜独立调用 | 闪切镜头回文补齐会多计费(补齐部分也按输出秒计费) | 拼带 reel §2.1/§5 |
| 背景全部来自 hook 图 | 每个镜头背景一样,原片场景丢失 | MIX 路线保原环境 §2.3 |
| 无 QC | 人脸/动作跑偏不拦截 | 质检梯子 §10 |
| 忽略 object_hook | 商品特写镜头原样保留(含原品牌画面!) | STILL 路线 §4 |
| 原片烧字会被重绘丢掉/变形 | 字幕不可读 | 字幕改写覆盖 §5 |

## 排错

- **提交 400 且报 url 相关错误**:确认请求头带了 `X-DashScope-OssResourceResolve`
  (代码已带);若你的账号/该模型不支持 `oss://` 临时资源,把素材放到公网可访问的
  OSS/CDN,`hooks.json` 与视频直接给 https URL。
- **轮询一直卡住**:本机系统代理会掐断长轮询,客户端已 `trust_env=False` 直连;
  在境外网络请把 `config.yaml` 的 `base_url` 换成 `dashscope-intl`。
- **切不出镜头**:降低 `split.threshold`(如 21),或调小 `fallback_interval_s`。
- **check_image 拒绝**:hook 图要单人、清晰、比例 1:3–3:1、最短边 ≥200px;
  官方建议参考图与原片"人物画幅占比相似"(全身舞蹈片配全身照)。
