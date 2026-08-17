# vimax_benchmark 跑测手册(用户自跑版,2026-08-13)

## 一、这套链路是什么

35 个英文故事(Type A 单人跨场景 / Type B 场景固定 / Type C 多人),
每故事**预分镜**(逐镜 first_frame + video_prompt)。链路三件套:

| 脚本 | 干什么 |
|---|---|
| `scripts/translate_benchmark.py` | 英译中 → `vimax_benchmark_zh/`(schema 原样,断点续跑) |
| `scripts/run_vimax_benchmark.py` | 中文题库 → 剧本契约 → 逐故事真跑 → 汇总表 |
| scene_write 预分镜法(rule 0) | 剧本带镜头编号时**结构照抄**,分镜只做标注不改写 |

## 二、跑法(推荐顺序)

```bash
# 1. 翻译全量(qwen/gpt 按 --config 的 llm;已翻 1 个,其余续跑即可)
python scripts/translate_benchmark.py

# 2. pilot:每型 1 个共 3 故事(36 镜,全链验证)
python scripts/run_vimax_benchmark.py --pilot

# 3. 验收 pilot 后放全量(35 故事 ≈ 420 次可灵调用,注意预算)
python scripts/run_vimax_benchmark.py

# 中断后续跑:直接重跑同一条命令 —— 断点以【盘上文件名】为准:
#   <story>*/movie.mp4 存在 = 完成跳过;半截目录原地保留,重跑进
#   <story>_r2 新目录;summary.json 丢了会按盘自动补记。
# 只跑指定故事 / 开评审 / 强制重跑:
#   --only chef_international_kitchens_typeA   --review   --redo
```

## 三、产物

```
outputs/benchmark/               # 固定目录(换批次 --out-root 另指)
├── summary.json                 # 登记簿(丢了可按盘上文件自动补记)
├── <story>.screenplay.json      # 喂给框架的剧本契约(可核对适配)
└── <story>[_rN]/                # run 目录(movie.mp4 在即算完成)
```

## 四、注意事项

- 默认 `--no-review`(省钱);要评审分就加 `--review`;
- 翻译质量抽查:一致性关键描述(外观逐项/场景几何)必须逐项在,
  已抽查 artist_extreme_weather_typeA 合格;
- 预分镜保真核对:跑完看 `<story>/storyboard.json` 的镜数是否等于
  summary 里的 `n_shots_requested`(12)——不等即 scene_write 违反
  预分镜法,截图发我;
- 汇总表逐故事即时落盘,随时 Ctrl-C 不丢进度。
