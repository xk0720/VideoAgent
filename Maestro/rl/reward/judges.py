"""reward v3 判官层(2026-08-14 用户设计):文本 + 视频双重评审。

隔离纪律:本模块只被 RL 收集器调用,零依赖 maestro 包 —— 读 rollout
落盘的档案(组记录/候选视频/storyboard.json)打分,生产管线一行不碰。

组成:
  TextJudge   qwen 文本模型 × prompt_review 技能:单候选四维 1-5 分,
              衔接维仅同人同景时评(不适用输出 null,总分按有效维归一)
  VideoRanker omni 模型【原生视频输入】(用户令:不抽帧,直接喂视频):
              三个排名维(演技/物理/运镜),一组四段一次调用,
              展示顺序随机打乱防位置偏好,允许并列
  ConsistencyChecker omni:外观(ViMax 七条)+ 空间布局对照清单直判
              (有锚不排名 —— 四段全烂时排名会把矮子将军当好样本)

合成:
  r = 0.15·format + 0.35·r_text + 0.50·r_video
  r_video = 0.30·演技 + 0.25·物理 + 0.15·运镜 + 0.30·一致性
  排名 → 点数 [3,2,1,0]/3(并列取平均名次),一致性 = 通过项/总项。
  某判官失败 → 该分量剔除、其余权重归一化留痕;全失败 → 退 v2
  (0.5·m1 + 0.5·p1)。诚实降级,绝不用 0 分冒充评审结果。
"""
from __future__ import annotations

import base64
import json
import random
import re
import time
from pathlib import Path

import requests

_SKILL_DIR = Path(__file__).parent / "skills"
_DEFAULT_JUDGE_LOG = (Path(__file__).resolve().parents[2]
                      / "rl/logs/judge_calls.jsonl")



class JudgeLog:
    """判官全量留痕(2026-08-19 用户令):每次评审一行 JSONL ——
    谁评的(judge/model)、评的什么(tag: run/镜/候选)、回了什么
    (parsed 全量 + raw 截断)、多久、成败。文件: rl/logs/judge_calls.jsonl"""

    def __init__(self, path=None):
        self.path = Path(path) if path else _DEFAULT_JUDGE_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        try:
            rec = {"ts": time.time(), **rec}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False,
                                   default=str) + "\n")
        except Exception:
            pass                      # 留痕失败绝不打断评审

W_FORMAT, W_TEXT, W_VIDEO = 0.15, 0.35, 0.50
VIDEO_W = {"action": 0.30, "physics": 0.25, "camera": 0.15,
           "consistency": 0.30}


def _skill(name: str) -> str:
    p = _SKILL_DIR / name / "SKILL.md"
    body = p.read_text(encoding="utf-8")
    return body.split("---", 2)[-1].strip() if body.startswith("---") \
        else body


def _extract_json(text: str):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class OpenAICompatChat:
    """极简 OpenAI 兼容客户端(文本或多模态消息;判官专用)。"""

    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: int = 300, extra_body: dict = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # 原样并进请求体(思考型模型关思考:MaaS 网关认顶层
        # enable_thinking:false;判官只要 JSON 判词,不需要思考链)
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}

    def chat(self, content, retries: int = 2) -> str:
        """content: str 或 OpenAI 多模态 content 列表。"""
        msg = [{"role": "user", "content": content}]
        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={**self.extra_body,
                          "model": self.model, "messages": msg},
                    timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt >= retries:
                    raise
                time.sleep(10 * (attempt + 1))
        return ""


def _video_part(path: str) -> dict:
    """本地视频 → base64 data-URI 的 video_url 部件(dashscope omni
    兼容层原生视频通道;用户令:不抽帧)。"""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{data}"}}


def _image_part(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"}}


# ── 文本判官 ──────────────────────────────────────────────────────────
class TextJudge:
    def __init__(self, client: OpenAICompatChat, log: "JudgeLog" = None):
        self.client = client
        self.skill = _skill("prompt_review")
        self.log = log

    def score(self, case: dict, tag: dict = None) -> tuple[float, dict]:
        """case 见 prompt_review 技能的 Input materials;返回 (0..1, 明细)。"""
        t0 = time.time()
        raw, err = "", ""
        try:
            raw = self.client.chat(
                self.skill + "\n\nTHE CASE FILE (JSON):\n"
                + json.dumps(case, ensure_ascii=False))
            data = _extract_json(raw)
            scores = (data or {}).get("scores") or {}
            vals = [v for v in scores.values()
                    if isinstance(v, (int, float))]
            if not vals:
                raise RuntimeError(
                    f"text judge unusable reply: {raw[:120]}")
            out = sum(vals) / (5.0 * len(vals)), {
                "scores": scores,
                "rationale": (data or {}).get("rationale")}
            return out
        except Exception as exc:
            err = str(exc)[:200]
            raise
        finally:
            if getattr(self, "log", None) is not None:
                self.log.write({"judge": "text",
                                "model": getattr(self.client, "model", "?"),
                                "tag": tag or {},
                                "scores": (_extract_json(raw) or {}
                                           ).get("scores"),
                                "rationale": (_extract_json(raw) or {}
                                              ).get("rationale"),
                                "raw": raw[:2000], "error": err,
                                "latency_s": round(time.time() - t0, 1)})


# ── 视频排名判官(一组四段一次调用)───────────────────────────────────
_RANK_SKILLS = {"action": "video_rank_action",
                "physics": "video_rank_physics",
                "camera": "video_rank_camera"}


def rank_to_points(ranking: list, n: int) -> dict:
    """名次 → 点数(第1名=n-1 … 末名=0,除以 n-1 归一;并列 =
    ranking 里的子列表,点数取平均)。ranking 元素:标签或标签列表。"""
    pts: dict = {}
    pos = 0
    for item in ranking:
        group = item if isinstance(item, list) else [item]
        ranks = list(range(pos, pos + len(group)))
        avg = sum((n - 1 - r) for r in ranks) / len(group)
        for g in group:
            pts[str(g)] = avg / max(1, n - 1)
        pos += len(group)
    return pts


class VideoRanker:
    def __init__(self, client: OpenAICompatChat, rng_seed: int = 0,
                 log: "JudgeLog" = None):
        self.client = client
        self.rng = random.Random(rng_seed)
        self.log = log

    def rank(self, dim: str, context: dict,
             videos: list[str], tag: dict = None) -> dict:
        """videos: 按候选序的路径表。返回 {候选下标: 点数 0..1}。
        展示顺序随机打乱(防位置偏好),映射留在返回明细里。"""
        n = len(videos)
        order = list(range(n))
        self.rng.shuffle(order)
        labels = [chr(ord("A") + i) for i in range(n)]
        content: list = [{"type": "text", "text":
                          _skill(_RANK_SKILLS[dim])
                          + "\n\nCONTEXT (JSON):\n"
                          + json.dumps(context, ensure_ascii=False)}]
        for lab, idx in zip(labels, order):
            content.append({"type": "text", "text": f"Video {lab}:"})
            content.append(_video_part(videos[idx]))
        t0 = time.time()
        raw, err = "", ""
        try:
            raw = self.client.chat(content)
            data = _extract_json(raw)
            ranking = (data or {}).get("ranking")
            if not ranking:
                raise RuntimeError(f"rank judge unusable: {raw[:120]}")
            pts_by_label = rank_to_points(ranking, n)
            out = {}
            for lab, idx in zip(labels, order):
                if lab not in pts_by_label:
                    raise RuntimeError(f"rank missing label {lab}")
                out[idx] = pts_by_label[lab]
            return {"points": out, "order": order,
                    "evidence": (data or {}).get("evidence")}
        except Exception as exc:
            err = str(exc)[:200]
            raise
        finally:
            if getattr(self, "log", None) is not None:
                d_ = _extract_json(raw) or {}
                self.log.write({"judge": f"rank_{dim}",
                                "model": getattr(self.client, "model", "?"),
                                "tag": tag or {},
                                "videos": [Path(v).name for v in videos],
                                "display_order": order,
                                "ranking": d_.get("ranking"),
                                "evidence": d_.get("evidence"),
                                "regime": d_.get("regime"),
                                "raw": raw[:2000], "error": err,
                                "latency_s": round(time.time() - t0, 1)})


# ── 一致性直判(对照清单,逐候选)─────────────────────────────────────
class ConsistencyChecker:
    def __init__(self, client: OpenAICompatChat, log: "JudgeLog" = None):
        self.client = client
        self.skill = _skill("consistency_check")
        self.log = log

    def score(self, video: str, refs: list[dict],
              context: dict, tag: dict = None) -> tuple[float, dict]:
        """refs: [{"kind": "portrait:<名>"|"space_view", "path", "note"}]。
        返回 (通过率 0..1, 明细)。"""
        content: list = [{"type": "text", "text":
                          self.skill + "\n\nCONTEXT (JSON):\n"
                          + json.dumps(context, ensure_ascii=False)}]
        for r in refs:
            content.append({"type": "text",
                            "text": f"Reference [{r['kind']}] "
                                    f"{r.get('note', '')}:"})
            content.append(_image_part(r["path"]))
        content.append({"type": "text", "text": "The video under review:"})
        content.append(_video_part(video))
        t0 = time.time()
        raw, err = "", ""
        try:
            raw = self.client.chat(content)
            data = _extract_json(raw)
            checks = (data or {}).get("checks")
            if not isinstance(checks, list) or not checks:
                raise RuntimeError(
                    f"consistency judge unusable: {raw[:120]}")
            valid = [c for c in checks if c.get("pass") is not None]
            if not valid:
                raise RuntimeError(
                    "consistency judge: all items unjudgeable")
            n_pass = sum(1 for c in valid if c.get("pass") is True)
            return n_pass / len(valid), {"checks": checks,
                                         "n_null": len(checks) - len(valid)}
        except Exception as exc:
            err = str(exc)[:200]
            raise
        finally:
            if getattr(self, "log", None) is not None:
                d_ = _extract_json(raw) or {}
                self.log.write({"judge": "consistency",
                                "model": getattr(self.client, "model", "?"),
                                "tag": tag or {},
                                "video": Path(video).name,
                                "refs": [r.get("kind") for r in refs],
                                "checks": d_.get("checks"),
                                "raw": raw[:2000], "error": err,
                                "latency_s": round(time.time() - t0, 1)})


# ── 合成 ──────────────────────────────────────────────────────────────
def compose_rewards(fmt: list[float], text: list[float | None],
                    video_parts: dict, n: int) -> list[dict]:
    """按 v3 公式逐候选合成;缺失分量剔除并归一化,全程留痕。
    fmt/text 按候选序;video_parts = {dim: {idx: score0..1} 或 None}。"""
    out = []
    for i in range(n):
        vw, vsum = 0.0, 0.0
        vdetail = {}
        for dim, w in VIDEO_W.items():
            comp = video_parts.get(dim)
            if comp is not None and i in comp:
                vw += w
                vsum += w * comp[i]
                vdetail[dim] = round(comp[i], 4)
        r_video = (vsum / vw) if vw > 0 else None
        parts, weights = [], []
        parts.append(fmt[i]); weights.append(W_FORMAT)
        if text[i] is not None:
            parts.append(text[i]); weights.append(W_TEXT)
        if r_video is not None:
            parts.append(r_video); weights.append(W_VIDEO)
        r = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
        out.append({"reward": round(r, 4),
                    "r_format": round(fmt[i], 4),
                    "r_text": (round(text[i], 4)
                               if text[i] is not None else None),
                    "r_video": (round(r_video, 4)
                                if r_video is not None else None),
                    "video_detail": vdetail,
                    "dropped_components": sorted(
                        set(VIDEO_W) - set(vdetail))})
    return out
