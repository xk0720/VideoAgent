#!/usr/bin/env python3
"""M0:百炼可灵真 API 验证(dev-music-bailian 重构的先行闸门)。

验证五种调用形态 + 本地文件上传链路(全部真调用,3s/std 控费):
  t2v      纯文生视频
  i2v      first_frame(本地图 → OSS 上传 → oss:// 引用)
  flf2v    first_frame + last_frame(转场工具的底座)
  ref2v    双 refer + <<<image_1>>>/<<<image_2>>> 服从度
  mix      first_frame + refer 混用可否(直接决定策略池形态)

用法:python scripts/playground/bailian_kling_probe.py [t2v|i2v|flf2v|ref2v|mix|all]
输出:outputs/bailian_probe_<ts>/ 下逐任务 JSON 回执 + 成片。
BASE_URL 按用户裁决用公共 dashscope 端点(2026-08-03)。
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE = "https://dashscope.aliyuncs.com/api/v1"

# 本机系统代理会掐断长轮询 —— 直连(阿里云国内可达,无需代理)
S = requests.Session()
S.trust_env = False
requests = S  # 下文所有 requests.get/post 走同一直连会话
SYNTH = f"{BASE}/services/aigc/video-generation/video-synthesis"
MODEL_OMNI = "kling/kling-v3-omni-video-generation"
MODEL_STD = "kling/kling-v3-video-generation"

# 测试素材:上一部片的真实产物(肖像 + 尾帧类画面)
IMG_A = REPO / "outputs/portrait_fix_20260731_154644/portraits/年轻面包师.png"
IMG_B = REPO / "outputs/portrait_fix_20260731_154644/portraits/撑伞的顾客.png"

OUT = REPO / "outputs" / f"bailian_probe_{time.strftime('%Y%m%d_%H%M%S')}"


def _headers(async_=True):
    h = {"Authorization": f"Bearer {API_KEY}",
         "Content-Type": "application/json"}
    if async_:
        h["X-DashScope-Async"] = "enable"
    return h


def upload(path: Path, model: str) -> str:
    """DashScope 官方临时上传:getPolicy → OSS 表单直传 → oss:// URL。"""
    r = requests.get(f"{BASE}/uploads",
                     params={"action": "getPolicy", "model": model},
                     headers={"Authorization": f"Bearer {API_KEY}"},
                     timeout=30)
    r.raise_for_status()
    pol = r.json()["data"]
    key = f"{pol['upload_dir']}/{path.name}"
    with path.open("rb") as f:
        files = {
            "OSSAccessKeyId": (None, pol["oss_access_key_id"]),
            "Signature": (None, pol["signature"]),
            "policy": (None, pol["policy"]),
            "x-oss-object-acl": (None, pol["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, pol["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (path.name, f),
        }
        up = requests.post(pol["upload_host"], files=files, timeout=120)
    up.raise_for_status()
    return f"oss://{key}"


def submit_and_poll(name: str, payload: dict, needs_oss: bool) -> dict:
    h = _headers()
    if needs_oss:
        h["X-DashScope-OssResourceResolve"] = "enable"
    r = requests.post(SYNTH, headers=h, json=payload, timeout=60)
    rec = {"name": name, "submit_status": r.status_code,
           "submit_body": r.json() if r.text else {}}
    print(f"[{name}] submit HTTP {r.status_code}: "
          f"{json.dumps(rec['submit_body'], ensure_ascii=False)[:300]}")
    task_id = (rec["submit_body"].get("output") or {}).get("task_id")
    if not task_id:
        return rec
    deadline = time.time() + 900
    while time.time() < deadline:
        q = requests.get(f"{BASE}/tasks/{task_id}",
                         headers=_headers(async_=False), timeout=30)
        out = (q.json().get("output") or {})
        st = out.get("task_status")
        if st in ("SUCCEEDED", "FAILED", "UNKNOWN"):
            rec["final"] = q.json()
            print(f"[{name}] {st}: "
                  f"{json.dumps(out, ensure_ascii=False)[:400]}")
            url = out.get("video_url")
            if url:
                v = requests.get(url, timeout=300)
                dst = OUT / f"{name}.mp4"
                dst.write_bytes(v.content)
                rec["local"] = str(dst)
                print(f"[{name}] saved → {dst}")
            return rec
        print(f"[{name}] {st} …")
        time.sleep(15)
    rec["final"] = {"error": "poll timeout"}
    return rec


def poll_only(task_id: str) -> None:
    """补轮询一个已提交的任务(代理断线等场景,不重复花钱)。"""
    OUT.mkdir(parents=True, exist_ok=True)
    q = requests.get(f"{BASE}/tasks/{task_id}",
                     headers=_headers(async_=False), timeout=30)
    print(json.dumps(q.json(), ensure_ascii=False, indent=1)[:800])
    url = (q.json().get("output") or {}).get("video_url")
    if url:
        dst = OUT / f"poll_{task_id[:8]}.mp4"
        dst.write_bytes(requests.get(url, timeout=300).content)
        print("saved →", dst)


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "poll":
        poll_only(sys.argv[2])
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    def run(name, payload, needs_oss=False):
        results.append(submit_and_poll(name, payload, needs_oss))
        (OUT / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1))

    if which in ("t2v", "all"):
        run("t2v", {
            "model": MODEL_STD,
            "input": {"prompt": "A small orange cat trots across a sunlit "
                                "wooden floor, then sits down. Static "
                                "camera, warm morning light."},
            "parameters": {"mode": "std", "duration": 3, "audio": False,
                           "aspect_ratio": "16:9"}})

    oss_a = oss_b = None
    if which in ("i2v", "flf2v", "ref2v", "mix", "all"):
        print("uploading local images …")
        oss_a = upload(IMG_A, MODEL_OMNI)
        oss_b = upload(IMG_B, MODEL_OMNI)
        print("oss urls ready")

    if which in ("i2v", "all"):
        run("i2v", {
            "model": MODEL_OMNI,
            "input": {"prompt": "The young baker blinks naturally and "
                                "gives a small warm smile; hair moves "
                                "slightly in the breeze. Camera static.",
                      "media": [{"type": "first_frame", "url": oss_a}]},
            "parameters": {"mode": "std", "duration": 3,
                           "audio": False}}, needs_oss=True)

    if which in ("flf2v", "all"):
        run("flf2v", {
            "model": MODEL_OMNI,
            "input": {"prompt": "A smooth cinematic transition: the view "
                                "shifts naturally from the baker to the "
                                "umbrella customer inside the same rainy "
                                "bakery, steady camera.",
                      "media": [{"type": "first_frame", "url": oss_a},
                                {"type": "last_frame", "url": oss_b}]},
            "parameters": {"mode": "std", "duration": 3,
                           "audio": False}}, needs_oss=True)

    if which in ("ref2v", "all"):
        run("ref2v", {
            "model": MODEL_OMNI,
            "input": {"prompt": "<<<image_1>>>的年轻面包师站在柜台后微笑点头,"
                                "<<<image_2>>>的撑伞顾客推门走进面包店,"
                                "镜头固定,雨天清晨柔和光线。",
                      "media": [{"type": "refer", "url": oss_a},
                                {"type": "refer", "url": oss_b}]},
            # M0 实测:无 first_frame 时 aspect_ratio 必填
            "parameters": {"mode": "std", "duration": 3, "audio": False,
                           "aspect_ratio": "16:9"}}, needs_oss=True)

    if which in ("mix", "all"):
        run("mix", {
            "model": MODEL_OMNI,
            "input": {"prompt": "The shot opens exactly on the first "
                                "frame. <<<image_1>>>的撑伞顾客从右侧走入"
                                "画面,面包师转头看向她。镜头固定。",
                      "media": [{"type": "first_frame", "url": oss_a},
                                {"type": "refer", "url": oss_b}]},
            "parameters": {"mode": "std", "duration": 3,
                           "audio": False}}, needs_oss=True)

    print(f"\nall receipts → {OUT}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
