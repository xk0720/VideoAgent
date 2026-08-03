"""百炼可灵后端(BailianKlingClient)回归:payload 形态 / media 顺序 /
引用方言 / 上传缓存 / FAILED 响亮 / wavespeed 代理裁决。全离线(requests
会话打桩),CPU-only。契约来源:scripts/playground/bailian_kling_probe.py
(M0 真 API 验证,2026-08-03)—— 测试断言的就是那份实测契约。"""
import json
import os
import sys
import types
from pathlib import Path

import pytest

from maestro.models.video_gen import MockVideoGenClient
from maestro.models.video_gen_backends import (BailianKlingClient,
                                               WaveSpeedClient,
                                               build_real_video_gen)

POLICY = {"upload_dir": "up/2026", "oss_access_key_id": "ak",
          "signature": "sig", "policy": "pol", "x_oss_object_acl": "private",
          "x_oss_forbid_overwrite": "true", "upload_host": "http://oss.fake"}


class _Resp:
    def __init__(self, code, data=None, content=b""):
        self.status_code = code
        self._data = data or {}
        self.content = content
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


class _FakeSession:
    """离线会话桩:按 URL 分发 getPolicy / OSS 表单直传 / 提交 / 轮询 /
    成片下载(照 test_condition_model_map 的打桩路数,只是打的是会话)。"""

    def __init__(self, poll_output=None):
        self.trust_env = True     # 客户端必须把它设成 False(M0 代理教训)
        self.submits = []         # [{url, payload, headers}]
        self.uploads = []         # OSS 直传的 key 列表
        self.policy_calls = 0
        self.downloads = []
        self.poll_output = poll_output or {
            "task_status": "SUCCEEDED", "video_url": "http://fake/v.mp4"}

    def get(self, url, params=None, headers=None, timeout=0, **kw):
        if "/uploads" in url:
            self.policy_calls += 1
            return _Resp(200, {"data": dict(POLICY)})
        if "/tasks/" in url:
            return _Resp(200, {"output": dict(self.poll_output)})
        self.downloads.append(url)
        return _Resp(200, {}, content=b"FAKEVIDEO")

    def post(self, url, json=None, headers=None, files=None, timeout=0, **kw):
        if files is not None:                          # OSS 表单直传
            self.uploads.append(files["key"][1])
            return _Resp(200, {})
        self.submits.append({"url": url, "payload": json, "headers": headers})
        return _Resp(200, {"output": {"task_id": "task-1"}})


def _client(monkeypatch, fake, **cfg):
    """构造客户端并把 requests 模块打桩:会话仍由 _session() 真实创建,
    trust_env=False 的 M0 修法路径被覆盖。默认无 wavespeed 代理。"""
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "requests",
                        types.SimpleNamespace(Session=lambda: fake))
    return BailianKlingClient(config={"api_key": "k", **cfg})


def test_t2v_payload(tmp_path, monkeypatch):
    """①纯文本:标准 kling 模型;aspect_ratio 必填(M0);duration 夹到
    [3,15]/None 不传;seed 不进 payload;异步头开、Oss 解析头不带;
    audio ← generate_audio 属性;会话直连(trust_env=False)。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake)
    out = c.generate("a cat trots", 20, tmp_path / "o.mp4", seed=42)
    assert out.read_bytes() == b"FAKEVIDEO"

    sub = fake.submits[-1]
    assert sub["url"].endswith(
        "/services/aigc/video-generation/video-synthesis")
    p = sub["payload"]
    assert p["model"] == "kling/kling-v3-video-generation"   # 纯文本 → 标准
    assert p["input"]["prompt"] == "a cat trots"
    assert "media" not in p["input"]
    assert p["parameters"]["aspect_ratio"] == "16:9"   # 无 first_frame 必填
    assert p["parameters"]["duration"] == 15           # 20 → 夹到上界 15
    assert p["parameters"]["mode"] == "std"
    assert p["parameters"]["audio"] is False
    assert "seed" not in json.dumps(p)                 # API 不支持 → 不传
    assert sub["headers"]["X-DashScope-Async"] == "enable"
    assert "X-DashScope-OssResourceResolve" not in sub["headers"]
    assert fake.trust_env is False                     # M0:代理掐轮询的修法

    c.generate("p2", 2, tmp_path / "o2.mp4")           # 2 → 夹到下界 3
    assert fake.submits[-1]["payload"]["parameters"]["duration"] == 3
    c.generate("p3", None, tmp_path / "o3.mp4")        # None → 不传字段
    assert "duration" not in fake.submits[-1]["payload"]["parameters"]

    c.generate_audio = True                            # 管线对白镜的临时开关
    c.generate("p4", None, tmp_path / "o4.mp4")
    assert fake.submits[-1]["payload"]["parameters"]["audio"] is True


def test_first_frame_plus_refs_mix(tmp_path, monkeypatch):
    """②mix(M0 已验证):omni 模型;media = first_frame + refer×2,refer
    顺序即 <<<image_N>>> 编号;带媒体 → Oss 解析头;有 first_frame →
    不带 aspect_ratio;refer-only → aspect_ratio 必填。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake)
    ff = tmp_path / "ff.png"
    ff.write_bytes(b"\x89PNG a")
    r1 = tmp_path / "r1.png"
    r1.write_bytes(b"\x89PNG b")
    r2 = tmp_path / "r2.png"
    r2.write_bytes(b"\x89PNG c")
    c.generate("<<<image_1>>> walks in", 5, tmp_path / "o.mp4",
               first_frame=ff, reference_images=[r1, r2])

    sub = fake.submits[-1]
    p = sub["payload"]
    assert p["model"] == "kling/kling-v3-omni-video-generation"  # 带媒体 → omni
    media = p["input"]["media"]
    assert [m["type"] for m in media] == ["first_frame", "refer", "refer"]
    assert [m["url"] for m in media] == [
        f"oss://{POLICY['upload_dir']}/ff.png",
        f"oss://{POLICY['upload_dir']}/r1.png",     # refer#1 = <<<image_1>>>
        f"oss://{POLICY['upload_dir']}/r2.png"]     # refer#2 = <<<image_2>>>
    assert "aspect_ratio" not in p["parameters"]    # 有 first_frame → 不填
    assert sub["headers"]["X-DashScope-OssResourceResolve"] == "enable"
    assert p["parameters"]["duration"] == 5

    c.generate("<<<image_1>>> smiles", None, tmp_path / "o2.mp4",
               reference_images=[r1])               # refer-only(ref2v 形态)
    p2 = fake.submits[-1]["payload"]
    assert p2["parameters"]["aspect_ratio"] == "16:9"  # M0:无 first_frame 必填


def test_frame_to_frame_media(tmp_path, monkeypatch):
    """③flf2v:media = first_frame + last_frame;omni 模型;有 first_frame
    → 无 aspect_ratio;duration 关键字兼容 timeline.py 的调用。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake)
    a = tmp_path / "a.png"
    a.write_bytes(b"A")
    b = tmp_path / "b.png"
    b.write_bytes(b"B")
    c.frame_to_frame(prompt="smooth transition", first_frame=a, last_frame=b,
                     out_path=tmp_path / "o.mp4", duration=3)
    p = fake.submits[-1]["payload"]
    assert p["model"] == "kling/kling-v3-omni-video-generation"
    assert [m["type"] for m in p["input"]["media"]] == ["first_frame",
                                                        "last_frame"]
    assert p["parameters"]["duration"] == 3        # 裁决:转场固定 3s 由调用方传
    assert "aspect_ratio" not in p["parameters"]


def test_reference_video_rejected_before_any_upload(tmp_path, monkeypatch):
    """④kling 没有参考视频通道 → 任何上传/提交发生之前响亮拒绝(诚实,
    不静默丢条件)。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake)
    ff = tmp_path / "f.png"
    ff.write_bytes(b"F")
    rv = tmp_path / "r.mp4"
    rv.write_bytes(b"\x00" * 8)
    with pytest.raises(RuntimeError, match="reference-video"):
        c.generate("p", 5, tmp_path / "o.mp4", first_frame=ff,
                   reference_video=rv)
    assert fake.uploads == [] and fake.submits == []


def test_failed_task_raises_with_code_and_message(tmp_path, monkeypatch):
    """⑤FAILED 任务响亮 raise,带 API 的 code + message(可审计)。"""
    fake = _FakeSession(poll_output={
        "task_status": "FAILED", "code": "InvalidParameter.DataInspection",
        "message": "input image blocked"})
    c = _client(monkeypatch, fake)
    with pytest.raises(RuntimeError) as ei:
        c.generate("p", 3, tmp_path / "o.mp4")
    s = str(ei.value)
    assert "InvalidParameter.DataInspection" in s
    assert "input image blocked" in s
    assert "FAILED" in s


def test_ref_token_dialect_three_clients():
    """⑥引用方言:kling = <<<image_N>>>;WaveSpeed/Mock = @ImageN
    (向后兼容)。窗口管线写引用句时问后端要方言,不再硬编码。"""
    assert BailianKlingClient(config={"api_key": "k"}).ref_token(2) \
        == "<<<image_2>>>"
    assert WaveSpeedClient(config={"api_key": "k"}).ref_token(2) == "@Image2"
    assert MockVideoGenClient().ref_token(2) == "@Image2"


def test_upload_cache_uploads_same_file_once(tmp_path, monkeypatch):
    """⑦同文件(路径+mtime)只上传一次;文件变了(mtime 变)必须重传。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake)
    ff = tmp_path / "kf.png"
    ff.write_bytes(b"PNG")
    c.generate("p1", 3, tmp_path / "o1.mp4", first_frame=ff)
    c.generate("p2", 3, tmp_path / "o2.mp4", first_frame=ff)
    assert len(fake.uploads) == 1 and fake.policy_calls == 1
    assert len(fake.submits) == 2                     # 生成照常提交了两次

    ff.write_bytes(b"PNG v2")                         # 内容变了
    os.utime(ff, ns=(10 ** 18, 10 ** 18))             # mtime 显式变(防同秒)
    c.generate("p3", 3, tmp_path / "o3.mp4", first_frame=ff)
    assert len(fake.uploads) == 2


def test_no_wavespeed_proxy_raises_loud(tmp_path, monkeypatch):
    """⑧没有 wavespeed 代理(无子配置、无 $WAVESPEED_API_KEY)时,
    t2i / _run_task(音乐)响亮 raise,不装死。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake)          # helper 已 delenv WAVESPEED_API_KEY
    assert c._wavespeed is None
    with pytest.raises(RuntimeError, match="[Ww]ave[Ss]peed"):
        c.text_to_image("a poster", tmp_path / "i.png")
    with pytest.raises(RuntimeError, match="[Ww]ave[Ss]peed"):
        c._run_task("sonilo/text-to-music", {"prompt": "x"},
                    tmp_path / "m.mp3")


def test_wavespeed_proxy_forwards_t2i_and_music(tmp_path, monkeypatch):
    """裁决"t2i/音乐暂留 wavespeed":有子配置 → 内部代理实例化,
    text_to_image / _run_task 全部转发。"""
    fake = _FakeSession()
    c = _client(monkeypatch, fake, wavespeed={"api_key": "wk"})
    assert isinstance(c._wavespeed, WaveSpeedClient)
    calls = []

    class _WS:
        def text_to_image(self, prompt, out_path, seed=0):
            calls.append(("t2i", prompt, seed))
            return Path(out_path)

        def _run_task(self, model_id, payload, out_path):
            calls.append(("task", model_id, payload["prompt"]))
            return Path(out_path)

    c._wavespeed = _WS()
    c.text_to_image("poster", tmp_path / "i.png", seed=7)
    c._run_task("sonilo/text-to-music", {"prompt": "calm piano"},
                tmp_path / "m.mp3")
    assert calls == [("t2i", "poster", 7),
                     ("task", "sonilo/text-to-music", "calm piano")]


def test_call_log_records_submit_and_completed(tmp_path, monkeypatch):
    """call_log 机制与 WaveSpeed 同款(任务 0):submit/completed 逐行
    JSONL;payload 里 oss:// URL 原样记,prompt 全文可读(审计教训)。"""
    fake = _FakeSession()
    logf = tmp_path / "calls.jsonl"
    c = _client(monkeypatch, fake, call_log=str(logf))
    ff = tmp_path / "kf.png"
    ff.write_bytes(b"PNG")
    long_prompt = "the baker smiles warmly at the customer. " * 12  # >200ch
    c.generate(long_prompt, 4, tmp_path / "o.mp4", first_frame=ff)

    lines = [json.loads(x) for x in logf.read_text().splitlines()]
    assert [x["event"] for x in lines] == ["submit", "completed"]
    assert all(x["model"] == "kling/kling-v3-omni-video-generation"
               for x in lines)
    rec = lines[0]["payload"]
    assert rec["input"]["prompt"] == long_prompt      # 全文,无缩写
    assert rec["input"]["media"][0]["url"] \
        == f"oss://{POLICY['upload_dir']}/kf.png"     # oss url 原样记
    assert lines[1]["task_id"] == "task-1"


def test_capabilities_and_registry():
    """能力集合精确(刻意不含 extend/ref_video/multi_i2v —— 旧菜单项按
    能力自然消失);注册表别名 bailian_kling / bailian / kling 全通。"""
    c = BailianKlingClient(config={"api_key": "k"})
    assert c.capabilities() == {"t2v", "i2v", "flf2v", "ref_images",
                                "first_frame_plus_refs"}
    assert c.supported_conditions() == {"first_frame", "reference_images"}
    for alias in ("bailian_kling", "bailian", "kling"):
        built = build_real_video_gen(alias, {"api_key": "k"})
        assert isinstance(built, BailianKlingClient), alias
