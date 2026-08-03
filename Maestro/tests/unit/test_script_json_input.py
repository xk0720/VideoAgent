"""用户剧本 JSON 输入(固定契约)回归:
① 解析:role 字典 / 顶层平铺双 schema;content 缺失响亮拒;
② 路径救援链:原路径 → 同名 → 数字归一(000014≡00014),救援/缺失/
   名字不在 content 均留痕(裁决:救援并记录、告警继续);
③ 角色提取带 given:名字逐字进正典;LLM 漏名确定性兜底;无 LLM 时
   given 直接成正典(given_only);
④ 管线预填:user_json 肖像入台账、§A' 跳过不再 t2i、正典含钦定名。
全部离线。"""
import json
from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.pipeline.script_input import parse_script_json


def _png(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n" + b"\x00" * 16)
    return p


def _write(tmp_path, data) -> Path:
    f = tmp_path / "script.json"
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return f


# ── ①② 解析与救援 ───────────────────────────────────────────────────

def test_parse_role_dict_and_flat_schema(tmp_path):
    img = _png(tmp_path / "a.png")
    f = _write(tmp_path, {"content": "安娜走进舞厅",
                          "role": {"安娜": str(img)},
                          "军官": str(img)})
    got = parse_script_json(f)
    assert got["content"] == "安娜走进舞厅"
    assert got["roles"]["安娜"] == str(img)      # role 字典
    assert got["roles"]["军官"] == str(img)      # 顶层平铺兼容
    # 军官不在 content → 告警留痕但继续
    assert any(n["action"] == "name_not_in_content"
               and n["name"] == "军官" for n in got["notes"])


def test_parse_requires_content(tmp_path):
    f = _write(tmp_path, {"role": {}})
    try:
        parse_script_json(f)
        assert False, "must raise"
    except ValueError as e:
        assert "content" in str(e)


def test_path_rescue_chain(tmp_path):
    real = _png(tmp_path / "ComfyUI_00014_.png")
    f = _write(tmp_path, {
        "content": "安娜与军官",
        "role": {"安娜": str(tmp_path / "ComfyUI_000014_.png"),  # 数字位数错
                 "军官": "/nowhere/ComfyUI_00014_.png",          # 目录错→同名
                 "路人": str(tmp_path / "gone.png")}})           # 无解
    got = parse_script_json(f)
    assert got["roles"]["安娜"] == str(real)     # 数字归一救援
    assert got["roles"]["军官"] == str(real)     # 同目录同名救援
    assert got["roles"]["路人"] is None          # 缺失如实为 None
    acts = {(n["name"], n["action"]) for n in got["notes"]}
    assert ("安娜", "path_rescued") in acts
    assert ("路人", "image_missing") in acts


# ── ③ 提取带 given ──────────────────────────────────────────────────

class _LLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt, **k):
        assert "given_characters" in prompt      # given 进了任务 JSON
        return self.reply


def test_extract_given_backstop_and_given_only():
    llm = _LLM(json.dumps({"characters": {
        "安娜": "static: blue-eyed noblewoman; dynamic: gown"}}))
    chars, via = wl._extract_characters(
        llm, "剧本…", given={"安娜": "a blue-eyed woman",
                             "军官": "a uniformed officer"})
    assert via == "llm"
    assert chars["安娜"].startswith("static: blue-eyed")
    # LLM 漏了军官 → 确定性兜底(以图像打标为 static 法源)
    assert "uniformed officer" in chars["军官"]
    # 无 LLM:given 直接成正典
    chars2, via2 = wl._extract_characters(
        None, "剧本…", given={"安娜": "a blue-eyed woman"})
    assert via2 == "given_only" and "安娜" in chars2


# ── ④ 管线预填(复用 M2 离线夹具)────────────────────────────────────

def test_pipeline_preseeds_user_json_portraits(tmp_path, monkeypatch):
    from maestro.agents.generator import GeneratorAgent
    from maestro.agents.orchestrator import OrchestratorAgent
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.models.video_gen import MockVideoGenClient
    from maestro.pipeline.window_loop import generate_movie_windowed

    img = _png(tmp_path / "anna.png")

    class _VG(MockVideoGenClient):
        def __init__(self):
            super().__init__(name="mock")
            self.t2i_prompts = []

        def capabilities(self):
            return {"t2v", "i2v", "t2i"}

        def text_to_image(self, prompt, out_path, seed=0):
            self.t2i_prompts.append(prompt)
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("MOCK IMAGE")
            return out

    class _LLM2:
        def complete(self, prompt, **k):
            if "Character Extraction" in prompt[:400]:
                return json.dumps({"characters": {
                    "安娜": "static: blue-eyed noblewoman in a pale gown; "
                            "dynamic: expression"}})
            if "Image Plan" in prompt[:200]:
                return json.dumps({"strategy": "none", "images": [],
                                   "reason": "stub"})
            return json.dumps({
                "strategy": "t2v", "reason": "stub",
                "shots": [{"description": "Shot 1: <安娜> walks in",
                           "duration_s": 5,
                           "end_state": "she stands; camera: static"}],
                "cast": {}, "setting": "a ballroom"})

    class _Concat:
        def run(self, clips, out):
            out = Path(out)
            out.write_text("MERGED")
            return out
    import maestro.tools.video_concat as vc
    monkeypatch.setattr(vc, "VideoConcatTool", _Concat)
    gen = GeneratorAgent(video_gen=_VG())
    res = generate_movie_windowed(
        "unused idea", cache_dir=tmp_path / "run", llm=_LLM2(),
        max_turns=1, n_candidates=1, enable_review=False,
        screenplay="第一场:安娜走进舞厅。",
        given_characters={"安娜": str(img), "缺图者": None},
        board=None, generator=gen, refiner=RefinerAgent(),
        verifier=VerifierAgent(),
        orchestrator=OrchestratorAgent(generator=gen))
    sb = res.storyboard
    assert sb.portraits["安娜"] == str(img)          # user_json 预填
    assert "安娜" in sb.cast                          # 正典含钦定名
    # §A' 对安娜跳过 → 没有为她花 t2i(场景锚/缺图者可能有 t2i,
    # 校验没有任何 t2i prompt 是"安娜的全身肖像")
    assert not any("安娜" in p and "portrait" in p
                   for p in gen.video_gen.t2i_prompts)
    assert any(d.get("via") == "user_json" for d in res.decisions
               if d.get("stage") == "cast_portrait")
