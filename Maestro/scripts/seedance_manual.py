#!/usr/bin/env python3
"""手写分镜 × WaveSpeed seedance-2.0 完整对照实验(2026-08-05 用户令)。

═══ 接缝设计(导演判断,逐缝说明)═══════════════════════════════
  1→2  软钉   同主体续拍。seedance 的 i2v 硬钉通道无参考图口
              (schema 只收 image+last_image)→ 改走库里验证过的 ti2v
              软钉:t2v 通道,把【上一镜末帧】本身作为 @Image1 参考,
              肖像跟在其后;prompt 格式(旧库铁律):首句显式声明
              "画面从@Image1精确开始——它是上一镜的最后一帧",随后
              写运动,人物用后续 @记号绑定。钉与 refs 兼得。
  2→3  硬切   对话轴反打(王子怒斥→安娜反应):安娜在两侧画面中
              皆在场,反打硬切成立。t2v+refs 全新构图。
  3→4  转场桥 换人接缝(安娜→安莉希娅+王子):flf2v 桥,prompt 让
              首端人物随运镜出画、新人物随落位入画,严禁原地变形。
  4→5  转场桥 换人接缝(二人→军官组):同上,横摇穿过舞厅。
  5→6  软钉   同两位军官续拍(乙转脸回应),格式同 1→2。
  6→7  转场桥 换人接缝(军官→持折扇女子):穿过人群的滑移+推进。
  拼接:1,2,3,[桥34],4,[桥45],5,6,[桥67],7
═══════════════════════════════════════════════════════════════════

用法: python scripts/seedance_manual.py          # 真实调用,花钱
输出: outputs/seedance_manual_<ts>/
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.script_input import parse_script_json
from maestro.pipeline.window_loop import (_extract_frame0, _last_frame,
                                          _probe_seconds)
from maestro.tools.video_concat import VideoConcatTool

SETTING = ("十九世纪欧式皇家宫廷大舞厅:金色雕花墙面与红色锦缎帷幔,"
           "多盏水晶吊灯烛光通明,抛光大理石地面倒影清晰,远端是带台阶的"
           "礼仪舞台;盛装贵族宾客与军官沿两侧墙边低声交谈,画面中央地面"
           "开阔无人。")

AUDIO_TAIL = "音频:只有角色说这句台词的人声——无背景音乐、无音效。"

# 转场桥 prompt(逐条手写)。换人桥的铁律:首末两帧人物不同 ——
# 必须【先让首端人物随运镜移出画面,新人物才随落位入画】,并明令
# "人物只随镜头运动出入画,绝不在原地变换面貌或服装",否则 flf2v
# 会走捷径把 A 原地变形成 B。
BRIDGE_34 = ("转场运镜,一镜到底:镜头从少女含泪的面部大特写平稳拉开并"
             "向左横摇,她随镜头运动移出画面右缘、完全离开画面;镜头掠过"
             "舞厅中央——水晶吊灯的暖色光斑与虚化的宾客剪影依次流过;"
             "横摇尽头镜头向前收拢,一位银发女子双手挽着黑金军装男子"
             "手臂、仰头望向他的中近景进入画面,构图对齐后停稳。人物只"
             "随镜头运动出入画,绝不在原地变换面貌或服装;全程同一空间"
             "与烛光,无剪辑感。")
BRIDGE_45 = ("转场运镜,一镜到底:镜头从挽臂而立的一对男女平稳向右横摇"
             "并缓缓拉开,二人随镜头运动移出画面左缘、完全离开画面;镜头"
             "摇过舞厅中景——吊灯光斑与两侧盛装宾客的剪影依次掠过,"
             "人群保持低声交谈的轻微动态;摇至人群边缘时减速,两名并肩"
             "而立、身穿深蓝镶金军装的年轻军官进入画面,构图收为双人"
             "中近景后停稳。人物只随镜头运动出入画,绝不在原地变换面貌"
             "或服装;全程同一空间与烛光,无剪辑感。")
BRIDGE_67 = ("转场运镜,一镜到底:镜头从两名并肩军官的双人中近景平稳"
             "向左前方滑移并轻微推进,二人随镜头运动移出画面右缘、完全"
             "离开画面;镜头穿过虚化的礼服裙摆与军装肩章之间,烛光光斑"
             "缓缓流过前景;滑移尽头,一名以黑色蕾丝折扇半遮下半张脸的"
             "女子进入画面,镜头继续推进并收紧为她的下半脸特写,扇沿与"
             "眼神清晰后停稳。人物只随镜头运动出入画,绝不在原地变换"
             "面貌或服装;全程同一空间与光线,无剪辑感。")

# mode: t2v(硬切,带 refs)| pin(钉上镜末帧,无 refs,只写运动)
SHOTS = [
    dict(  # 1 定场(t2v 冷开场)
        mode="t2v", refs=["安莉希娅", "芬莱克殿下", "安娜"],
        duration=8, audio=False,
        prompt=(f"大远景,固定镜头:{SETTING} @Image1挽着@Image2的手臂静立"
                "在舞台台阶前的中央,@Image3独自站在他们对面不远处,孤立"
                "无援。宾客仅在远处轻微走动交谈,无人进入中央前景。整段"
                "镜头保持固定,三人静立,画面庄重安静。")),
    dict(  # 2 软钉续拍:俯冲地板倒影→上摇过肩→退婚宣告
        mode="pin", refs=["芬莱克殿下", "安莉希娅", "安娜"],
        duration=8, audio=True,
        # 运行时 @Image1=上一镜末帧;@Image2=芬莱克,@Image3=安莉希娅,
        # @Image4=安娜(refs 顺延一位)
        prompt=("画面从@Image1精确开始——@Image1是上一镜的最后一帧,"
                "开场构图、人物与光线与其完全一致,不重新构图。镜头快速"
                "下俯并推进,落到抛光大理石地面的特写:地面映出@Image2与"
                "@Image3并立的倒影,烛光闪烁;@Image4的脚步走入倒影视野,"
                "步伐平稳。镜头随后从@Image4的脚部沿背影平稳上摇,越过"
                "她的肩膀形成过肩中近景;@Image2的脸清晰入画,神情冷酷"
                "严厉,开口说:“你这种女人不配做王后,安莉希娅才是真正"
                "合适的人选!”@Image3挽着他的手臂未动。说完后三人静止,"
                f"镜头停稳。{AUDIO_TAIL}")),
    dict(  # 3 硬切反打:安娜泪眼质疑
        mode="t2v", refs=["安娜"], duration=6, audio=True,
        prompt=("面部大特写,固定镜头:@Image1正对镜头,身后舞厅宾客完全"
                "虚化成暖色光斑。她蓝眸噙满泪水,下唇微微颤抖,神情不可"
                "置信,凝望画外;泪水在眼眶里打转而不落下。她艰难地轻声"
                "挤出一句:“……为什么?”说完后嘴唇停止颤动,面容僵住,"
                f"含泪双眼仍望向画外,镜头静止。{AUDIO_TAIL}")),
    dict(  # 4 硬切反打:安莉希娅假意劝阻
        mode="t2v", refs=["安莉希娅", "芬莱克殿下"], duration=6, audio=True,
        prompt=("中近景,固定镜头:@Image1站在@Image2身旁,双手挽着他的"
                "手臂,仰头望向他,面露不忍,嘴角却藏着一丝得意。她柔声"
                "说:“芬莱克殿下……这对于安娜小姐来说,会不会太过了?”"
                "说完后她保持仰望,笑意含而不露,@Image2神情僵硬不动,"
                f"镜头静止。{AUDIO_TAIL}")),
    dict(  # 5 军官甲低语(t2v 新构图;其前将插入 桥45)
        mode="t2v", refs=["男性军官"], duration=5, audio=True,
        prompt=("双人中近景,固定镜头:两名同穿深蓝镶金军装的年轻军官"
                "并肩站在宾客人群边缘,身后宾客虚化。左侧的@Image1侧身"
                "凑近右侧同伴,抬起白手套半遮嘴角,神情轻蔑而世故地低声"
                "说:“政治联姻的工具罢了。”右侧军官目视前方倾听。说完"
                f"后两人保持并肩静止,镜头静止。{AUDIO_TAIL}")),
    dict(  # 6 软钉续拍:军官乙转脸回应
        mode="pin", refs=["男性军官"], duration=5, audio=True,
        # 运行时 @Image1=上一镜末帧;@Image2=军官肖像(两人同像)
        prompt=("画面从@Image1精确开始——@Image1是上一镜的最后一帧,"
                "双人构图与光线与其完全一致,不重新构图。右侧军官(容貌"
                "同@Image2)缓缓转过脸,望向画面外远处,眉间浮起怜悯,"
                "低声回应:“真可怜啊,公爵千金……”左侧军官(容貌同"
                "@Image2)沉默地观察他的反应。说完后两人恢复并肩静止,"
                f"镜头始终固定。{AUDIO_TAIL}")),
    dict(  # 7 持折扇女子讥讽(t2v 新构图;其前将插入 桥67)
        mode="t2v", refs=["持折扇女子"], duration=6, audio=True,
        prompt=("下半脸特写,固定镜头:@Image1以黑色蕾丝折扇微微遮住下半"
                "张脸,只露出眼睛与扇沿,身后宾客虚化。她一边缓缓收拢"
                "折扇,一边露出轻蔑上扬的嘴角,望向画外低声讥讽:“真可怜"
                "啊,公爵千金……”说完后折扇收拢停在下巴旁,轻蔑笑意"
                f"定格,镜头静止。{AUDIO_TAIL}")),
]

# (前镜索引, 后镜索引, 桥 prompt):桥插在后镜之前。
# 换人接缝全部架桥:3→4(安娜→安莉希娅+王子)、4→5(二人→军官)、
# 6→7(军官→持折扇女子);2→3 安娜两侧皆在场,保留硬切反打。
BRIDGES = [(2, 3, BRIDGE_34), (3, 4, BRIDGE_45), (5, 6, BRIDGE_67)]


def main() -> None:
    parsed = parse_script_json(
        Path("/Users/kevin/Desktop/script-wedding/script.json"))
    roles = parsed["roles"]
    out_dir = Path("outputs") / f"seedance_manual_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({"name": "wavespeed", "resolution": "720p",
                          "call_log": str(out_dir / "wavespeed_calls.jsonl")})
    shots_out: list = [None] * len(SHOTS)
    ledger = []

    for i, sh in enumerate(SHOTS):
        outp = out_dir / f"shot{i:03d}.mp4"
        vg.generate_audio = bool(sh["audio"])
        kw = {}
        tag = sh["mode"]
        if sh["mode"] == "pin":
            prev_v = shots_out[i - 1]
            lf = _last_frame(Path(prev_v),
                             out_dir / f"pin_{i:03d}.png") if prev_v else None
            if lf is None:
                print(f"[shot {i+1}] 上镜末帧缺失 — 软钉降级为纯 refs")
                kw["reference_images"] = [roles[n] for n in sh["refs"]]
                tag = "t2v_degraded"
            else:
                # ti2v 软钉:末帧本身 = @Image1,肖像顺延 @Image2…
                kw["reference_images"] = [lf] + [roles[n]
                                                 for n in sh["refs"]]
                tag = "ti2v_pin"
        else:
            kw["reference_images"] = [roles[n] for n in sh["refs"]]
        print(f"[shot {i+1}/{len(SHOTS)}] {tag} {sh['duration']}s "
              f"audio={sh['audio']}")
        try:
            vg.generate(sh["prompt"], sh["duration"], outp, fps=24,
                        seed=0, **kw)
            shots_out[i] = outp
            ledger.append({"shot": i, "mode": tag, "ok": True,
                           "prompt": sh["prompt"]})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            ledger.append({"shot": i, "mode": tag, "ok": False,
                           "error": str(exc)[:300]})

    # 转场桥:末帧(前)→ 首帧(后),flf2v,只写运镜(时长按后端下限吸附)
    bridge_before: dict = {}
    if hasattr(vg, "frame_to_frame"):
        for a, b, bprompt in BRIDGES:
            if not (shots_out[a] and shots_out[b]):
                continue
            try:
                fa = _last_frame(Path(shots_out[a]),
                                 out_dir / f"bridge_{a}{b}_prev.png")
                fb = _extract_frame0(Path(shots_out[b]),
                                     out_dir / f"bridge_{a}{b}_next.png")
                if fa is None or fb is None:
                    continue
                vg.generate_audio = False
                bp = vg.frame_to_frame(
                    prompt=bprompt, first_frame=fa, last_frame=fb,
                    out_path=out_dir / f"bridge_{a}{b}.mp4",
                    duration=4, seed=777)
                bridge_before[b] = Path(bp)
                print(f"[桥 {a+1}→{b+1}] OK")
                ledger.append({"bridge": [a, b], "ok": True})
            except Exception as exc:
                print(f"[桥 {a+1}→{b+1}] FAILED: {exc} — 硬切保底")
                ledger.append({"bridge": [a, b], "ok": False,
                               "error": str(exc)[:300]})

    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))

    clips = []
    for i, v in enumerate(shots_out):
        if i in bridge_before:
            clips.append(bridge_before[i])
        if v is not None:
            clips.append(v)
    if clips:
        concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
            if any_audio(clips) else clips
        final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
        print("成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
