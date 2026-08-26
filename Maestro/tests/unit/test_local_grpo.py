"""rl/local 本地推理 GRPO 单测(2026-08-21)。零 GPU:模型/分词器注入
假件,只验我们自己的逻辑——那正是历次事故的所在地。"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl"))

torch = pytest.importorskip("torch")

from rl.local import config as C                                # noqa: E402
from rl.local.broadcast import (AdapterPublisher, AdapterSubscriber,  # noqa: E402
                                GroupQueue)
from rl.local.policy import LocalPolicy                         # noqa: E402
from rl.local.trainer import group_advantage, train_one_group   # noqa: E402


# ── 假件 ──────────────────────────────────────────────────────────────
class FakeTok:
    """chat 模板加两头哨兵,好让测试能看出"模板有没有套上"。"""
    pad_token_id, eos_token_id = 0, 2

    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            tokenize=True, enable_thinking=False):
        body = [10 + (ord(c) % 50) for c in msgs[0]["content"][:8]]
        head = [101] + ([777] if enable_thinking else [])
        return head + body + [102]          # 102 = <assistant> 生成起点

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


class FakeModel:
    """确定性 logits:词表 8,分布只依赖上一个 token,便于精确对账。"""
    VOCAB = 8

    def __init__(self):
        self.adapter_on = True
        self.calls = []
        # 真实可训练叶子:让 logits 确实依赖参数,反向传播才成立
        self.w = torch.zeros(1, requires_grad=True)

    # 只认新名字 —— 假件刻意【不】接受 num_logits_to_keep,这样一旦代码
    # 退回旧参数名,探针会判定截断失效并落到全量分支,测试立刻抓到
    def __call__(self, ids, logits_to_keep=None, use_cache=None, **kw):
        n = ids.shape[1]
        keep = logits_to_keep or n
        self.calls.append({"len": n, "keep": keep, "use_cache": use_cache})
        base = torch.arange(self.VOCAB, dtype=torch.float32)
        rows = []
        for pos in range(n - keep, n):
            prev = int(ids[0, pos])
            # adapter 改变分布【形状】而非整体平移 ——
            # log_softmax 对常数平移不变,平移式假件测不出任何差别
            scale = 1.0 if self.adapter_on else 0.35
            rows.append(base * 0.1 * scale + prev * 0.01 + self.w)
        return type("O", (), {"logits": torch.stack(rows).unsqueeze(0)})()

    def generate(self, ids, num_return_sequences=1, **kw):
        out = []
        for j in range(num_return_sequences):
            out.append(torch.cat([ids[0],
                                  torch.tensor([3 + j, 4, 5])]))
        return torch.stack(out)

    def disable_adapter(self):
        model = self

        class _Ctx:
            def __enter__(self_):
                model.adapter_on = False

            def __exit__(self_, *a):
                model.adapter_on = True
        return _Ctx()

    def parameters(self):
        return [self.w]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
        (Path(path) / "adapter_model.safetensors").write_bytes(b"fake")

    def load_adapter(self, path, adapter_name=None, is_trainable=False):
        self.loaded_from = str(path)

    def set_adapter(self, name):
        pass


def _policy(**over):
    hp = C.HParams(**{"group": 4, "max_new_tokens": 3, **over})
    return LocalPolicy(FakeModel(), FakeTok(), hp, device="cpu")


# ── ① chat 模板:采样与训练拿到的是同一串 id(旧路径的头号 bug)──────
def test_prompt_ids_come_from_chat_template():
    p = _policy()
    ids = p.encode_prompt("深夜便利店")
    assert ids[0] == 101 and ids[-1] == 102, "模板头尾哨兵必须在"
    assert 777 not in ids, "enable_thinking 默认关"
    assert _policy(enable_thinking=True).encode_prompt("x")[1] == 777


def test_sampling_and_training_share_the_same_ids_and_logprob():
    """采样时记的 old_logprob,必须与训练端用同一函数重算的结果一致。
    这是"训练=采样"这条铁律的硬锁。"""
    p = _policy()
    samples = p.sample_group("深夜便利店", k=3)
    assert len(samples) == 3
    for s in samples:
        assert s.prompt_ids == p.encode_prompt("深夜便利店")
        again = p.seq_logprob(s.prompt_ids, s.response_ids, s.temperature)
        assert torch.allclose(torch.tensor(s.logp_old), again, atol=1e-5)


def test_temperature_is_applied_to_logits():
    p = _policy()
    ids = p.encode_prompt("x")
    hot = p.seq_logprob(ids, [3, 4], temperature=2.0)
    cold = p.seq_logprob(ids, [3, 4], temperature=0.5)
    assert not torch.allclose(hot, cold), "温度没进 logits 就是错的"


def test_only_completion_logits_are_computed():
    """logits_to_keep = R+1 —— 这是 batch 上不去那堵显存墙的解法。"""
    p = _policy()
    ids = p.encode_prompt("x")
    p.seq_logprob(ids, [3, 4, 5], temperature=1.0)
    assert p.logits_kwarg() == "logits_to_keep", "探针必须认出新参数名"
    assert p.model.calls[-1]["keep"] == 4          # R+1
    assert p.model.calls[-1]["len"] == len(ids) + 3
    assert p.model.calls[-1]["use_cache"] is False, "训练前向必须关 KV cache"


class SwallowingModel(FakeModel):
    """复刻真实事故:forward 带 **kwargs,截断参数被【静默吞掉】——
    不报错、不告警,只是默默把全序列 logits 都算出来。"""

    def __call__(self, ids, **kw):                 # 吞掉一切,不截断
        return FakeModel.__call__(self, ids)


def test_correctness_does_not_depend_on_the_truncation_kwarg():
    """核心回归:截断失效时显存会高,但【算出来的数必须一模一样】。
    负索引切窗口就是为此 —— 正确性不能押在一个参数名上。"""
    hp = C.HParams(group=4, max_new_tokens=3)
    good = LocalPolicy(FakeModel(), FakeTok(), hp, device="cpu")
    bad = LocalPolicy(SwallowingModel(), FakeTok(), hp, device="cpu")
    ids = good.encode_prompt("深夜便利店")
    a = good.seq_logprob(ids, [3, 4, 5], temperature=0.7)
    b = bad.seq_logprob(ids, [3, 4, 5], temperature=0.7)
    assert bad.logits_kwarg() is None, "探针必须诚实报告截断没生效"
    assert torch.allclose(a, b, atol=1e-6), "截断与否不得改变数值"


def test_selftest_grad_catches_dead_backward():
    """PEFT + 梯度检查点少一步设置 → 反向静默不回传、梯度恒零。
    这个自检就是为拦住它而存在的。"""
    p = _policy()
    assert p.selftest_grad() > 0

    class Frozen(FakeModel):
        def parameters(self):
            leaf = torch.zeros(1, requires_grad=True)
            return [leaf]                          # 与图无关的孤立叶子
    dead = LocalPolicy(Frozen(), FakeTok(), C.HParams(), device="cpu")
    assert dead.selftest_grad() == 0.0


def test_ref_context_toggles_and_restores():
    p = _policy()
    ids = p.encode_prompt("x")
    on = p.seq_logprob(ids, [3], temperature=1.0)
    with p.ref_context():
        off = p.seq_logprob(ids, [3], temperature=1.0)
    after = p.seq_logprob(ids, [3], temperature=1.0)
    assert not torch.allclose(on, off), "θ_ref 必须与 θ 不同"
    assert torch.allclose(on, after), "退出后必须恢复"
    assert p.model.adapter_on is True


def test_sampling_never_truncates_the_distribution():
    """top_p=1 / top_k=0:否则真实行为分布 ≠ softmax(logits/T),
    记下来的 old_logprob 就是错的(用户裁决①)。"""
    hp = C.HParams()
    assert hp.top_p == 1.0 and hp.top_k == 0


# ── ② 优势与损失 ─────────────────────────────────────────────────────
def test_group_advantage_divides_by_std():
    adv = group_advantage([1.0, 0.0, 0.0, 0.0])
    assert abs(sum(adv)) < 1e-6                    # 组内和恒为 0
    assert abs(adv[0] - 1.5) < 0.01                # (1-0.25)/0.5
    assert group_advantage([0.5] * 4) == [0.0] * 4  # 打平 → 全 0
    assert group_advantage([0.7]) == [0.0]          # 单样本无基线


def test_zero_advantage_group_is_skipped():
    p = _policy()
    opt = torch.optim.SGD(p.model.parameters(), lr=1e-3)
    ids = p.encode_prompt("x")
    g = {"samples": [{"prompt_ids": ids, "response_ids": [3],
                      "logp_old": [-1.0], "reward": 0.5,
                      "sample_temperature": 0.7} for _ in range(4)]}
    assert train_one_group(p, opt, g, p.hp)["skipped"] == "zero_advantage"


def test_ratio_clipping_and_kl_are_live():
    """logp_old 被人为压得很低 → ratio 远大于 1 → 必须被裁剪计数。"""
    p = _policy()
    opt = torch.optim.SGD(p.model.parameters(), lr=1e-3)
    ids = p.encode_prompt("x")
    g = {"samples": [
        {"prompt_ids": ids, "response_ids": [3], "logp_old": [-9.0],
         "reward": r, "sample_temperature": 0.7} for r in (1.0, 0.0, 0.0, 0.0)]}
    m = train_one_group(p, opt, g, p.hp)
    assert "skipped" not in m
    assert m["ratio"] > 1.0 and m["clipfrac"] > 0.0
    assert m["kl"] != 0.0, "KL 项必须真的在算"
    assert m["group_size"] == 4
    # 指标必须是【归一化前】的组内奖励 std —— 归一化后的优势 RMS 是
    # 常数 √((K−1)/K)=0.866,同义反复(2026-08-25 实测发现后换掉)
    assert "adv_std" not in m
    assert abs(m["reward_std"] - 0.5) < 1e-9   # [1,0,0,0] 的无偏 std


# ── ③ 广播与队列 ─────────────────────────────────────────────────────
def test_broadcast_cadence_and_atomic_version(tmp_path):
    hp = C.HParams(broadcast_every=4, keep_adapters=2)
    pub = AdapterPublisher(hp, root=tmp_path)
    p = _policy()
    published = [pub.maybe_publish(p, s) for s in range(1, 13)]
    assert [x for x in published if x is not None] == [1, 2, 3]
    assert (tmp_path / "VERSION").read_text() == "3"
    assert not (tmp_path / "v1").exists(), "只留 keep_adapters 代"

    sub = AdapterSubscriber(tmp_path)
    assert sub.maybe_reload(p) == 3 and p.version == 3
    assert sub.maybe_reload(p) is None, "同版本不该重复换脑"


def test_group_queue_atomic_and_claim_once(tmp_path):
    q = GroupQueue(tmp_path / "q", tmp_path / "q/claimed")
    q.put({"label": "s1", "policy_version": 2,
           "samples": [{"reward": 1.0}, {"reward": 0.0}]})
    assert q.depth() == 1
    assert not list((tmp_path / "q").glob("*.tmp")), "临时文件必须已改名"
    g, path = q.claim()
    assert g["label"] == "s1" and q.depth() == 0
    assert q.claim() == (None, None), "认领过的不该再被拿到"
    q.done(path)
    assert not path.exists()


def test_archive_survives_prune_and_rollback_republishes(tmp_path):
    """峰值回滚三件套:① 归档不受滚动删除影响;② 回滚 = 老版本内容
    以新版本号重新发布(订阅端只认版本变大);③ 目标不存在时点名报错。"""
    from rl.local.broadcast import rollback_adapter
    hp = C.HParams(broadcast_every=1, keep_adapters=2, archive_every=2)
    pub = AdapterPublisher(hp, root=tmp_path)
    p = _policy()
    for s in range(1, 6):
        pub.maybe_publish(p, s)                     # v1..v5
    assert not (tmp_path / "v2").exists(), "live 只留 2 代,v2 应被滚掉"
    assert (tmp_path / "archive/v2/adapter_config.json").exists(), \
        "v2 是归档版,必须在 archive/ 里活着"
    assert (tmp_path / "archive/v4").exists()
    assert not (tmp_path / "archive/v5").exists(), "5 不是归档节拍"

    # 归档自含 reward 信息:META.json 内嵌,不依赖中央账本
    pub.annotate_archive(2, {"v": 2, "reward_recent": 0.71})
    pub.annotate_archive(3, {"v": 3})           # 非归档版:静默跳过
    import json as _json
    assert _json.loads((tmp_path / "archive/v2/META.json")
                       .read_text())["reward_recent"] == 0.71
    assert not (tmp_path / "archive/v3").exists()

    new = rollback_adapter(2, root=tmp_path)        # 峰值在 v2
    assert new == 6
    assert (tmp_path / "VERSION").read_text() == "6"
    assert (tmp_path / "v6/adapter_config.json").exists()
    sub = AdapterSubscriber(tmp_path)
    assert sub.maybe_reload(p) == 6, "订阅端应把回滚版当新版换用"

    with pytest.raises(FileNotFoundError):
        rollback_adapter(99, root=tmp_path)


def test_best_version_is_always_kept(tmp_path):
    """2026-08-25 用户令:始终保存 reward_mean 最优的那版权重。
    打擂台:创新高才更新 best/,打不过不动;重启后历史最优仍算数;
    --to best 可直接回滚到它。"""
    import json as _json

    from rl.local.broadcast import rollback_adapter
    hp = C.HParams(broadcast_every=1, keep_adapters=2)
    pub = AdapterPublisher(hp, root=tmp_path)
    p = _policy()
    for s, r in ((1, 0.60), (2, 0.75), (3, 0.70)):   # 峰值在 v2
        pub.maybe_publish(p, s)
        pub.maybe_save_best(s, r)
    meta = _json.loads((tmp_path / "best/BEST.json").read_text())
    assert meta["v"] == 2 and meta["reward"] == 0.75, "best 必须是峰值版"
    assert (tmp_path / "best/adapter_config.json").exists()

    # 模拟重启:新 publisher 也打不过盘上的历史最优
    pub2 = AdapterPublisher(hp, root=tmp_path)
    pub2.maybe_publish(p, 4)
    assert pub2.maybe_save_best(4, 0.74) is False
    assert _json.loads((tmp_path / "best/BEST.json").read_text())["v"] == 2

    assert rollback_adapter("best", root=tmp_path) == 5


def test_stale_claims_are_recovered(tmp_path):
    q = GroupQueue(tmp_path / "q", tmp_path / "q/claimed")
    q.put({"label": "s", "samples": []})
    _g, path = q.claim()
    import os
    os.utime(path, (0, 0))                    # 假装很久没处理
    assert q.recover_stale_claims(older_than_s=10) == 1
    assert q.depth() == 1


# ── ④ 决策解析与任务池 ───────────────────────────────────────────────
def test_parse_decision_matches_production_brain_pick():
    """本地解析必须与 window_core._brain_pick 的纪律逐条一致。"""
    import env.window_core as W
    from rl.local.stream import parse_decision
    menu = [{"name": "ref2v", "description": "x"},
            {"name": "t2v", "description": "y"}]
    raw = json.dumps({"strategy": "ref2v", "reason": "r",
                      "video_prompt": " p ", "images": [{"source": "t2i",
                                                         "description": "d"}],
                      "junk": 1}, ensure_ascii=False)
    got = parse_decision(raw, menu, W._CONDITION_PRIORITY)
    assert got["strategy"] == "ref2v" and got["via"] == "llm"
    assert got["video_prompt"] == "p"          # strip 过
    assert got["images"] == [{"source": "t2i", "description": "d"}]
    assert "junk" not in got                   # 机械字段丢弃

    # 菜单越界 / 不可解析 → 兜底,且 via=fallback(r_format 会记 0)
    for bad in (json.dumps({"strategy": "flf2v_bridge"}), "我拒绝输出 JSON"):
        d = parse_decision(bad, menu, W._CONDITION_PRIORITY)
        assert d["via"] == "fallback" and d["strategy"] in {"ref2v", "t2v"}


def test_task_pool_is_striped_across_workers(tmp_path):
    from rl.local.stream import pick_task
    pool = tmp_path / "pool.yaml"
    pool.write_text(
        "mix: {screenplay_weight: 3, idea_weight: 2}\n"
        "screenplays:\n" + "".join(
            f"  - {{file: f{i}.json, prompt: p{i}}}\n" for i in range(6)) +
        "ideas:\n" + "".join(f"  - idea{i}\n" for i in range(4)))
    W = 3
    first = [pick_task(str(pool), 0, W, w)["prompt"] for w in range(W)]
    assert len(set(first)) == W, "同一轮里 N 条流不能撞同一个任务"
