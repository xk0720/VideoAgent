"""按 hook 人物数量给镜头分配参考人物(测试链路的"极简选角")。

sequential : 时间轴连续均分 —— 前 1/K 的镜头归 hook_1, 依此类推
             (成片观感 = 分段换人)。
round_robin: 逐镜轮转 —— 相邻镜头换不同的人(成片观感 = 快速交替)。

注意: 超长镜头被切块后的子镜头必须跟原镜头同一个人, 所以分配按
"原镜头"计数, 块继承。
"""
import logging
from typing import Dict, List

from interfaces import Shot

log = logging.getLogger("hook_remake")


def assign_hooks(shots: List[Shot], hook_slots: List[str],
                 strategy: str = "sequential") -> Dict[int, str]:
    if not hook_slots:
        raise ValueError("没有可用的 person_hook")

    # 原镜头编号列表(切块的子镜头折叠回原镜头)
    logical_ids: List[int] = []
    for s in shots:
        lid = s.chunk_of if s.chunk_of is not None else s.idx
        if lid not in logical_ids:
            logical_ids.append(lid)

    n, k = len(logical_ids), len(hook_slots)
    if strategy == "round_robin":
        logical_map = {lid: hook_slots[i % k] for i, lid in enumerate(logical_ids)}
    elif strategy == "sequential":
        # i*K//N 产生均衡的连续分块(块大小差至多 1)
        logical_map = {lid: hook_slots[i * k // n] for i, lid in enumerate(logical_ids)}
    else:
        raise ValueError(f"未知分配策略: {strategy}")

    assignment = {
        s.idx: logical_map[s.chunk_of if s.chunk_of is not None else s.idx]
        for s in shots
    }
    for slot in hook_slots:
        cnt = sum(1 for v in assignment.values() if v == slot)
        log.info("分配(%s): %s ← %d 个镜头", strategy, slot, cnt)
        if cnt == 0:
            log.warning("%s 没有分到镜头(镜头数 < hook 数)", slot)
    return assignment
