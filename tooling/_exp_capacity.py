# -*- coding: utf-8 -*-
"""快速容量实验：只训练第二层 DishScorer，比较不同 embedding 上限下的排序质量。

背景：菜品库从 47 道扩到 317 道后，品类内候选池从平均 5.9 道涨到 17.6 道，
但 emb_dim 的上限 12 是给 47 道菜标定的。怀疑 dish_id_emb 装不下区分度。

这个脚本刻意用「短轮数 + 只用 chosen 行」来快速比较方案排序，
不追求复现完整训练的绝对数值，只看不同配置之间的**相对**差异。
"""
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\123\Desktop\桌面应用\文档\meal-decision")
sys.path.insert(0, str(BASE))

import train as T  # noqa: E402  先按默认 cap 导入一次，后续按 cap 重新加载

SAMPLES = pd.read_csv(BASE / "artifacts" / "samples.csv")
SAMPLES["taboo"] = SAMPLES["taboo"].fillna("").astype(str)

SHORT_EPOCHS = int(os.environ.get("EXP_EPOCHS", "12"))
CAPS = [int(c) for c in os.environ.get("EXP_CAPS", "12,24,36,48").split(",")]


def reload_with_cap(cap: int):
    """按给定 EMB_DIM_CAP 重新加载 features / train，返回新模块。"""
    for m in ("features", "train", "build_dataset"):
        sys.modules.pop(m, None)
    os.environ["EMB_DIM_CAP"] = str(cap)
    return importlib.import_module("train")


def topk_hit(scores, target):
    """target 是否落在 scores 的 top-k。scores: [N, K] 组内分, target: [N] 组内索引"""
    if scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, axis=1)
    rank = np.argmax(order == target[:, None], axis=1)
    return float((rank == 0).mean()), float((rank < 3).mean())


print(f"短轮数={SHORT_EPOCHS}  对比 EMB_DIM_CAP={CAPS}")
print(f"数据：{len(SAMPLES)} 行 / {SAMPLES.scenario_id.nunique()} 情境")
print()

results = []
for cap in CAPS:
    T2 = reload_with_cap(cap)
    from features import N_DISH, N_CAT, emb_dim
    dim = emb_dim("dish_id", N_DISH)
    T2.EPOCHS = SHORT_EPOCHS          # 覆盖轮数，做快速比较
    T2.SEED = 7

    tr = T2.attach_gold(SAMPLES[SAMPLES.split == "train"].reset_index(drop=True))
    va = T2.attach_gold(SAMPLES[SAMPLES.split == "val"].reset_index(drop=True))
    P, V = T2._pack(tr), T2._pack(va)

    t0 = time.time()
    model, hist = T2.train_dish(SAMPLES, verbose=False)
    dt = time.time() - t0

    Pv = T2._pack(va)
    vr = T2._eval_rank(model, Pv)
    t1 = T2._eval_top1(model, Pv)
    t3 = T2._eval_topk(model, Pv, k=3)

    results.append(dict(cap=cap, dish_dim=dim, val_rank=vr, top1=t1, top3=t3, secs=dt))
    print(f"cap={cap:>2}  dish_id_emb={dim:>2}  val_rank={vr:.5f}  "
          f"品类内top1={t1:6.2%}  品类内top3={t3:6.2%}  ({dt:.0f}s)", flush=True)

print()
best = max(results, key=lambda r: r["top3"])
base12 = [r for r in results if r["cap"] == 12][0]
print(f"按品类内 top-3 选：EMB_DIM_CAP={best['cap']} "
      f"(top3={best['top3']:.2%}, top1={best['top1']:.2%})")
print(f"默认 cap=12:        top3={base12['top3']:.2%}, top1={base12['top1']:.2%}")
