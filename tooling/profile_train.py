# -*- coding: utf-8 -*-
"""
profile_train.py —— 找出 train.py 的 360 秒到底花在哪

单次前向只要 0.36ms，60 轮 × 22 步的纯矩阵运算合计约 1.5 秒。
所以必须逐步计时，定位真正的瓶颈（而不是猜）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import ART, DISH_CAT_IDX, N_DISH
import train as T


def tic():
    return time.perf_counter()


def toc(t0, label):
    dt = time.perf_counter() - t0
    print(f"  {label:<52s} {dt:8.3f} s")
    return dt


print("=" * 74)
print("逐阶段计时")
print("=" * 74)

t = tic()
samples = pd.read_csv(ART / "samples.csv")
scenarios = pd.read_csv(ART / "scenarios.csv")
for df in (samples, scenarios):
    df["taboo"] = df["taboo"].fillna("").astype(str)
toc(t, "读 CSV + 清洗")

tr = T.attach_gold(samples[samples.split == "train"].reset_index(drop=True))
va = T.attach_gold(samples[samples.split == "val"].reset_index(drop=True))
P = T._pack(tr)
V = T._pack(va)
toc(t := tic(), "attach_gold + _pack（特征编码）")

groups: dict = {}
for r, sid in enumerate(P["sid"]):
    groups.setdefault(sid, []).append(r)
group_list = list(groups.values())
toc(t := tic(), "按情境分组（group_list）")

# --- 关键嫌疑：group_targets 是纯 Python 双层循环，每步都调用一次 ---
idx = np.arange(64)
t = tic()
for _ in range(22 * 60):                      # 一个完整训练的所有 step 数
    T.group_targets(P, idx)
t_group = toc(t, "group_targets：全部 step（22×60 次，每次 64 行）")

# --- 对比：矢量化写法能快多少 ---
cat_arr = P["cat_id"]
chosen_arr = P["chosen"]
y_arr = P["y"]
dish_cat = DISH_CAT_IDX[:, 0]


def group_targets_fast(P, idx):
    """矢量化版本：用广播一次性算出 [B, N_DISH] 的目标矩阵"""
    B_ = len(idx)
    cid = P["cat_id"][idx][:, None]                 # [B,1]
    gm = (dish_cat[None, :] == cid)                 # [B,47] 同品类
    # 该情境下 chosen 且属于该品类的行，取最大效用
    same_scen = (P["sid"][:, None] == P["sid"][idx][None, :])   # 太慢，不用
    return gm


t = tic()
for _ in range(22 * 60):
    gt = T.group_targets(P, idx)
t_slow = toc(t, "（重复确认）group_targets 全部 step")

# 用「预先算好每个情境的目标」代替每次重算
t = tic()
tgt_cache = {}
for sid in set(P["sid"]):
    rows = groups[sid]
    tgt_cache[sid] = T.group_targets(P, np.array(rows))
n_cache = len(tgt_cache)
t_build = toc(t, f"预计算全部 {n_cache} 个情境的目标（一次）")

print()
print("=" * 74)
print("其他阶段")
print("=" * 74)
t = tic()
T._eval_rank
toc(t, "占位")
print(f"\n  实测 train.py 总耗时 ≈ 360 s")
print(f"  group_targets 独占     ≈ {t_slow:.1f} s  ({100 * t_slow / 360:.0f}%)")
