# -*- coding: utf-8 -*-
"""训练耗时剖析：定位两层各自的时间去向。

目的：回答「训练为什么这么慢、慢在哪一步」。
做法：按真实训练的超参复现一个 batch / 一个 epoch 的关键步骤并计时。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\123\Desktop\桌面应用\文档\meal-decision")
sys.path.insert(0, str(BASE))

import train as T  # noqa: E402
from features import N_DISH, N_CAT, DISH_CAT_IDX, emb_dim  # noqa: E402

SAMPLES = pd.read_csv(BASE / "artifacts" / "samples.csv")
SAMPLES["taboo"] = SAMPLES["taboo"].fillna("").astype(str)

tr = T.attach_gold(SAMPLES[SAMPLES.split == "train"].reset_index(drop=True))
P = T._pack(tr)

groups = {}
for r, sid in enumerate(P["sid"]):
    groups.setdefault(sid, []).append(r)
group_list = list(groups.values())
n_groups = len(group_list)
rows_per_group = len(tr) / n_groups

print("=" * 70)
print("训练规模")
print("=" * 70)
print(f"  N_DISH={N_DISH}  N_CAT={N_CAT}  BATCH(情境)={T.BATCH}  EPOCHS={T.EPOCHS}")
print(f"  训练行数={len(tr)}  训练情境数={n_groups}  平均每情境 {rows_per_group:.1f} 行")
print(f"  每 epoch 的 batch 数 = {int(np.ceil(n_groups / T.BATCH))}")
print(f"  dish_id_emb={emb_dim('dish_id', N_DISH)}  cat_emb={emb_dim('category', N_CAT)}")

# ---- 1) 两层各自的参数量 ----
rng = np.random.default_rng(0)
cm = T.CuisineNet(rng)
dm = T.DishScorer(rng)
n_c = sum(v.size for v in cm.params().values())
n_d = sum(v.size for v in dm.params().values())
print(f"  参数量: CuisineNet={n_c:,}  DishScorer={n_d:,}  合计={n_c+n_d:,}")

# ---- 2) 一个 batch 的分解计时 ----
print()
print("=" * 70)
print("单个 batch 分解计时（batch=64 情境，全部菜品参与打分）")
print("=" * 70)
perm = np.random.default_rng(7).permutation(n_groups)
batch = [r for gi in perm[:T.BATCH] for r in group_list[gi]]
b = np.array(batch)
B_ = len(b)
all_ids = np.arange(N_DISH)[None, :]
print(f"  batch 行数 B_={B_}  ->  打分矩阵 {B_} x {N_DISH} = {B_*N_DISH:,} 个 (情境,菜品) 对")

t0 = time.time()
ids, dc, dn = T.tile_dishes(all_ids, B_)
t_tile = time.time() - t0

ck, nk = P["cat"][b], P["num"][b]
t0 = time.time()
s = dm.forward(ck, nk, ids, dc, dn)
t_fwd = time.time() - t0

gt = T.group_targets(P, b)
t_gt = time.time() - t0 - t_fwd

is_chosen = P["chosen"][b]
group = (DISH_CAT_IDX[:, 0][None, :] == P["cat_id"][b][:, None])
s_m = np.where(group, s, -1e9)
gt_soft = T.softmax_rows(gt)
loss_rank, dp = T.softmax_ce_soft(s_m[is_chosen], gt_soft[is_chosen])
ds = np.zeros_like(s)
ds[is_chosen] = dp
obs = np.zeros_like(s)
obs[np.arange(B_), P["dish"][b]] = 1.0
pred_obs = (s * obs).sum(1)
t0 = time.time()
dm.backward(ds)
t_bwd = time.time() - t0

t_batch = t_tile + t_fwd + t_gt + t_bwd
print(f"  tile_dishes          {t_tile*1000:8.1f} ms")
print(f"  DishScorer.forward   {t_fwd*1000:8.1f} ms")
print(f"  group_targets+loss   {t_gt*1000:8.1f} ms")
print(f"  DishScorer.backward  {t_bwd*1000:8.1f} ms")
print(f"  --------------------------------------")
print(f"  合计/batch           {t_batch*1000:8.1f} ms")

n_batch = int(np.ceil(n_groups / T.BATCH))
print()
print(f"  推算第 2 层每 epoch ({n_batch} batch): {t_batch*n_batch:.1f}s")
print(f"  推算第 2 层全训练 ({T.EPOCHS} epoch): {t_batch*n_batch*T.EPOCHS/60:.1f} 分钟")

# ---- 3) 有效计算占比：多少打分行是「有用的」 ----
n_chosen = int(is_chosen.sum())
n_used = int(is_chosen.sum()) + B_          # 排序只用 chosen，回归每行用 1 个
print()
print("=" * 70)
print("浪费在哪里")
print("=" * 70)
print(f"  batch 里 chosen 行 = {n_chosen} / {B_}")
print(f"  但为这 {B_} 行算了全部 {N_DISH} 道菜的分 = {B_*N_DISH:,} 个分数")
print(f"  其中参与排序损失的点数 = chosen行 x 该行品类菜数 ≈ "
      f"{n_chosen} x {N_DISH/N_CAT:.1f} = {n_chosen*N_DISH/N_CAT:.0f}")
print(f"  -> 有效利用率 ≈ {n_chosen*N_DISH/N_CAT/(B_*N_DISH):.1%}")
print(f"  -> 重复度：同一情境平均有 {rows_per_group:.1f} 行，"
      f"每行都把该情境 x 全部菜品重算了一遍（重复 {rows_per_group:.1f}x）")
