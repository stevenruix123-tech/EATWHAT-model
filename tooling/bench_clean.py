# -*- coding: utf-8 -*-
"""
bench_clean.py —— 干净对照：前向/反向各阶段，同时测 CPU 时间与墙钟时间

目的：区分「真的算得慢」和「被别的进程抢了 CPU」。
  - 墙钟时间  = 实际等待
  - CPU 时间  = 真正占用处理器的时间
若 CPU 时间 << 墙钟时间，说明时间花在等待/争抢上，不是计算量的问题。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from features import ART, DISH_CAT_IDX, N_DISH
import train as T


def measure(label, fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        fn()
    wall = (time.perf_counter() - w0) / n * 1000
    cpu = (time.process_time() - c0) / n * 1000
    print(f"  {label:<34s} 墙钟 {wall:8.3f} ms   CPU {cpu:8.3f} ms   "
          f"CPU/墙钟 {cpu / max(wall, 1e-9):5.2f}")
    return wall, cpu


def main():
    samples = pd.read_csv(ART / "samples.csv")
    samples["taboo"] = samples["taboo"].fillna("").astype(str)
    tr = T.attach_gold(samples[samples.split == "train"].reset_index(drop=True))
    P = T._pack(tr)
    model = T.DishScorer(np.random.default_rng(T.SEED + 1))

    b = np.arange(0, 64)
    B_ = len(b)
    ids, dc, dn = T.tile_dishes(np.arange(N_DISH)[None, :], B_)
    cat, num = P["cat"][b], P["num"][b]

    print("=" * 86)
    print("权重 dtype 检查（float32 vs float64 会影响速度）")
    print("=" * 86)
    for name, arr in [("W1", model.W1), ("W2", model.W2), ("dish_id_emb", model.dish_id_emb),
                      ("enc.W1", model.enc.W1), ("x(输入)", cat), ("num", num)]:
        print(f"  {name:<14s} {arr.dtype}  shape={arr.shape}")

    print()
    print("=" * 86)
    print("前向 / 反向 分阶段（n=30，CPU 时间与墙钟时间对比）")
    print("=" * 86)
    measure("① 前向 forward", lambda: model.forward(cat, num, ids, dc, dn))

    s = model.forward(cat, num, ids, dc, dn)
    ds = np.random.default_rng(0).normal(size=s.shape).astype(s.dtype) * 0.01
    measure("② 反向 backward（整体）", lambda: model.backward(ds))

    # 逐行
    _, _, _, x, h_pre, h = model.cache
    measure("   gW2 einsum", lambda: np.einsum("bkh,bk->h", h, ds)[:, None])
    measure("   dh_pre", lambda: (ds[:, :, None] @ model.W2.T) * (h_pre > 0))
    dh_pre = (ds[:, :, None] @ model.W2.T) * (h_pre > 0)
    measure("   gW1 einsum", lambda: np.einsum("bkd,bkh->dh", x, dh_pre))
    dx = dh_pre @ model.W1.T
    measure("   dx @ W1.T", lambda: dh_pre @ model.W1.T)
    measure("   情境 d_ctx mean", lambda: dx[:, :, :model.d_ctx].mean(1))

    # 关键嫌疑：encoder 反向里的 embedding scatter
    d_ctx = dx[:, :, :model.d_ctx].mean(1)
    measure("   enc.backward（整体）", lambda: model.enc.backward(d_ctx, reduce_k=True))

    print()
    print("=" * 86)
    print("对照：孤立小矩阵（不涉及模型状态）")
    print("=" * 86)
    rng = np.random.default_rng(0)
    xf = rng.normal(size=(64, 47, 61)).astype(np.float32)
    wf = (rng.normal(size=(61, 48)) * 0.1).astype(np.float32)
    measure("float32 (64,47,61)x(61,48)", lambda: xf @ wf)
    xd = xf.astype(np.float64)
    wd = wf.astype(np.float64)
    measure("float64 (64,47,61)x(61,48)", lambda: xd @ wd)
    measure("float32 同样大小 einsum", lambda: np.einsum("bkd,dh->bkh", xf, wf))

    print()
    print("=" * 86)
    print("结论数据：一次完整训练 step（前向+损失+反向+优化器）")
    print("=" * 86)


if __name__ == "__main__":
    main()
