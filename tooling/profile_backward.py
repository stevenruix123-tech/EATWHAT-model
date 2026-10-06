# -*- coding: utf-8 -*-
"""
profile_backward.py —— 把 DishScorer.backward 拆到每一行计时

已定位：反向传播占训练时间 83.9%（1.82s/步），而前向只要 0.19s。
前向/反向的矩阵规模相当，差 10 倍必有实现问题。这里逐行找出来。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from features import ART, DISH_CAT_IDX, N_DISH
import train as T


def main():
    samples = pd.read_csv(ART / "samples.csv")
    samples["taboo"] = samples["taboo"].fillna("").astype(str)
    tr = T.attach_gold(samples[samples.split == "train"].reset_index(drop=True))
    P = T._pack(tr)
    model = T.DishScorer(np.random.default_rng(T.SEED + 1))

    # 取一个真实 batch
    b = np.arange(0, 64)
    B_ = len(b)
    ids, dc, dn = T.tile_dishes(np.arange(N_DISH)[None, :], B_)

    # ---- 先测前向（对照）----
    t = time.perf_counter()
    for _ in range(5):
        s = model.forward(P["cat"][b], P["num"][b], ids, dc, dn)
    fwd = (time.perf_counter() - t) / 5
    print(f"前向                        {fwd * 1000:9.2f} ms")

    cat_idx, dish_ids, dish_cat, x, h_pre, h = model.cache
    ds = np.random.default_rng(0).normal(size=s.shape) * 0.01
    d_ctx_dim = model.d_ctx

    def bench(label, fn, n=5):
        for _ in range(2):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn()
        dt = (time.perf_counter() - t0) / n
        print(f"{label:<28s}{dt * 1000:9.2f} ms")
        return dt, out

    print()
    print("逐行拆分（每行各测 5 次取均值）")
    print("-" * 46)
    t_w2, _ = bench("gW2 (einsum bkh,bk->h)", lambda: np.einsum("bkh,bk->h", h, ds)[:, None])
    t_dh, dh_pre = bench("dh_pre = (ds @ W2.T) * relu",
                         lambda: (ds[:, :, None] @ model.W2.T) * (h_pre > 0))
    t_w1, _ = bench("gW1 (einsum bkd,bkh->dh)", lambda: np.einsum("bkd,bkh->dh", x, dh_pre))
    t_dx, dx = bench("dx = dh_pre @ W1.T", lambda: dh_pre @ model.W1.T)

    # ---- 嫌疑最大：embeddings 用 Python 循环 + bincount ----
    def scatter_loop(dx):
        out = []
        off = d_ctx_dim
        for i, e in enumerate(model.dish_emb):
            w = e.shape[1]
            idx = dish_cat[:, :, i].ravel()
            vals = dx[:, :, off:off + w].reshape(idx.size, w)
            ge = np.stack([np.bincount(idx, weights=vals[:, j], minlength=e.shape[0])
                           for j in range(w)], axis=1)
            out.append(ge)
            off += w
        return out

    t_emb, _ = bench("dish_emb scatter（循环+bincount）", lambda: scatter_loop(dx))

    def scatter_fast(dx):
        """矢量化：用 np.add.at 一次写完，避免每列一次 bincount"""
        out = []
        off = d_ctx_dim + model.dish_id_emb.shape[1]
        flat_ids = dish_cat.reshape(-1, dish_cat.shape[2])
        for i, e in enumerate(model.dish_emb):
            w = e.shape[1]
            ge = np.zeros_like(e, dtype=np.float64)
            np.add.at(ge, flat_ids[:, i], dx[:, :, off:off + w].reshape(-1, w))
            out.append(ge.astype(e.dtype))
            off += w
        return out

    t_fast, _ = bench("dish_emb scatter（np.add.at）", lambda: scatter_fast(dx))

    # ---- 对比：完全不传梯度到 embeddings（下界估计）----
    total = t_w2 + t_dh + t_w1 + t_dx + t_emb
    print("-" * 46)
    print(f"{'上面几行合计':<28s}{total * 1000:9.2f} ms")
    print(f"{'实测 backward 单次':<28s}{1820.75:9.2f} ms")
    print(f"{'差值（未解释）':<28s}{(1.82075 - total) * 1000:9.2f} ms")
    print()
    print(f"dish_emb scatter 占 backward 的比例: {100 * t_emb / 1.82075:.1f}%")
    print(f"换成 np.add.at 的加速比: {t_emb / max(t_fast, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
