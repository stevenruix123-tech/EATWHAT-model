# -*- coding: utf-8 -*-
"""定位 DishScorer.backward 里到底哪一步慢（forward 137ms vs backward 1321ms）。"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\123\Desktop\桌面应用\文档\meal-decision")
sys.path.insert(0, str(BASE))
import train as T  # noqa: E402
from features import N_DISH  # noqa: E402

S = pd.read_csv(BASE / "artifacts" / "samples.csv")
S["taboo"] = S["taboo"].fillna("").astype(str)
tr = T.attach_gold(S[S.split == "train"].reset_index(drop=True))
P = T._pack(tr)
groups = {}
for r, sid in enumerate(P["sid"]):
    groups.setdefault(sid, []).append(r)
gl = list(groups.values())
perm = np.random.default_rng(7).permutation(len(gl))
b = np.array([r for gi in perm[:T.BATCH] for r in gl[gi]])
B_ = len(b)

rng = np.random.default_rng(0)
m = T.DishScorer(rng)
all_ids = np.arange(N_DISH)[None, :]
ids, dc, dn = T.tile_dishes(all_ids, B_)
s = m.forward(P["cat"][b], P["num"][b], ids, dc, dn)
ds = rng.normal(0, 1e-3, size=s.shape)

cat_idx, dish_ids, dish_cat, x, h_pre, h = m.cache
print(f"B_={B_}  K={N_DISH}  x{x.shape}  h{h.shape}  dish_emb维数={len(m.dish_emb)}")

t0 = time.time()
g_W2 = np.einsum("bkh,bk->h", h, ds)[:, None]
t_W2 = time.time() - t0

t0 = time.time()
dh_pre = (ds[:, :, None] @ m.W2.T) * (h_pre > 0)
t_dh = time.time() - t0

t0 = time.time()
g_W1 = np.einsum("bkd,bkh->dh", x, dh_pre)
t_W1 = time.time() - t0

t0 = time.time()
g_b1 = dh_pre.sum(axis=(0, 1))
t_b1 = time.time() - t0

t0 = time.time()
dx = dh_pre @ m.W1.T
t_dx = time.time() - t0

t0 = time.time()
d_ctx = dx[:, :, :m.d_ctx].mean(1)
t_dctx = time.time() - t0

t0 = time.time()
m.enc.backward(d_ctx, reduce_k=True)
t_enc = time.time() - t0


def scatter(indices, vals, n_rows, dtype):
    idx = indices.ravel()
    v = vals.reshape(idx.size, vals.shape[-1])
    return np.stack([np.bincount(idx, weights=v[:, j], minlength=n_rows)
                     for j in range(v.shape[1])], axis=1).astype(dtype)


off = m.d_ctx
t0 = time.time()
scatter(dish_ids, dx[:, :, off:off + m.dish_id_emb.shape[1]],
        m.dish_id_emb.shape[0], m.dish_id_emb.dtype)
t_sc_id = time.time() - t0
off += m.dish_id_emb.shape[1]
t_sc = 0.0
for i, e in enumerate(m.dish_emb):
    w = e.shape[1]
    t0 = time.time()
    scatter(dish_cat[:, :, i], dx[:, :, off:off + w], e.shape[0], e.dtype)
    t_sc += time.time() - t0
    off += w

items = [("W2 einsum", t_W2), ("dh_pre  (matmul+relu)", t_dh), ("W1 einsum", t_W1),
         ("b1 sum", t_b1), ("dx matmul", t_dx), ("d_ctx mean", t_dctx),
         ("enc.backward", t_enc), ("scatter dish_id_emb", t_sc_id),
         (f"scatter dish_emb x{len(m.dish_emb)}", t_sc)]
tot = sum(t for _, t in items)
print()
for n, t in items:
    bar = "#" * int(t / max(tot, 1e-9) * 50)
    print(f"  {n:<28} {t*1000:8.1f} ms  {t/tot:6.1%}  {bar}")
print(f"  {'合计':<28} {tot*1000:8.1f} ms")
