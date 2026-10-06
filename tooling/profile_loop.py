# -*- coding: utf-8 -*-
"""
profile_loop.py —— 给真实训练循环插桩（不修改 train.py）

对 train_dish 的每一步分阶段计时：数据打包 / group_targets / 前向 / 反向 /
优化器 / tile_dishes，跑 3 轮就足够定位瓶颈（3 轮 ≈ 真实 60 轮的 5%）。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from features import ART, DISH_CAT_IDX, N_DISH
import train as T

ACC: dict = {}


def add(k, dt):
    ACC[k] = ACC.get(k, 0.0) + dt


def main():
    samples = pd.read_csv(ART / "samples.csv")
    samples["taboo"] = samples["taboo"].fillna("").astype(str)
    tr = T.attach_gold(samples[samples.split == "train"].reset_index(drop=True))
    P = T._pack(tr)

    rng = np.random.default_rng(T.SEED + 1)
    model = T.DishScorer(rng)
    params = model.params()
    opt = T.Adam(params, T.LR)
    all_ids = np.arange(N_DISH)[None, :]

    t = time.perf_counter()
    groups: dict = {}
    for r, sid in enumerate(P["sid"]):
        groups.setdefault(sid, []).append(r)
    group_list = list(groups.values())
    add("按情境分组", time.perf_counter() - t)
    n_groups = len(group_list)

    N_EPOCHS = 3
    for ep in range(N_EPOCHS):
        perm = rng.permutation(n_groups)
        for i in range(0, n_groups, T.BATCH):
            t = time.perf_counter()
            batch = [r for gi in perm[i:i + T.BATCH] for r in group_list[gi]]
            b = np.array(batch)
            add("① 组装 batch", time.perf_counter() - t)

            B_ = len(b)
            t = time.perf_counter()
            ids, dc, dn = T.tile_dishes(all_ids, B_)
            add("② tile_dishes", time.perf_counter() - t)

            t = time.perf_counter()
            s = model.forward(P["cat"][b], P["num"][b], ids, dc, dn)
            add("③ 前向", time.perf_counter() - t)

            t = time.perf_counter()
            is_chosen = P["chosen"][b]
            group = (DISH_CAT_IDX[:, 0][None, :] == P["cat_id"][b][:, None])
            s_m = np.where(group, s, -1e9)
            gt = T.group_targets(P, b)
            loss_rank, dp = T.softmax_ce(s_m[is_chosen], gt.argmax(1)[is_chosen])
            add("④ 排序损失 + group_targets", time.perf_counter() - t)

            t = time.perf_counter()
            ds = np.zeros_like(s)
            ds[is_chosen] = dp
            obs = np.zeros_like(s)
            obs[np.arange(B_), P["dish"][b]] = 1.0
            pred_obs = (s * obs).sum(1)
            d_reg = T.d_huber(pred_obs, P["y"][b], T.HUBER_DELTA)
            ds[np.arange(B_), P["dish"][b]] += T.LAM_REG * d_reg / B_
            add("⑤ 回归损失", time.perf_counter() - t)

            t = time.perf_counter()
            model.backward(ds)
            add("⑥ 反向传播", time.perf_counter() - t)

            t = time.perf_counter()
            g = dict(model.grads)
            for k in list(g):
                if k.endswith(("W1", "W2")) and not k.startswith("enc."):
                    g[k] = g[k] + T.REG * params[k]
            opt.step(params, g)
            add("⑦ 优化器 step", time.perf_counter() - t)

        t = time.perf_counter()
        T._eval_rank(model, P)
        add("⑧ 验证评估", time.perf_counter() - t)

    total = sum(ACC.values())
    print("=" * 74)
    print(f"真实训练循环插桩（{N_EPOCHS} 轮，每轮 {int(np.ceil(n_groups / T.BATCH))} 步）")
    print("=" * 74)
    for k, v in sorted(ACC.items(), key=lambda x: -x[1]):
        n = N_EPOCHS * int(np.ceil(n_groups / T.BATCH))
        if k == "⑧ 验证评估":
            n = N_EPOCHS
        if k == "按情境分组":
            n = 1
        print(f"  {k:<28s} {v:8.3f} s   {100 * v / total:5.1f}%   "
              f"{v / n * 1000:7.2f} ms/次")
    print("-" * 74)
    print(f"  {'合计':<28s} {total:8.3f} s")
    print(f"  推算 60 轮 ≈ {total / N_EPOCHS * 60:.0f} s")
    print(f"  实测 train.py 总耗时 ≈ 360 s（含梯度检验约 20s、两层训练、评估）")


if __name__ == "__main__":
    main()
