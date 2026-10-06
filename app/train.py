# -*- coding: utf-8 -*-
"""
train.py —— 训练两层「今天该吃什么」决策模型（纯 numpy，含手写反向传播）

两层结构：
    第一层  CuisineNet  ：情境 -> 各品类的偏好分布（softmax，品类数从 dishes.csv 自动推导）
    第二层  DishScorer  ：情境 + 菜品 -> 效用分（在候选品类内排序）

训练信号：samples.csv 的 observed_signal
    连续值，表示「那道菜在当时那个情境下有多让人满意」。
    注意它只有被选中的菜能被观测到 —— 这正是决策数据（bandit feedback）的形态。

评估用决策指标而不是分类准确率：
    后悔值(regret) = 真实最优菜的效用 - 模型推荐菜的效用   （越低越好）
    top-1 / top-3 命中率、品类命中率、与随机/热门基线的对比
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    ART, CATEGORIES, CAT_TO_ID, CTX_CAT_COLS, CTX_NUM_COLS, DISH_CAT_COLS,
    DISH_CAT_IDX, DISH_NUM_FEAT, DISH_NUM_COLS, DISH_NAMES, N_CAT, N_DISH,
    emb_dim, encode_context, encode_dishes,
)

SEED = 7
EPOCHS = 60
LR = 3e-3
BATCH = 64
REG = 1e-4
HUBER_DELTA = 0.15
LAM_REG = 0.3


# ===========================================================================
# 基础层：手写 MLP 组件
# ===========================================================================
def he_init(rng, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def softmax_ce(logits, labels=None):
    """交叉熵 + 梯度。labels=None 时取自身 argmax 为目标（用于列表内排序）。"""
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=-1, keepdims=True)
    if labels is None:
        pick = logits.argmax(-1, keepdims=True)
        loss = float(-np.log(np.take_along_axis(p, pick, -1) + 1e-12).mean())
        return loss, p
    B = logits.shape[0]
    loss = float(-np.log(p[np.arange(B), labels] + 1e-12).mean())
    d = p.copy()
    d[np.arange(B), labels] -= 1.0
    return loss, d / B


def huber(pred, target, delta):
    """标准 Huber 损失：残差小用平方（梯度线性），残差大用线性（梯度饱和）。
    注意必须写成「二次段 + 线性段」两段式；只写 clip 的绝对值形式在
    |残差|>δ 时是常数函数，梯度为 0，与其导数 clip(diff) 不自洽。"""
    d = pred - target
    absd = np.abs(d)
    return np.where(absd <= delta, 0.5 * d * d, delta * (absd - 0.5 * delta))


def d_huber(pred, target, delta):
    return np.clip(pred - target, -delta, delta)


class Adam:
    def __init__(self, params: dict, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ===========================================================================
# 情境编码器：类别 embedding + 数值特征 -> 低维情境向量
# ===========================================================================
class ContextEncoder:
    def __init__(self, rng, d_out=24, hidden=96):
        self.emb = [rng.normal(0, 0.1, size=(len(_vocab(c)), emb_dim(c, len(_vocab(c)))))
                    for c in CTX_CAT_COLS]
        d_in = sum(e.shape[1] for e in self.emb) + len(CTX_NUM_COLS)
        self.W1 = he_init(rng, d_in, hidden)
        self.b1 = np.zeros(hidden)
        self.W2 = he_init(rng, hidden, d_out)
        self.b2 = np.zeros(d_out)
        self.d_in = d_in

    def names(self):
        return (["W1", "b1", "W2", "b2"] + [f"emb{i}" for i in range(len(self.emb))])

    def forward(self, cat_idx, num):
        """cat_idx: [B, C] 类别索引；num: [B, n_num] 或 [B, K, n_num] 数值特征。
        当 num 是 3D 时，类别 embedding 会广播到 K 维（情境对 K 个候选菜品是同一个）。"""
        e = np.concatenate([self.emb[i][cat_idx[..., i]] for i in range(len(self.emb))],
                           axis=-1)                                # [B, E]
        self.broadcast = (num.ndim == 3)
        if self.broadcast:
            e = np.broadcast_to(e[:, None, :], (e.shape[0], num.shape[1], e.shape[1]))
        x = np.concatenate([e, num], axis=-1)                       # [..., E + n_num]
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        z = h @ self.W2 + self.b2
        self.cache = (cat_idx, x, h_pre, h)
        return z

    def backward(self, dz, reduce_k=False):
        """reduce_k=True：上游 dz 是 [B, d]（已按 K 求和），但前向时隐藏层其实是
        在 K 个候选菜品上各自独立前向的。因此参数梯度要对 K 求和（不能只算一次），
        而「按索引取用的 embedding」梯度本来就是靠 scatter 累加，无需额外处理。"""
        cat_idx, x, h_pre, h = self.cache
        if reduce_k:
            # 各 K 位置共享同一个上游梯度 -> 广播到 [B, K, d]
            dz = np.broadcast_to(dz[:, None, :], (x.shape[0], x.shape[1], dz.shape[-1]))
        lead = tuple(range(dz.ndim - 1))
        g = {"W2": np.einsum("...h,...o->...ho", h, dz).sum(axis=lead),
             "b2": dz.sum(axis=lead)}
        dh_pre = (dz @ self.W2.T) * (h_pre > 0)
        g["W1"] = np.einsum("...d,...h->...dh", x, dh_pre).sum(axis=lead)
        g["b1"] = dh_pre.sum(axis=lead)
        dx = dh_pre @ self.W1.T
        n_num = len(CTX_NUM_COLS)
        d_num = dx[..., dx.shape[-1] - n_num:]
        d_e = dx[..., :dx.shape[-1] - n_num]
        off = 0
        for i, e in enumerate(self.emb):
            w = e.shape[1]
            # 类别索引是 [B, C]，而 d_e 可能是 [B, K, E]（数值特征被广播到 K 维）
            # -> 先把索引广播到与 d_e 相同的前导维度，再按索引 scatter 累加
            ci = cat_idx[..., i]
            if d_e.ndim == 3:
                ci = np.broadcast_to(ci[:, None], (d_e.shape[0], d_e.shape[1]))
                vals = d_e[..., off:off + w].reshape(-1, w)
            else:
                vals = d_e[..., off:off + w].reshape(-1, w)
            idx = np.asarray(ci).ravel()
            ge = np.stack([np.bincount(idx, weights=vals[:, j], minlength=e.shape[0])
                           for j in range(w)], axis=1).astype(e.dtype)
            g[f"emb{i}"] = ge
            off += w
        self.grads = g
        return d_num


def _vocab(col: str):
    from features import CTX_VOCAB
    return CTX_VOCAB[col]


# ===========================================================================
# 第一层：情境 -> 品类
# ===========================================================================
class CuisineNet:
    def __init__(self, rng, d_ctx=24, hidden=96):
        self.enc = ContextEncoder(rng, d_out=d_ctx, hidden=hidden)
        self.W = he_init(rng, d_ctx, N_CAT)
        self.b = np.zeros(N_CAT)

    def params(self):
        p = {}
        for n in self.enc.names():
            p[f"enc.{n}"] = getattr(self.enc, n) if n.startswith("W") or n.startswith("b") \
                else self.enc.emb[int(n[3:])]
        p["W"], p["b"] = self.W, self.b
        return p

    def forward(self, cat_idx, num):
        self.z = self.enc.forward(cat_idx, num)
        return self.z @ self.W + self.b

    def backward(self, dlogits):
        g = {"W": self.z.T @ dlogits, "b": dlogits.sum(0)}
        self.enc.backward(dlogits @ self.W.T)
        for k, v in self.enc.grads.items():
            g[f"enc.{k}"] = v
        self.grads = g


# ===========================================================================
# 第二层：情境 + 菜品 -> 效用分
# ===========================================================================
class DishScorer:
    def __init__(self, rng, d_ctx=24, hidden=48, id_dim=8):
        self.enc = ContextEncoder(rng, d_out=d_ctx, hidden=96)
        self.d_ctx = d_ctx
        self.dish_id_emb = rng.normal(0, 0.1, size=(N_DISH, id_dim))
        self.dish_emb = [rng.normal(0, 0.1, size=(len(_dvocab(c)), emb_dim(c, len(_dvocab(c)))))
                         for c in DISH_CAT_COLS]
        self.d_dish = id_dim + sum(e.shape[1] for e in self.dish_emb) + len(DISH_NUM_COLS)
        self.W1 = he_init(rng, d_ctx + self.d_dish, hidden)
        self.b1 = np.zeros(hidden)
        self.W2 = he_init(rng, hidden, 1)
        self.b2 = np.zeros(1)
        self.hidden = hidden

    def params(self):
        p = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
             "dish_id_emb": self.dish_id_emb}
        for i, e in enumerate(self.dish_emb):
            p[f"dish_emb{i}"] = e
        for n in self.enc.names():
            p[f"enc.{n}"] = getattr(self.enc, n) if n.startswith("W") or n.startswith("b") \
                else self.enc.emb[int(n[3:])]
        return p

    def forward(self, cat_idx, num, dish_ids, dish_cat, dish_num):
        B, K = dish_ids.shape
        # 情境编码**只算一次**：同一行的 K 个候选菜品共享完全相同的情境。
        # 旧实现把数值特征铺成 [B,K,n_num] 传给 encoder，让 encoder 在 K 个位置上
        # 各算一遍（encoder 内部再把类别 embedding 广播到 K）—— 结果一样，但
        # 前向要物化 [B,K,63] 的张量，反向要在 [B,K,·] 上做 einsum，
        # 实测 backward 因此占掉 93.6% 的训练时间（1265ms / batch）。
        z2 = self.enc.forward(cat_idx, num)                     # [B, d_ctx]
        z = np.broadcast_to(z2[:, None, :], (B, K, z2.shape[1]))
        e_id = self.dish_id_emb[dish_ids]                       # [B, K, id_dim]
        e_cat = np.concatenate(
            [self.dish_emb[i][dish_cat[:, :, i]] for i in range(len(self.dish_emb))], axis=2)
        x = np.concatenate([z, e_id, e_cat, dish_num], axis=2)
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        s = (h @ self.W2 + self.b2).squeeze(-1)
        self.cache = (cat_idx, dish_ids, dish_cat, x, h_pre, h)
        return s

    def backward(self, ds):
        cat_idx, dish_ids, dish_cat, x, h_pre, h = self.cache
        g = {"W2": np.einsum("bkh,bk->h", h, ds)[:, None],
             "b2": np.array([ds.sum()])}
        dh_pre = (ds[:, :, None] @ self.W2.T) * (h_pre > 0)
        g["W1"] = np.einsum("bkd,bkh->dh", x, dh_pre)
        g["b1"] = dh_pre.sum(axis=(0, 1))
        dx = dh_pre @ self.W1.T
        # 情境部分：把 K 个候选菜品上的贡献**求和**，得到该情境用了 K 次的总梯度。
        # 注意不能除以 K：前向是「一次编码、K 次使用」，所以 embedding 梯度必须累计
        # K 次贡献（W1/W2 在上面也是按 (0,1) 对 K 求和）。旧实现因为前向把 encoder
        # 在 K 个位置上各跑了一遍，才需要用 mean 抵消那次多余广播，
        # 而它漏掉了 embedding 的 1/K 修正，等于把情境 embedding 的梯度缩小了 K 倍
        # （317 倍）。现在前向只跑一次，用 sum 才是正确值，且 gradient check 会验证。
        d_ctx = dx[:, :, :self.d_ctx].sum(1)
        self.enc.backward(d_ctx, reduce_k=False)
        off = self.d_ctx

        def scatter(indices, vals, n_rows, dtype):
            """按索引累加梯度（bincount 版本，维度语义清晰）"""
            idx = indices.ravel()
            v = vals.reshape(idx.size, vals.shape[-1])
            return np.stack([np.bincount(idx, weights=v[:, j], minlength=n_rows)
                             for j in range(v.shape[1])], axis=1).astype(dtype)

        g["dish_id_emb"] = scatter(dish_ids, dx[:, :, off:off + self.dish_id_emb.shape[1]],
                                   self.dish_id_emb.shape[0], self.dish_id_emb.dtype)
        off += self.dish_id_emb.shape[1]
        for i, e in enumerate(self.dish_emb):
            w = e.shape[1]
            g[f"dish_emb{i}"] = scatter(dish_cat[:, :, i], dx[:, :, off:off + w],
                                        e.shape[0], e.dtype)
            off += w
        for k, v in self.enc.grads.items():
            g[f"enc.{k}"] = v
        self.grads = g


def _dvocab(col: str):
    from features import DISH_VOCAB
    return DISH_VOCAB[col]


# ===========================================================================
# 工具：把情境平铺到所有候选菜品
# ===========================================================================
def tile_context(cat, num, K):
    B = cat.shape[0]
    return (np.repeat(cat[:, None, :], K, axis=1),
            np.repeat(num[:, None, :], K, axis=1))


def tile_dishes(dish_ids, B):
    """把候选菜品特征平铺到 batch 的每一行 -> [B, K, ...]"""
    ids = np.repeat(np.asarray(dish_ids).reshape(1, -1), B, axis=0)   # [B, K]
    dc = DISH_CAT_IDX[ids]          # [B, K, n_cat]
    dn = DISH_NUM_FEAT[ids]         # [B, K, n_num]
    return ids, dc, dn


# ===========================================================================
# 训练：第一层
# ===========================================================================
def train_cuisine(scenarios, verbose=True):
    tr = scenarios[scenarios.split == "train"].reset_index(drop=True)
    va = scenarios[scenarios.split == "val"].reset_index(drop=True)
    y = tr.gold_category.map(CAT_TO_ID).values
    yv = va.gold_category.map(CAT_TO_ID).values
    Xc, Xn = encode_context(tr)
    Vc, Vn = encode_context(va)

    rng = np.random.default_rng(SEED)
    model = CuisineNet(rng)
    params = model.params()
    opt = Adam(params, LR)
    cnt = np.bincount(y, minlength=N_CAT).astype(float)
    cw = (cnt.sum() / (N_CAT * np.maximum(cnt, 1)))
    cw = cw / cw.mean()

    n = len(y)
    best = (1e9, None, -1)
    hist = []
    for ep in range(EPOCHS):
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = idx[i:i + BATCH]
            logits = model.forward(Xc[b], Xn[b])
            loss, dl = softmax_ce(logits, y[b])
            dl = dl * cw[y[b]][:, None]
            tot += loss * len(b)
            model.backward(dl)
            g = dict(model.grads)
            for k in list(g):
                if k.endswith(("W", "W1", "W2")):
                    g[k] = g[k] + REG * params[k]
            opt.step(params, g)
        acc = float((model.forward(Vc, Vn).argmax(1) == yv).mean())
        hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": acc})
        if 1 - acc < best[0]:
            best = (1 - acc, {k: v.copy() for k, v in params.items()}, ep)
        if verbose and (ep % 10 == 0 or ep == EPOCHS - 1):
            print(f"    epoch {ep:3d}  loss={tot / n:.4f}  val_acc={acc:.4f}")
    for k in params:
        params[k][...] = best[1][k]
    if verbose:
        print(f"    -> 取验证最好的第 {best[2]} 轮，val_acc={1 - best[0]:.4f}")
    return model, hist


# ===========================================================================
# 训练：第二层
# ===========================================================================
def train_dish(samples, verbose=True):
    tr = attach_gold(samples[samples.split == "train"].reset_index(drop=True))
    va = attach_gold(samples[samples.split == "val"].reset_index(drop=True))
    P = _pack(tr)
    V = _pack(va)
    n_chosen = int(P["chosen"].sum())
    if verbose:
        print(f"    训练行数 {len(tr)}（其中被选中(chosen) {n_chosen} 行，"
              f"考虑集评分(considered) {len(tr) - n_chosen} 行）")

    rng = np.random.default_rng(SEED + 1)
    model = DishScorer(rng)
    params = model.params()
    opt = Adam(params, LR)
    all_ids = np.arange(N_DISH)[None, :]

    # 按「情境」分组取样：把每个情境下的 chosen + considered 行一起喂进去。
    # 这样批次内既有排序目标（chosen），又有密集的数值监督（considered）。
    groups: dict = {}
    for r, sid in enumerate(P["sid"]):
        groups.setdefault(sid, []).append(r)
    group_list = list(groups.values())
    n_groups = len(group_list)
    if verbose:
        print(f"    训练情境数 {n_groups}，平均每情境 "
              f"{len(tr) / max(n_groups, 1):.1f} 行观测")

    best = (1e9, None, -1)
    hist = []
    for ep in range(EPOCHS):
        perm = rng.permutation(n_groups)
        tr_rank = tr_reg = 0.0
        n_rank_rows = 0
        for i in range(0, n_groups, BATCH):
            batch = [r for gi in perm[i:i + BATCH] for r in group_list[gi]]
            b = np.array(batch)
            B_ = len(b)
            ck, nk = P["cat"][b], P["num"][b]
            ids, dc, dn = tile_dishes(all_ids, B_)
            s = model.forward(ck, nk, ids, dc, dn)

            # (1) 排序损失：只对 chosen 行（被选中的那个才是「结果」）。
            #     目标是候选品类内每道菜的真实效用（分级相关性），用软目标交叉熵，
            #     而不是对它取 argmax —— 取 argmax 会退化成「每品类只学一道菜」。
            is_chosen = P["chosen"][b]
            if is_chosen.any():
                group = (DISH_CAT_IDX[:, 0][None, :] == P["cat_id"][b][:, None])
                s_m = np.where(group, s, -1e9)
                gt = group_targets(P, b)                      # [B_, N_DISH]，非候选为 -1e9
                gt_soft = softmax_rows(gt)                     # 归一成候选内分布
                loss_rank, dp = softmax_ce_soft(s_m[is_chosen], gt_soft[is_chosen])
                ds = np.zeros_like(s)
                ds[is_chosen] = dp
                tr_rank += loss_rank * int(is_chosen.sum())
                n_rank_rows += int(is_chosen.sum())
            else:
                loss_rank, ds = 0.0, np.zeros_like(s)

            # (2) 回归损失：对全部观测行（chosen + considered）学准效用数值。
            #     梯度按观测行数归一 —— 这是发散与否的关键。
            obs = np.zeros_like(s)
            obs[np.arange(B_), P["dish"][b]] = 1.0
            pred_obs = (s * obs).sum(1)
            loss_reg = float(huber(pred_obs, P["y"][b], HUBER_DELTA).mean())
            d_reg = d_huber(pred_obs, P["y"][b], HUBER_DELTA)
            ds[np.arange(B_), P["dish"][b]] += LAM_REG * d_reg / B_
            tr_reg += loss_reg * B_

            model.backward(ds)
            g = dict(model.grads)
            for k in list(g):
                if k.endswith(("W1", "W2")) and not k.startswith("enc."):
                    g[k] = g[k] + REG * params[k]
            opt.step(params, g)

        vl = _eval_rank(model, V)
        va_top1 = _eval_top1(model, V)
        hist.append({"epoch": ep, "train_rank": tr_rank / max(n_rank_rows, 1),
                     "train_reg": tr_reg / max(len(tr), 1), "val_rank": vl,
                     "val_top1": va_top1})
        # 用「验证排序损失」选轮次，不用命中率。
        # 实测教训：精确命中率从 epoch 0 到 epoch 59 都卡在 13.4% 不动
        # （被 12% 的随手选噪声主导，天花板约 1/6），完全没有区分度；
        # 而排序损失能稳定反映学习进度，且与最终 top-3 / 后悔值同向。
        if vl < best[0]:
            best = (vl, {k: v.copy() for k, v in params.items()}, ep)
        if verbose and (ep % 10 == 0 or ep == EPOCHS - 1):
            print(f"    epoch {ep:3d}  rank={tr_rank / max(n_rank_rows, 1):.4f}"
                  f"  reg={tr_reg / max(len(tr), 1):.4f}"
                  f"  val_rank={vl:.4f}  val_top1(仅参考)={va_top1:.2%}")
    for k in params:
        params[k][...] = best[1][k]
    if verbose:
        print(f"    -> 取验证排序损失最低的第 {best[2]} 轮，val_rank={best[0]:.4f}")
    return model, hist


def _pack(df: pd.DataFrame):
    c, n = encode_context(df)
    return dict(cat=c, num=n, dish=df.dish_id.values,
                y=df.observed_signal.values.astype(float),
                cat_id=df.category.map(CAT_TO_ID).values,
                sid=df.scenario_id.values,
                gold=df["gold_dish_id"].values if "gold_dish_id" in df
                else np.full(len(df), -1),
                chosen=(df.get("source", pd.Series(["chosen"] * len(df)))
                        .values == "chosen"))


def attach_gold(df: pd.DataFrame) -> pd.DataFrame:
    """把每个情境的「理论最优菜品」挂到样本上（只用于模型选择时的命中率统计）。"""
    gold = pd.read_csv(ART / "scenarios.csv")[["scenario_id", "gold_dish_id"]]
    return df.merge(gold, on="scenario_id", how="left")


# 真实效用矩阵：行序 = scenarios.csv 行序（scenario_id 0..N-1），列 = dish_id
_UTIL_MATRIX = None


def util_matrix() -> np.ndarray:
    """[N_SCENARIO, N_DISH] 每个情境下所有菜的真实效用（惰性加载一次）。

    这是第二层排序目标的**唯一正确来源**：第 2 层要学「同品类内哪道菜更好」，
    就必须拿到品类内每道菜各自的效用，而不是某一行观测到的那一个值。
    """
    global _UTIL_MATRIX
    if _UTIL_MATRIX is None:
        p = ART / "scenario_utility.npy"
        if not p.exists():
            raise SystemExit(
                f"缺少 {p}。第二层排序需要每个情境的真实效用矩阵，"
                f"请先重跑 build_dataset.py 生成它。\n"
                f"（旧版本没落盘这个矩阵，训练时只能用「单行观测值」当整组目标，"
                f"排序目标会退化：品类内所有菜拿到同一个目标值。）"
            )
        _UTIL_MATRIX = np.load(p)
        ids = np.load(ART / "scenario_ids.npy") if (ART / "scenario_ids.npy").exists() else None
        if ids is not None and len(ids) != len(_UTIL_MATRIX):
            raise SystemExit("scenario_utility.npy 与 scenario_ids.npy 行数不一致")
        if _UTIL_MATRIX.shape[1] != N_DISH:
            raise SystemExit(
                f"scenario_utility.npy 列数 {_UTIL_MATRIX.shape[1]} 与菜品数 {N_DISH} 不符，"
                f"说明菜品库或数据集被单独改过，请重跑 build_dataset.py。"
            )
    return _UTIL_MATRIX


def group_targets(P, idx):
    """排序目标 = 该情境候选品类内、**每道菜各自的真实效用**。

    关键：目标必须逐菜不同。旧实现写成
        tgt[r, gm] = P["y"][i]        # gm = 同品类全部菜, y = 该行观测到的单个效用
    等于把「一行观测到的效用」广播给该品类下每一道菜，于是
    tgt.argmax(1) 永远返回品类内 dish_id 最小的那道菜 —— 模型学到的是
    「每个品类固定答同一道菜」，训练损失能压到 0.003，而品类内 top-1 只有
    4.83%（≈ 随机 1/17.6）。这个缺陷在 47 道菜 / 每类 6 道时被掩盖，
    品类内候选池涨到 17.6 道后暴露。
    """
    B_ = len(idx)
    U = util_matrix()
    # 掩码用「该情境的 gold 品类」：模型实际是在第一层选出的品类内排序，
    # 用行自身品类（= 被选中那道菜的品类）当掩码会把「品类选错」也编码进目标。
    gold_cat = None
    for k in ("gold_category",):
        if k in P:
            gold_cat = P[k]
    tgt = np.full((B_, N_DISH), -1e9)
    for r, i in enumerate(idx):
        if gold_cat is not None:
            cid = CAT_TO_ID.get(str(gold_cat[i]), int(P["cat_id"][i]))
        else:
            cid = int(P["cat_id"][i])
        gm = (DISH_CAT_IDX[:, 0] == cid)
        tgt[r, gm] = U[P["sid"][i]][gm]
    return tgt


def softmax_rows(x, temp=0.02):
    """按行归一成概率分布，用于把效用向量变成软目标。

    温度 temp 决定「排序目标有多尖」：效用尺度约 0.5~0.9、品类内最优与次优
    差距中位仅 0.0075，若用 temp=1 目标几乎是均匀分布，学不到排序；
    取 0.02 让最优菜拿到明显更大的权重，同时保留次优菜的名次信息。
    """
    z = np.asarray(x, dtype=float)
    z = np.where(z > -1e8, z / max(temp, 1e-9), -1e9)
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    e = np.where(np.isfinite(e), e, 0.0)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-300)


def _eval_rank(model, P, chunk=256):
    """验证集的排序损失（只在 chosen 样本上算，与训练目标一致）"""
    keep = np.where(P["chosen"])[0]
    n = len(keep)
    tot = 0.0
    for i in range(0, n, chunk):
        b = keep[i:i + chunk]
        B_ = len(b)
        ids, dc, dn = tile_dishes(np.arange(N_DISH)[None, :], B_)
        s = model.forward(P["cat"][b], P["num"][b], ids, dc, dn)
        group = (DISH_CAT_IDX[:, 0][None, :] == P["cat_id"][b][:, None])
        s_m = np.where(group, s, -1e9)
        gt = group_targets(P, b)
        loss, _ = softmax_ce_soft(s_m, softmax_rows(gt))
        tot += loss * B_
    return tot / max(n, 1)


def _eval_top1(model, P, chunk=256):
    """验证集命中率：在候选品类内，argmax 是否命中该品类**理论上最优**的菜品。

    注意不能用「已观测菜品中的最优」当分母 —— 考虑集里本来就含同品类前三名，
    那样算出来会虚高到 100%，失去挑选轮次的作用。这里对照的是 gold（真实最优）。"""
    keep = np.where(P["chosen"])[0]
    n = len(keep)
    hit = 0
    for i in range(0, n, chunk):
        b = keep[i:i + chunk]
        B_ = len(b)
        ids, dc, dn = tile_dishes(np.arange(N_DISH)[None, :], B_)
        s = model.forward(P["cat"][b], P["num"][b], ids, dc, dn)
        group = (DISH_CAT_IDX[:, 0][None, :] == P["cat_id"][b][:, None])
        s_m = np.where(group, s, -1e9)
        hit += int((s_m.argmax(1) == P["gold"][b]).sum())
    return hit / max(n, 1)


# ===========================================================================
# 推理
# ===========================================================================
def _eval_topk(model, P, k=3, chunk=256):
    """同 _eval_top1，但统计 gold 是否落在候选品类内的前 k 名。

    只用 chosen 行、对照 gold（真实最优），口径与 _eval_top1 一致。
    候选池从 6 道扩到平均 17.6 道后，top-1 天然更难，top-3 更能反映
    「用户会从推荐里挑」的实际体验。
    """
    keep = np.where(P["chosen"])[0]
    n = len(keep)
    hit = 0
    for i in range(0, n, chunk):
        b = keep[i:i + chunk]
        B_ = len(b)
        ids, dc, dn = tile_dishes(np.arange(N_DISH)[None, :], B_)
        s = model.forward(P["cat"][b], P["num"][b], ids, dc, dn)
        group = (DISH_CAT_IDX[:, 0][None, :] == P["cat_id"][b][:, None])
        s_m = np.where(group, s, -1e9)
        order = np.argsort(-s_m, axis=1)[:, :k]
        hit += int((order == P["gold"][b][:, None]).any(axis=1).sum())
    return hit / max(n, 1)


def softmax_ce_soft(logits, target):
    """软目标交叉熵 + 对 logits 的梯度（逆序，已按 batch 归一）。

    用于第二层排序：目标是候选品类内**每道菜各自的真实效用**，
    是有多有少的分级相关性，不是「唯一正确那一道」。

    为什么不能用 softmax_ce(..., gt.argmax(1))：把效用向量取 argmax 又会退化成
    「每个品类只学一道菜」，等于没修 group_targets。实测那样 top-1 只有 4.8%。
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits{logits.shape} 与 target{target.shape} 形状不一致")
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=-1, keepdims=True)
    B_ = logits.shape[0]
    loss = float(-np.sum(target * np.log(p + 1e-12)) / max(B_, 1))
    return loss, (p - target) / max(B_, 1)


def predict_row(row, cuisine_model, dish_model, topk=3, use_filter=True):
    ctx = pd.DataFrame([row])
    c, n = encode_context(ctx)
    if use_filter:
        w = int(cuisine_model.forward(c, n)[0].argmax())
        cand = np.where(DISH_CAT_IDX[:, 0] == w)[0]
    else:
        w, cand = -1, np.arange(N_DISH)
    K = len(cand)
    ck, nk = c, n
    ids, dc, dn = tile_dishes(cand[None, :], 1)
    s = dish_model.forward(ck, nk, ids, dc, dn)[0]
    order = np.argsort(-s)
    return cand[order[:topk]], s[order[:topk]], w


# ===========================================================================
# 评估
# ===========================================================================
def evaluate(scenarios, cuisine_model, dish_model, split="test", verbose=True):
    ev = scenarios[scenarios.split == split].reset_index(drop=True)
    name2id = {n: i for i, n in enumerate(DISH_NAMES)}
    gold_ids = ev.gold_dish.map(name2id).values
    gold_cat = ev.gold_category.map(CAT_TO_ID).values
    gold_u = ev.gold_utility.values

    res = {}
    rows_out = []
    for tag, use_filter in (("两层模型", True), ("单层模型(不过滤品类)", False)):
        h1 = h3 = cat_ok = 0
        rec_u, rec_rand, rec_pop = [], [], []
        for i, r in ev.iterrows():
            top, sc, w = predict_row(r, cuisine_model, dish_model, topk=3, use_filter=use_filter)
            u = _utility_of(r, top[0])
            rec_u.append(u)
            rec_rand.append(_utility_of(r, int(np.random.default_rng(i).integers(N_DISH))))
            rec_pop.append(_utility_of(r, POPULAR_DISH))
            h1 += int(top[0] == gold_ids[i])
            h3 += int(gold_ids[i] in top)
            # 品类命中：用「两个模型都推荐同一个品类」来算，单层时取 top1 的品类
            pred_cat = int(DISH_CAT_IDX[top[0], 0])
            cat_ok += int(pred_cat == gold_cat[i])
            if use_filter:
                rows_out.append({
                    "scenario_id": int(r.scenario_id), "time_slot": r.time_slot,
                    "hunger": r.hunger, "available_time": r.available_time,
                    "mood": r.mood, "weather": r.weather, "companion": r.companion,
                    "health_goal": r.health_goal, "spicy_tol": r.spicy_tol,
                    "taboo": r.taboo,
                    "gold_dish": r.gold_dish, "gold_category": r.gold_category,
                    "model_top1": DISH_NAMES[top[0]],
                    "model_top1_category": CATEGORIES[pred_cat],
                    "model_top1_score": round(float(sc[0]), 4),
                    "model_top2": DISH_NAMES[top[1]], "model_top3": DISH_NAMES[top[2]],
                    "hit_top1": int(top[0] == gold_ids[i]),
                    "hit_top3": int(gold_ids[i] in top),
                    "utility_model": round(u, 4),
                    "utility_gold": round(float(gold_u[i]), 4),
                    "regret": round(float(gold_u[i] - u), 4),
                    "shift": r.get("shift", "normal"),
                })
        N = len(ev)
        res[tag] = {
            "top1": h1 / N, "top3": h3 / N, "category_acc": cat_ok / N,
            "mean_utility": float(np.mean(rec_u)),
            "mean_utility_random": float(np.mean(rec_rand)),
            "mean_utility_popularity": float(np.mean(rec_pop)),
            "mean_regret": float(np.mean(gold_u - np.array(rec_u))),
            "n": N,
        }
    return res, pd.DataFrame(rows_out)


POPULAR_DISH = int(np.bincount(
    pd.read_csv(ART / "samples.csv").query("split == 'train'").dish_id.values,
    minlength=N_DISH).argmax())


_UTIL_CACHE: dict = {}
_DISH_ARR = None


def _dish_arr():
    """惰性构造一次菜品属性数组，供 utility_vector 复用。"""
    global _DISH_ARR
    if _DISH_ARR is None:
        import build_dataset as B
        _DISH_ARR = B._dish_arrays(B.dishes)
    return _DISH_ARR


def utility_vector(row) -> np.ndarray:
    """给定情境，算出所有菜品的真实效用（评估用的「上帝视角」）。

    [N_DISH] 维。用 build_dataset.utility_all 向量化计算：
    菜品库 317 道时，若沿用逐行标量实现，每个情境要跑 317 次 python 级调用，
    评估阶段要算几千个情境，会白等几分钟。
    """
    # 缓存键必须包含情境里**所有影响效用的列**。
    # 原实现漏了 taboo：taboo 在效用函数第 (10)/(11) 条是「一票否决」，
    # 两个只差 taboo 的情境（""=无忌口 与 "素食" / "海鲜,乳制品"）会算出完全不同的
    # 效用向量，却共用同一个缓存键 —— 后一个情境会拿到前一个的错误效用向量，
    # 评估指标随之全错（实测 1162 行测试集里 70% 命中过缓存）。
    key = (tuple(str(row[c]) for c in CTX_CAT_COLS)
           + (str(row["taboo"]), int(row.hunger), int(row.available_time)))
    if key in _UTIL_CACHE:
        return _UTIL_CACHE[key]
    # 快路径：数据集自带真实效用矩阵时直接按 scenario_id 取。
    # 这样评估的「上帝视角」与训练目标**同源**，不会出现「训练用一套、
    # 评估用另一套」的口径漂移（原实现在 317 道菜上每次还要重算一遍）。
    if "scenario_id" in row and (ART / "scenario_utility.npy").exists():
        U = util_matrix()
        sid = int(row["scenario_id"])
        if 0 <= sid < len(U):
            _UTIL_CACHE[key] = U[sid]
            return U[sid]
    import build_dataset as B
    sc = {c: row[c] for c in CTX_CAT_COLS}
    sc["hunger"] = int(row.hunger)
    sc["available_time"] = int(row.available_time)
    v = B._final_scale(B.utility_all(_dish_arr(), sc))
    _UTIL_CACHE[key] = v
    return v


def _utility_of(row, dish_id: int) -> float:
    return float(utility_vector(row)[dish_id])


# ===========================================================================
# 梯度检验：确认手写反向传播没写错（改模型后必跑）
# ===========================================================================
def grad_check(verbose=True):
    samples = pd.read_csv(ART / "samples.csv")
    scenarios = pd.read_csv(ART / "scenarios.csv")
    for df in (samples, scenarios):
        df["taboo"] = df["taboo"].fillna("").astype(str)
    trs = scenarios[scenarios.split == "train"].head(32).reset_index(drop=True)
    trd = samples[samples.split == "train"].head(12).reset_index(drop=True)

    def sc_loss(m, cat, num, y):
        logits = m.forward(cat.astype(np.int64), num.astype(np.float64))
        return softmax_ce(logits, y.astype(np.int64))[0]

    def dl_loss(m, cat, num, dish, y, cat_id):
        B_ = len(dish)
        ids, dc, dn = tile_dishes(np.arange(N_DISH)[None, :], B_)
        s = m.forward(cat.astype(np.int64), num.astype(np.float64), ids, dc, dn)
        group = (DISH_CAT_IDX[:, 0][None, :] == cat_id[:, None])
        sm = np.where(group, s, -1e9)
        tgt = np.where(group, y[:, None], -1e9).argmax(1)
        lr, _ = softmax_ce(sm, tgt)
        obs = np.zeros_like(s)
        obs[np.arange(B_), dish] = 1.0
        pred = (s * obs).sum(1)
        lreg = float(huber(pred, y, HUBER_DELTA).mean())
        return lr + LAM_REG * lreg

    rng = np.random.default_rng(0)
    print("梯度检验（方向导数法：沿随机方向比较解析/数值导数，对 ReLU 折点稳健）")
    report = []

    def check(name, p, g_an, loss_fn, r):
        """方向导数检验：把整个参数张量当向量，沿随机方向 v 求方向导数。
        比逐坐标扰动稳健得多 —— 单点扰动极易落在 ReLU 折点上，数值导数不可信。"""
        v = r.normal(size=p.shape)
        v /= np.linalg.norm(v)
        an = float((g_an * v).sum())
        eps = 1e-6
        p += eps * v
        lp = loss_fn()
        p -= 2 * eps * v
        lm = loss_fn()
        p += eps * v
        num = (lp - lm) / (2 * eps)
        rel = abs(num - an) / max(abs(num), abs(an), 1e-10)
        report.append((name, an, num, rel))

    # ---- 第一层 ----
    cm = CuisineNet(np.random.default_rng(0))
    for p in cm.params().values():          # 数值梯度需要双精度
        p[...] = p.astype(np.float64)
    c, n = encode_context(trs)
    y = trs.gold_category.map(CAT_TO_ID).values

    def sc_loss():
        return softmax_ce(cm.forward(c.astype(np.int64), n.astype(np.float64)), y)[0]

    _, dl = softmax_ce(cm.forward(c.astype(np.int64), n.astype(np.float64)), y)
    cm.backward(dl)
    for k in ["W", "enc.W1", "enc.W2", "enc.emb0", "enc.emb3"]:
        check("CuisineNet " + k, cm.params()[k], cm.grads[k], sc_loss,
              np.random.default_rng(hash(k) % 10_000))

    # ---- 第二层 ----
    dm = DishScorer(np.random.default_rng(2))
    for p in dm.params().values():
        p[...] = p.astype(np.float64)
    # 只取 chosen 样本，与训练时的排序目标一致
    trd_obs = trd[trd["source"] == "chosen"].reset_index(drop=True) if "source" in trd \
        else trd
    Pchk = _pack(trd_obs)
    c2, n2 = Pchk["cat"], Pchk["num"]
    dish_ids = Pchk["dish"]
    yy = Pchk["y"]
    cid = Pchk["cat_id"]
    B_ = len(dish_ids)
    ids, dc, dn = tile_dishes(np.arange(N_DISH)[None, :], B_)
    gt = group_targets(Pchk, np.arange(B_))

    def dl_loss():
        s = dm.forward(c2, n2, ids, dc, dn)
        group = (DISH_CAT_IDX[:, 0][None, :] == cid[:, None])
        sm = np.where(group, s, -1e9)
        lr, _ = softmax_ce(sm, gt.argmax(1))
        obs = np.zeros_like(s)
        obs[np.arange(B_), dish_ids] = 1.0
        pred = (s * obs).sum(1)
        return lr + LAM_REG * float(huber(pred, yy, HUBER_DELTA).mean())

    s = dm.forward(c2, n2, ids, dc, dn)
    group = (DISH_CAT_IDX[:, 0][None, :] == cid[:, None])
    sm = np.where(group, s, -1e9)
    lr, dp = softmax_ce(sm, gt.argmax(1))
    obs = np.zeros_like(s)
    obs[np.arange(B_), dish_ids] = 1.0
    pred = (s * obs).sum(1)
    ds = dp.copy()
    ds[np.arange(B_), dish_ids] += LAM_REG * d_huber(pred, yy, HUBER_DELTA) / B_
    dm.backward(ds)
    for k in ["W1", "W2", "b1", "dish_id_emb", "dish_emb0", "enc.W1", "enc.W2", "enc.emb2"]:
        check("DishScorer " + k, dm.params()[k], dm.grads[k], dl_loss,
              np.random.default_rng(100 + hash(k) % 10_000))

    ok = True
    for name, ga, gn, rel in report:
        flag = "OK " if rel < 1e-5 else "!! "
        if rel >= 1e-5:
            ok = False
        if verbose:
            print(f"  {flag}{name:28s} 解析={ga:+.6f}  数值={gn:+.6f}  相对误差={rel:.2e}")
    print(f"  梯度检验结论：{'通过' if ok else '不通过（反向传播有误）'}")
    return ok


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("训练「今天该吃什么」两层决策模型")
    print("=" * 72)
    grad_check()

    print("\n" + "=" * 72)
    samples = pd.read_csv(ART / "samples.csv")
    scenarios = pd.read_csv(ART / "scenarios.csv")
    for df in (samples, scenarios):
        df["taboo"] = df["taboo"].fillna("").astype(str)
    print(f"数据规模：{len(samples)} 行 / {len(scenarios)} 个情境 / "
          f"{N_DISH} 道菜 / {N_CAT} 个品类")

    print("\n【第一层】情境 -> 品类")
    cuisine_model, hist1 = train_cuisine(scenarios)

    print("\n【第二层】情境 + 菜品 -> 效用分")
    dish_model, hist2 = train_dish(samples)

    print("\n【评估】测试集（含雪天等训练中未出现的情境）")
    res, pred_df = evaluate(scenarios, cuisine_model, dish_model, split="test")
    for tag, m in res.items():
        print(f"  {tag}")
        print(f"    top-1 命中率   : {m['top1']:.2%}")
        print(f"    top-3 命中率   : {m['top3']:.2%}")
        print(f"    品类命中率     : {m['category_acc']:.2%}")
        print(f"    平均效用       : {m['mean_utility']:.4f}   "
              f"(随机 {m['mean_utility_random']:.4f} / 热门菜 {m['mean_utility_popularity']:.4f})")
        print(f"    平均后悔值     : {m['mean_regret']:.4f}")

    # 分布偏移：雪天 vs 普通
    if pred_df["shift"].nunique() > 1:
        print("\n  分布偏移检查（两层模型）：")
        for s, g in pred_df.groupby("shift"):
            print(f"    {s:6s}  n={len(g):4d}  top1={g.hit_top1.mean():.2%}  "
                  f"top3={g.hit_top3.mean():.2%}  后悔值={g.regret.mean():.4f}")

    # 保存模型与结果
    np.savez(ART / "model.npz",
             **{f"cuisine.{k}": v for k, v in cuisine_model.params().items()},
             **{f"dish.{k}": v for k, v in dish_model.params().items()})
    pred_df.to_csv(ART / "predictions_test.csv", index=False, encoding="utf-8-sig")
    (ART / "metrics.json").write_text(json.dumps(
        {"test": res, "history_cuisine": hist1, "history_dish": hist2},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n模型已保存 -> artifacts/model.npz")
    print(f"预测明细   -> artifacts/predictions_test.csv")
    print(f"总用时 {time.time() - t0:.1f}s")
    return cuisine_model, dish_model, res, pred_df


if __name__ == "__main__":
    main()
