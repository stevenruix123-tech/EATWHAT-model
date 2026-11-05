# -*- coding: utf-8 -*-
"""
train_torch.py —— 用 PyTorch 训练同一套两层决策模型（GPU 可选）

设计原则：**先把 numpy 版原样等价地搬到 PyTorch，再用 GPU 做更强的版本。**
理由：直接上「更花哨的模型 + 不同的数据管线」时，指标一有变化就分不清
是「迁移写错了」还是「新想法有用」。所以本文件提供两种模式：

  --mode equiv  严格等价：与 train.py 同数据、同超参、同损失、同选轮口径，
                目标是验证集指标与 numpy 版**同量级**（不要求逐位相同，
                因为两边的 epoch 内随机顺序不同）。
                 用途：证明迁移没写错。

  --mode fast   GPU 原生：情境成组整批上卡、全品类候选评分、
                自动混合精度（bf16/fp16）、更大的 batch。
                 用途：这才是新环境真正该用的训练方式。

与原实现逐条对齐的细节（代码里都有注释）：
  · ContextEncoder  : 8 个类别 embedding + 2 个数值特征 -> 96 -> 24
  · CuisineNet      : 24 -> 19 个品类 logits，类别加权交叉熵（按类频倒数归一）
  · DishScorer      : [情境编码(24) ‖ dish_id_emb(8) ‖ 7 个菜品属性 embedding ‖ 2 数值]
                      = 63 -> 48 -> 1
  · 排序损失        : 品类内 masked soft-CE，目标是「每道菜各自的真实效用」
                      经 softmax_rows(temp=0.02) 得到的软分布
  · 回归损失        : 只在被观测的那道菜上算 Huber(delta=0.15)，权重 LAM_REG=0.3
  · 权重衰减        : 只作用于 W1/W2（不含 bias），与 numpy 版一致
  · 选轮            : 一律用验证集排序损失（不看 top-1，历史教训）

用法：
    python train_torch.py --mode equiv              # 等价验证（约 1 分钟）
    python train_torch.py --mode fast --epochs 200  # GPU 原生训练
    python train_torch.py --mode fast --fp16 bf16 --batch 512
    python train_torch.py --mode fast --out artifacts_torch   # 不覆盖现有模型

产物（--out 目录）：
    model_torch.pt       PyTorch state_dict + 元信息
    metrics_torch.json   训练/验证/测试指标与耗时
  默认写到 artifacts_torch/，**不会覆盖** artifacts/model.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch.nn.functional as F

import features as feat
from features import (ART, CTX_CAT_COLS, CTX_NUM_COLS, CAT_TO_ID, CTX_VOCAB,
                      DISH_CAT_COLS, DISH_VOCAB, DISH_CAT_IDX, DISH_NAMES,
                      N_DISH, N_CAT, emb_dim)
import train as NP          # numpy 版：复用数据打包、目标构造、评估函数

# ---- 与 train.py 完全一致的超参 ----
SEED = 7
EPOCHS = 60
LR = 3e-3
BATCH = 64
REG = 1e-4
HUBER_DELTA = 0.15
LAM_REG = 0.3
TEMP = 0.02
D_CTX = 24
HID_CTX = 96
# 第二层隐层宽度。
#
# 曾一度改成 96：单 seed（seed=7）下 h96 的 top-1 比 h48 高 5.16pp、后悔值低 36%，
# 看起来是很划算的容量提升（参数只 +14%）。
#
# 但 sweep_seeds.py 用 5 个 seed 复测后推翻了这个结论：
#   top-1   h48 41.77%±2.02%  vs  h96 43.55%±1.33%   配对 t=+1.60, p=0.110
#   后悔值  h48 0.0161±0.0012 vs  h96 0.0151±0.0078  配对 t=-0.26, p=0.796
#   top-3   h48 54.21%±3.26%  vs  h96 55.92%±1.63%   配对 t=+1.34, p=0.179
# 三项全部不显著，且 h96 只在 3/5 个 seed 上占优、逐 seed 方差反而更大
# （后悔值 sd 0.0078 vs 0.0012）。结论：当前数据量（5,162 行）撑不起更大容量，
# 保持 48 与线上模型一致。要再试容量，先把数据量放大（见 docs/交接文档.md）。
HID_DISH = 48
ID_DIM = 8


# ===========================================================================
# 模型：与 numpy 版结构、维度、初始化口径一致
# ===========================================================================
def he_init_(t: torch.Tensor, fan_in: int):
    with torch.no_grad():
        t.normal_(0.0, (2.0 / fan_in) ** 0.5)


class ContextEncoder(nn.Module):
    """8 个类别列各自的 embedding ‖ 2 个数值列 -> 96 -> 24"""

    def __init__(self, d_out=D_CTX, hidden=HID_CTX):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(len(CTX_VOCAB[c]), emb_dim(c, len(CTX_VOCAB[c])))
            for c in CTX_CAT_COLS
        ])
        for e in self.embs:
            with torch.no_grad():
                e.weight.normal_(0.0, 0.1)
        d_in = sum(e.embedding_dim for e in self.embs) + len(CTX_NUM_COLS)
        self.d_in = d_in
        self.W1 = nn.Linear(d_in, hidden)
        self.W2 = nn.Linear(hidden, d_out)
        he_init_(self.W1.weight, d_in)
        he_init_(self.W2.weight, hidden)
        with torch.no_grad():
            self.W1.bias.zero_(); self.W2.bias.zero_()

    def forward(self, cat_idx, num):
        e = torch.cat([emb(cat_idx[:, i]) for i, emb in enumerate(self.embs)], dim=-1)
        x = torch.cat([e, num], dim=-1)
        return self.W2(torch.relu(self.W1(x)))


class CuisineNet(nn.Module):
    def __init__(self, hidden=HID_CTX, d_ctx=D_CTX):
        super().__init__()
        self.enc = ContextEncoder(d_ctx, hidden)
        self.head = nn.Linear(d_ctx, N_CAT)
        he_init_(self.head.weight, d_ctx)
        with torch.no_grad():
            self.head.bias.zero_()

    def forward(self, cat_idx, num):
        return self.head(self.enc(cat_idx, num))


class DishScorer(nn.Module):
    """[情境编码 ‖ dish_id_emb ‖ 菜品属性 embedding ‖ 菜品数值] -> 48 -> 1"""

    def __init__(self, hidden=HID_DISH, d_ctx=D_CTX, id_dim=ID_DIM):
        super().__init__()
        self.enc = ContextEncoder(d_ctx, HID_CTX)
        self.d_ctx = d_ctx
        self.dish_id_emb = nn.Embedding(N_DISH, id_dim)
        self.dish_embs = nn.ModuleList([
            nn.Embedding(len(DISH_VOCAB[c]), emb_dim(c, len(DISH_VOCAB[c])))
            for c in DISH_CAT_COLS
        ])
        for m in [self.dish_id_emb, *self.dish_embs]:
            with torch.no_grad():
                m.weight.normal_(0.0, 0.1)
        d_dish = id_dim + sum(e.embedding_dim for e in self.dish_embs) + len(feat.DISH_NUM_COLS)
        self.d_dish = d_dish
        self.W1 = nn.Linear(d_ctx + d_dish, hidden)
        self.W2 = nn.Linear(hidden, 1)
        he_init_(self.W1.weight, d_ctx + d_dish)
        he_init_(self.W2.weight, hidden)
        with torch.no_grad():
            self.W1.bias.zero_(); self.W2.bias.zero_()

    def forward(self, cat_idx, num, dish_ids, dish_cat, dish_num):
        """dish_* 支持 [B,K,...] 或 [K,...]（后者会在 batch 维广播）。"""
        B = cat_idx.shape[0]
        z = self.enc(cat_idx, num)                              # [B, d_ctx]
        z = z[:, None, :].expand(B, dish_ids.shape[-1], z.shape[-1])
        e_id = self.dish_id_emb(dish_ids)
        e_cat = torch.cat([emb(dish_cat[..., i]) for i, emb in enumerate(self.dish_embs)],
                          dim=-1)
        if e_cat.shape[0] != B:
            e_cat = e_cat.expand(B, -1, -1)
            e_id = e_id.expand(B, -1, -1)
            dish_num = dish_num.expand(B, -1, -1)
        x = torch.cat([z, e_id, e_cat, dish_num], dim=-1)       # [B, K, 63]
        return self.W2(torch.relu(self.W1(x))).squeeze(-1)      # [B, K]


# ===========================================================================
# 损失：与 numpy 版同口径
# ===========================================================================
def soft_targets(gt: np.ndarray, device) -> torch.Tensor:
    """复用 numpy 版 group_targets 的输出，按 temp=0.02 归一成软分布。

    刻意不在 torch 里重写「目标怎么来」——那样一旦目标构造有偏差，
    numpy 与 torch 两边会各错一套，等价性验证就失去意义。
    """
    z = np.where(np.asarray(gt, dtype=np.float64) > -1e8,
                 np.asarray(gt, dtype=np.float64) / TEMP, -1e9)
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    e = np.where(np.isfinite(e), e, 0.0)
    p = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-300)
    return torch.as_tensor(p, dtype=torch.float32, device=device)


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """软目标交叉熵，按 batch 归一（= numpy softmax_ce_soft 的 loss 口径）。"""
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(-1).mean()


# ===========================================================================
# 数据：一次性搬到 GPU
# ===========================================================================
def _masked_utilities() -> torch.Tensor:
    """[N_SCENARIO, N_DISH] 真实效用矩阵，用于构造排序目标。"""
    return torch.as_tensor(np.array(NP.util_matrix(), dtype=np.float32, copy=True))


def build_cuisine_data(scen: pd.DataFrame, device):
    tr = scen[scen.split == "train"].reset_index(drop=True)
    va = scen[scen.split == "val"].reset_index(drop=True)
    def enc(df):
        c, n = feat.encode_context(df)
        return (torch.as_tensor(c, device=device), torch.as_tensor(n, device=device))
    y = torch.as_tensor(np.array(tr.gold_category.map(CAT_TO_ID).values, dtype=np.int64), device=device)
    yv = torch.as_tensor(np.array(va.gold_category.map(CAT_TO_ID).values, dtype=np.int64), device=device)
    return enc(tr), enc(va), y, yv


def build_dish_data(samples: pd.DataFrame, device):
    tr = NP.attach_gold(samples[samples.split == "train"].reset_index(drop=True))
    va = NP.attach_gold(samples[samples.split == "val"].reset_index(drop=True))
    P = NP._pack(tr)
    V = NP._pack(va)

    FLOAT_KEYS = {"y"}          # observed_signal 在 numpy 里是 float64
    def to_dev(d):
        out = {}
        for k, v in d.items():
            arr = np.array(v, copy=True)
            if k in FLOAT_KEYS:
                arr = arr.astype(np.float32)      # 与模型 dtype 对齐，避免 Double/Float 冲突
            out[k] = torch.as_tensor(np.ascontiguousarray(arr), device=device)
        return out
    Pd, Vd = to_dev(P), to_dev(V)

    # 按情境分组（与 numpy 版同样的组采样，保证每批含 chosen+considered）
    groups: dict[int, list[int]] = {}
    for r, sid in enumerate(P["sid"]):
        groups.setdefault(int(sid), []).append(r)
    return Pd, Vd, list(groups.values()), tr


def dish_batch_tensors(device):
    """全部菜品作为候选：ids/dish_cat/dish_num 都是 [K,...]，batch 维广播。"""
    ids = torch.arange(N_DISH, device=device)[None, :]
    dc = torch.as_tensor(np.ascontiguousarray(DISH_CAT_IDX), device=device)           # [K, 7]
    dn = torch.as_tensor(np.array(feat.DISH_NUM_FEAT, dtype=np.float32, copy=True), device=device)  # [K, 2]
    return ids, dc, dn


# ===========================================================================
# 训练：第一层
# ===========================================================================
def train_cuisine_torch(scen, device, epochs, batch, verbose=True):
    (Xc, Xn), (Vc, Vn), y, yv = build_cuisine_data(scen, device)
    torch.manual_seed(SEED)
    model = CuisineNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    cnt = torch.bincount(y, minlength=N_CAT).float()
    cw = (cnt.sum() / (N_CAT * cnt.clamp(min=1))).to(device)
    cw = cw / cw.mean()

    n = y.shape[0]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    best = (1e9, None, -1)
    hist = []
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch].to(device)
            logits = model(Xc[b], Xn[b])
            per = F.cross_entropy(logits, y[b], reduction="none")
            w = cw[y[b]]
            loss = (per * w).sum() / w.sum()     # 加权后按权重和归一
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in model.named_parameters():
                if name.endswith("weight"):
                    p.grad.add_(REG * p)
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            acc = float((model(Vc, Vn).argmax(1) == yv).float().mean())
        hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": acc})
        if 1 - acc < best[0]:
            best = (1 - acc, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  loss={tot / n:.4f}  val_acc={acc:.4f}", flush=True)
    model.load_state_dict(best[1])
    if verbose:
        print(f"    -> 取验证最好的第 {best[2]} 轮，val_acc={1 - best[0]:.4f}", flush=True)
    return model, hist


# ===========================================================================
# 训练：第二层
# ===========================================================================
def train_dish_torch(samples, device, epochs, batch, verbose=True, amp="none"):
    P, V, group_list, tr = build_dish_data(samples, device)
    ids, dc, dn = dish_batch_tensors(device)
    umat = _masked_utilities().to(device)
    dish_cat_of = torch.as_tensor(np.ascontiguousarray(DISH_CAT_IDX[:, 0]), device=device)

    torch.manual_seed(SEED)
    model = DishScorer().to(device)
    decays, no_decay = [], []
    for name, p in model.named_parameters():
        (decays if name.endswith("weight") else no_decay).append(p)
    opt = torch.optim.Adam([{"params": decays, "weight_decay": 0.0},
                            {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    # 注意：L2 只在 W1/W2 上，且与 numpy 版一样按梯度累加，不用 Adam 的 weight_decay

    use_amp = amp != "none" and device.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    n_groups = len(group_list)
    rng = np.random.default_rng(SEED + 1)
    best = (1e9, None, -1)
    hist = []
    # 【性能关键】_pack(tr) 必须提到循环外。
    # 它会把**整张**训练表重新编码一遍（encode_context + pandas map），
    # 而旧代码每个 batch 都调一次 —— 复杂度 O(行数 × batch 数 × epoch 数)，
    # 也就是 O(n²)。2000 情境（5180 行 × 81 batch）时只是慢，能忍；
    # 放量到 2 万情境（58676 行 × 1160 batch）后变成 40 亿次行编码，
    # 实测一个 seed 跑 80 分钟都出不来（GPU 却只有 15% 利用率）。
    # 它是纯函数、参数也不随迭代变化，提到循环外即可，数值完全等价。
    packed_tr = NP._pack(tr)
    t_start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_groups)
        n_rank = n_rank_rows = n_reg_rows = 0
        tot_reg = 0.0
        for i in range(0, n_groups, batch):
            rows = [r for gi in perm[i:i + batch] for r in group_list[gi]]
            b = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)
            B = len(rows)
            ck, nk = P["cat"][b], P["num"][b]

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                s = model(ck, nk, ids, dc, dn)                  # [B, K]
            s = s.float()

            is_chosen = P["chosen"][b]
            gt = NP.group_targets(packed_tr, rows)               # 复用 numpy 的目标构造
            tgt = soft_targets(gt, device)
            group_mask = (dish_cat_of[None, :] == P["cat_id"][b][:, None])
            s_m = s.masked_fill(~group_mask, -1e9)

            if bool(is_chosen.any()):
                loss_rank = soft_ce(s_m[is_chosen], tgt[is_chosen])
            else:
                loss_rank = torch.zeros((), device=device)

            pred_obs = s[torch.arange(B, device=device), P["dish"][b]]
            # 与 numpy 版逐字对齐：numpy 用 huber(pred, y).mean() 得到标量，
            # 梯度按 d_reg / B 回传 —— 等价于「对 batch 内 B 行取平均」。
            # 用 reduction="sum"/B 会得到 mean 的 B 倍，梯度大 B 倍；
            # 而用 mean 又不能再把累积量当 sum 存，否则打印出来的 reg 放大 B 倍
            # （这正是先前 reg=10.8 越训越大的原因：口径错，不是发散）。
            loss_reg = F.huber_loss(pred_obs, P["y"][b], delta=HUBER_DELTA,
                                    reduction="mean")

            loss = loss_rank + LAM_REG * loss_reg
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                for name, p in model.named_parameters():
                    if name.endswith("weight"):
                        p.grad.add_(REG * p)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                for name, p in model.named_parameters():
                    if name.endswith("weight"):
                        p.grad.add_(REG * p)
                opt.step()

            n_rank += float(loss_rank.detach()) * int(is_chosen.sum())
            n_rank_rows += int(is_chosen.sum())
            tot_reg += float(loss_reg.detach()) * B
            n_reg_rows += B

        vl = eval_rank_torch(model, V, ids, dc, dn, dish_cat_of, umat)
        v1 = eval_top1_torch(model, V, ids, dc, dn, dish_cat_of)
        hist.append({"epoch": ep, "train_rank": n_rank / max(n_rank_rows, 1),
                     "train_reg": tot_reg / max(n_reg_rows, 1),
                     "val_rank": vl, "val_top1": v1})
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  rank={n_rank / max(n_rank_rows, 1):.4f}"
                  f"  reg={tot_reg / max(n_reg_rows, 1):.4f}"
                  f"  val_rank={vl:.4f}  val_top1(仅参考)={v1:.2%}", flush=True)
    model.load_state_dict(best[1])
    if verbose:
        print(f"    -> 取验证排序损失最低的第 {best[2]} 轮，val_rank={best[0]:.4f}", flush=True)
    return model, hist, time.perf_counter() - t_start


# ===========================================================================
# 评估（与 numpy 版同口径：只看 chosen 行，对照 gold）
# ===========================================================================
@torch.no_grad()
def _scores_for_rows(model, P, b, ids, dc, dn):
    return model(P["cat"][b], P["num"][b], ids, dc, dn)


@torch.no_grad()
def eval_rank_torch(model, P, ids, dc, dn, dish_cat_of, umat, chunk=256):
    model.eval()
    keep = torch.nonzero(P["chosen"]).flatten()
    tot, n = 0.0, 0
    for i in range(0, len(keep), chunk):
        b = keep[i:i + chunk]
        s = _scores_for_rows(model, P, b, ids, dc, dn)
        gt = umat[P["sid"][b]].clone()
        gt = torch.where(dish_cat_of[None, :] == P["cat_id"][b][:, None], gt,
                         torch.full_like(gt, -1e9))
        tgt = soft_targets(gt.float().cpu().numpy(), s.device)
        s_m = s.masked_fill(~(dish_cat_of[None, :] == P["cat_id"][b][:, None]), -1e9)
        tot += float(soft_ce(s_m, tgt)) * len(b)
        n += len(b)
    return tot / max(n, 1)


@torch.no_grad()
def eval_top1_torch(model, P, ids, dc, dn, dish_cat_of, chunk=256):
    model.eval()
    keep = torch.nonzero(P["chosen"]).flatten()
    hit, n = 0, 0
    for i in range(0, len(keep), chunk):
        b = keep[i:i + chunk]
        s = _scores_for_rows(model, P, b, ids, dc, dn)
        s_m = s.masked_fill(~(dish_cat_of[None, :] == P["cat_id"][b][:, None]), -1e9)
        hit += int((s_m.argmax(1) == P["gold"][b]).sum())
        n += len(b)
    return hit / max(n, 1)


@torch.no_grad()
def eval_val_topk(model, P, ids, dc, dn, dish_cat_of, k=3, chunk=256):
    model.eval()
    keep = torch.nonzero(P["chosen"]).flatten()
    hit, n = 0, 0
    for i in range(0, len(keep), chunk):
        b = keep[i:i + chunk]
        s = _scores_for_rows(model, P, b, ids, dc, dn)
        s_m = s.masked_fill(~(dish_cat_of[None, :] == P["cat_id"][b][:, None]), -1e9)
        topk = s_m.topk(k, dim=1).indices
        hit += int((topk == P["gold"][b][:, None]).any(1).sum())
        n += len(b)
    return hit / max(n, 1)


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["equiv", "fast"], default="equiv")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--out", default="artifacts_torch")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_s)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    epochs = args.epochs or (EPOCHS if args.mode == "equiv" else 200)
    batch = args.batch or (BATCH if args.mode == "equiv" else 256)
    amp = "none" if args.mode == "equiv" else args.amp

    outdir = HERE / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    scen = pd.read_csv(ART / "scenarios.csv")
    samples = pd.read_csv(ART / "samples.csv")
    for df in (scen, samples):
        df["taboo"] = df["taboo"].fillna("").astype(str)

    print("=" * 68)
    print(f"PyTorch 训练  模式={args.mode}  设备={device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"  菜品 {N_DISH} 道 / 品类 {N_CAT} 个 / 训练行 {len(samples[samples.split=='train'])}"
          f" / epochs={epochs} batch={batch} amp={amp}")
    print("=" * 68)

    t0 = time.perf_counter()
    print("\n【第一层】情境 -> 品类")
    cm, hist_c = train_cuisine_torch(scen, device, epochs, batch)
    t1 = time.perf_counter()
    print(f"    用时 {t1 - t0:.1f}s")

    print("\n【第二层】情境 + 菜品 -> 效用分")
    dm, hist_d, t_dish = train_dish_torch(samples, device, epochs, batch,
                                          verbose=True, amp=amp)
    t2 = time.perf_counter()
    print(f"    用时 {t_dish:.1f}s")

    # ---- 评估：与 numpy 版同一套口径 ----
    P, V, _, _ = build_dish_data(samples, device)
    ids, dc, dn = dish_batch_tensors(device)
    dish_cat_of = torch.as_tensor(np.ascontiguousarray(DISH_CAT_IDX[:, 0]), device=device)
    v_rank = eval_rank_torch(dm, V, ids, dc, dn, dish_cat_of, _masked_utilities().to(device))
    v_top1 = eval_top1_torch(dm, V, ids, dc, dn, dish_cat_of)
    v_top3 = eval_val_topk(dm, V, ids, dc, dn, dish_cat_of, k=3)
    yv_acc = None

    res = {
        "mode": args.mode, "device": str(device), "epochs": epochs, "batch": batch,
        "amp": amp, "seed": SEED,
        "n_dish": N_DISH, "n_cat": N_CAT,
        "val_rank": round(float(v_rank), 5),
        "val_top1": round(float(v_top1), 5),
        "val_top3": round(float(v_top3), 5),
        "train_seconds_total": round(t2 - t0, 2),
        "train_seconds_layer1": round(t1 - t0, 2),
        "train_seconds_layer2": round(t_dish, 2),
        "n_params": int(sum(p.numel() for p in dm.parameters())
                        + sum(p.numel() for p in cm.parameters())),
        "hist_cuisine": hist_c,
        "hist_dish": hist_d,
    }
    torch.save({"cuisine": cm.state_dict(), "dish": dm.state_dict(),
                "meta": {k: v for k, v in res.items() if not k.startswith("hist")}},
               outdir / "model_torch.pt")
    (outdir / "metrics_torch.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n【结果】")
    print(f"  验证排序损失 : {res['val_rank']}")
    print(f"  验证 top-1   : {res['val_top1']:.2%}")
    print(f"  验证 top-3   : {res['val_top3']:.2%}")
    print(f"  参数量       : {res['n_params']:,}")
    print(f"  训练总用时   : {res['train_seconds_total']:.1f}s"
          f"  (第一层 {res['train_seconds_layer1']:.1f}s / 第二层 {res['train_seconds_layer2']:.1f}s)")
    print(f"  已保存 -> {outdir / 'model_torch.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
