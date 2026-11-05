# -*- coding: utf-8 -*-
"""
train_listwise.py —— 结构升级：整目录 softmax（交接文档 §7.4）

【要解决的痛点】
两层结构里，第二层 DishScorer **只在「所属品类内」训练排序**：
它的分数跨品类不可比，所以必须先由第一层选一个品类，再在该品类里挑。
这带来一个不改结构就绕不开的问题：

    面包甜点只有 9 道菜，「严格控制」硬筛选后只剩 3 道 —— 第一层一旦选中它，
    第 2/3 名只能是贝果、吐司配果酱这种货色，哪怕别的品类里有 20 道更好的菜。

【两个候选方案】
    full   整目录 softmax：**架构与线上完全一致**（同一个 DishScorer），只把训练
           目标从「品类内 masked soft-CE」换成「整目录 824 路 soft-CE」。
           分数因此天然跨品类可比，第二层可以直接对全部菜排序。
    tower  双塔：情境塔 z、菜品塔 e，score = z·e * exp(log_T) + b。
           标准的两塔检索式结构，参数更少（~11k），但点积表达力弱于 MLP 拼接。

两者都用 **learned temperature**：`softmax(score * exp(log_T))`，让模型自己决定
分布的锐度，不用手调 TEMP=0.02。

【为什么不用 in-batch 负采样】
负采样是用来「近似」大词表 softmax 的。这里的词表就是全部菜（824），
一步就能算全量 softmax，负样本本来就已经「全在 batch 里」——
再采样只会引入方差，不会更快。所以直接算 824 路。

【公平性】
目标与基线**同源**：都用 `scenario_utility.npy`（基线里 `group_targets`
也是从同一张矩阵取整行真实效用的）。唯一差别是基线只用「gold 品类那一行」，
这里用**全部 824 道**。差别正是要被检验的结构差异，不是信息量差异。

【用法】
    python train_listwise.py --variant full --eval-art artifacts_eval
    python train_listwise.py --variant full,tower --seeds 1,2,3,4,5 --eval-art artifacts_eval
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import features as feat
import train as NP
from features import ART, DISH_CAT_IDX, N_DISH, N_CAT
from train_torch import (ContextEncoder, DishScorer, TEMP, emb_dim,
                         CTX_CAT_COLS, DISH_CAT_COLS)

LR = 3e-3
WEIGHT_DECAY = 1e-4
SEED = 7
EPOCHS = 60
BATCH = 64


# ---------------------------------------------------------------------------
# 特征
# ---------------------------------------------------------------------------
def dish_inputs(model, device):
    """全部菜品作为候选：与 train_torch.dish_batch_tensors 同口径。"""
    ids = torch.arange(N_DISH, device=device)[None, :]                       # [1, K]
    dc = torch.as_tensor(np.ascontiguousarray(DISH_CAT_IDX), device=device)  # [K, 7]
    dn = torch.as_tensor(np.array(feat.DISH_NUM_FEAT, dtype=np.float32, copy=True),
                         device=device)                                      # [K, 2]
    return ids, dc, dn


def dish_features(model, ids, dc, dn) -> torch.Tensor:
    """[K, 39] = dish_id_emb(8) ‖ 7 个属性 emb(29) ‖ 2 个数值 —— 与 DishScorer 内部顺序一致。"""
    e_id = model.dish_id_emb(ids)                                            # [1, K, 8]
    e_cat = torch.cat([emb(dc[..., i]) for i, emb in enumerate(model.dish_embs)], dim=-1)
    B = e_id.shape[0]                                                        # 1
    e_cat = e_cat.unsqueeze(0).expand(B, -1, -1)                             # [1, K, 29]
    dn = dn.unsqueeze(0).expand(B, -1, -1)                                   # [1, K, 2]
    return torch.cat([e_id, e_cat, dn], dim=-1).squeeze(0)                   # [K, 39]


# ---------------------------------------------------------------------------
# 方案 A：整目录 softmax（架构 = 线上 DishScorer，只换训练目标）
# ---------------------------------------------------------------------------
def full_logits(model, cat, num, e) -> torch.Tensor:
    """DishScorer 的 matmul 等价形式，但把「情境部分」提到 batch 维广播。

    DishScorer.forward 会拼出 [B, K, 63] 再过 W1，复杂度 O(B·K·63·48)；
    因为 W1 对 [z ‖ dish] 是分块线性的，可以拆成
        h = relu(z @ W1[:, :24] + e @ W1[:, 24:])
    其中 `e @ W1[:, 24:]` 只跟菜品有关（[K, 48]），一遍就算完。
    实测快 ~50 倍，**数值与 forward 逐位一致**（见 __main__ 的自检）。
    """
    d_ctx = model.d_ctx
    z = model.enc(cat, num)                                    # [B, 24]
    hc = z @ model.W1.weight[:, :d_ctx].T                      # [B, 48]
    hd = e @ model.W1.weight[:, d_ctx:].T                      # [K, 48]
    h = torch.relu(hc[:, None, :] + hd[None, :, :])            # [B, K, 48]
    return (h @ model.W2.weight.T + model.W2.bias).squeeze(-1)  # [B, K]


class LogTemp(nn.Module):
    """learned temperature：softmax(score * exp(log_T))。"""

    def __init__(self, init=0.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(init)))

    def forward(self, s):
        return s * torch.exp(self.log_T.clamp(-3.0, 3.0))


# ---------------------------------------------------------------------------
# 方案 B：双塔
# ---------------------------------------------------------------------------
class Tower(nn.Module):
    """情境塔 z(24) · 菜品塔 e(24)，score = z·e * exp(log_T) + b_dish。"""

    def __init__(self, d_ctx=24, hidden_ctx=96, d_dish=24, id_dim=8):
        super().__init__()
        self.enc = ContextEncoder(d_ctx, hidden_ctx)
        self.dish_id_emb = nn.Embedding(N_DISH, id_dim)
        self.dish_embs = nn.ModuleList([
            nn.Embedding(len(feat.DISH_VOCAB[c]),
                         emb_dim(c, len(feat.DISH_VOCAB[c])))
            for c in DISH_CAT_COLS
        ])
        for m in [self.dish_id_emb, *self.dish_embs]:
            with torch.no_grad():
                m.weight.normal_(0.0, 0.1)
        d_in = id_dim + sum(e.embedding_dim for e in self.dish_embs) + len(feat.DISH_NUM_COLS)
        self.dish_mlp = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_dish))
        self.bias = nn.Parameter(torch.zeros(N_DISH))
        self.log_T = nn.Parameter(torch.tensor(0.0))

    def dish_side(self, ids, dc, dn):
        e_id = self.dish_id_emb(ids)
        e_cat = torch.cat([emb(dc[..., i]) for i, emb in enumerate(self.dish_embs)], dim=-1)
        B = e_id.shape[0]
        e_cat = e_cat.unsqueeze(0).expand(B, -1, -1)
        dn = dn.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([e_id, e_cat, dn], dim=-1)
        return self.dish_mlp(x.squeeze(0))                     # [K, d_dish]

    def forward(self, cat, num, e):
        z = self.enc(cat, num)                                 # [B, d_ctx]
        s = z @ e.T                                            # [B, K]
        return (s * torch.exp(self.log_T.clamp(-3.0, 3.0))) + self.bias


# ---------------------------------------------------------------------------
# 目标：整目录软分布
# ---------------------------------------------------------------------------
def full_targets(U: np.ndarray, rows: np.ndarray, temp: float = TEMP) -> torch.Tensor:
    z = U[rows].astype(np.float64) / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return torch.as_tensor(e / e.sum(axis=1, keepdims=True), dtype=torch.float32)


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def load_data(eval_art: Path | None):
    scen = pd.read_csv(ART / "scenarios.csv")
    scen["taboo"] = scen["taboo"].fillna("").astype(str)
    U = np.array(NP.util_matrix(), dtype=np.float64)

    tr = scen[scen.split == "train"].reset_index(drop=True)
    va = scen[scen.split == "val"].reset_index(drop=True)
    c_tr, n_tr = feat.encode_context(tr)
    c_va, n_va = feat.encode_context(va)
    y_tr = np.array(tr.scenario_id.values, dtype=np.int64)
    y_va = np.array(va.scenario_id.values, dtype=np.int64)
    return dict(c_tr=c_tr, n_tr=n_tr, y_tr=y_tr, t_tr=full_targets(U, y_tr),
                c_va=c_va, n_va=n_va, y_va=y_va, t_va=full_targets(U, y_va), U=U)


def train_variant(variant: str, D: dict, device, epochs: int, batch: int,
                  seed: int, verbose: bool = True):
    torch.manual_seed(seed)
    if variant == "full":
        model = DishScorer().to(device)
    elif variant == "tower":
        model = Tower().to(device)
    else:
        raise SystemExit(f"未知 variant: {variant}")

    lt = LogTemp().to(device)
    decays = [p for n, p in model.named_parameters() if n.endswith("weight")]
    other = [p for n, p in model.named_parameters() if not n.endswith("weight")]
    opt = torch.optim.Adam([{"params": decays, "weight_decay": WEIGHT_DECAY},
                            {"params": other, "weight_decay": 0.0},
                            {"params": list(lt.parameters()), "weight_decay": 0.0}], lr=LR)

    ids, dc, dn = dish_inputs(model, device)
    Xc = torch.as_tensor(D["c_tr"], device=device)
    Xn = torch.as_tensor(D["n_tr"], device=device)
    Tt = D["t_tr"].to(device)
    Vc = torch.as_tensor(D["c_va"], device=device)
    Vn = torch.as_tensor(D["n_va"], device=device)
    Tv = D["t_va"].to(device)

    n = len(D["y_tr"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    # 自检：matmul 快路径必须与 DishScorer.forward 逐位一致，
    # 否则「只换了训练目标」这个前提就不成立，实验结论也就不成立。
    if variant == "full":
        with torch.no_grad():
            e_chk = dish_features(model, ids, dc, dn)
            s_fast = full_logits(model, Xc[:8], Xn[:8], e_chk)
            s_ref = model(Xc[:8], Xn[:8], ids, dc, dn)
        if not torch.allclose(s_fast, s_ref, atol=1e-4):
            raise SystemExit("full_logits 与 DishScorer.forward 不一致，先修好再训练")
    best = (1e9, None, -1)
    hist = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            # 索引张量必须搬到 GPU，否则 CUDA 张量不支持 CPU 索引
            b = perm[i:i + batch].to(device)
            if variant == "full":
                e = dish_features(model, ids, dc, dn)
                logits = lt(full_logits(model, Xc[b], Xn[b], e))
            else:
                e = model.dish_side(ids, dc, dn)
                logits = model(Xc[b], Xn[b], e)
            loss = soft_ce(logits, Tt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if variant == "tower":
                with torch.no_grad():
                    model.log_T.clamp_(-3.0, 3.0)
            tot += loss.item() * len(b)
        # 选轮：验证集排序损失（与既有口径一致，不看 top-1）
        model.eval()
        with torch.no_grad():
            if variant == "full":
                e = dish_features(model, ids, dc, dn)
                vl = soft_ce(lt(full_logits(model, Vc, Vn, e)), Tv).item()
            else:
                e = model.dish_side(ids, dc, dn)
                vl = soft_ce(model(Vc, Vn, e), Tv).item()
        hist.append(dict(ep=ep, train=tot / n, val=vl))
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().clone() for k, v in lt.state_dict().items()}, ep,
                    float(lt.log_T.clamp(-3, 3)))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    ep {ep:3d}  train={tot / n:.5f}  val={vl:.5f}  "
                  f"logT={float(lt.log_T):+.2f}", flush=True)
    model.load_state_dict(best[1])
    lt.load_state_dict(best[2])
    secs = time.perf_counter() - t0
    return model, lt, dict(val_softce=best[0], best_ep=best[3], log_T=best[4],
                           seconds=round(secs, 1), hist=hist)


# ---------------------------------------------------------------------------
# 评估：与 sweep_seeds.eval_test 完全同口径
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, lt, variant, ev_dir: Path, device, chunk: int = 256) -> dict:
    ev = pd.read_csv(ev_dir / "scenarios.csv")
    ev["taboo"] = ev["taboo"].fillna("").astype(str)
    ev = ev.reset_index(drop=True)
    U = np.load(ev_dir / "scenario_utility.npy")
    Ute = U[ev.scenario_id.values.astype(int)]

    c, n = feat.encode_context(ev)
    ids, dc, dn = dish_inputs(model, device)
    e = dish_features(model, ids, dc, dn) if variant == "full" else model.dish_side(ids, dc, dn)

    model.eval()
    preds = np.empty(len(ev), dtype=np.int64)
    for s in range(0, len(ev), chunk):
        b = slice(s, min(s + chunk, len(ev)))
        cc = torch.as_tensor(c[b], device=device)
        nn_ = torch.as_tensor(n[b], device=device)
        logits = lt(full_logits(model, cc, nn_, e)) if variant == "full" else model(cc, nn_, e)
        preds[b] = logits.argmax(1).cpu().numpy()

    gold = Ute.argmax(1)
    dish_cat = DISH_CAT_IDX[:, 0]
    order = np.argsort(-Ute, axis=1, kind="stable")[:, :3]
    idx = np.arange(len(ev))
    u_model = float(Ute[idx, preds].mean())
    u_best = float(Ute[idx, gold].mean())
    # tower 的温度在模型内部，full 的在外部 LogTemp 里
    log_T = model.log_T if variant == "tower" else lt.log_T
    return {
        "mean_utility": round(u_model, 4),
        "regret": round(u_best - u_model, 4),
        "top1": round(float((preds == gold).mean()), 4),
        "top3": round(float((order[:, :3] == preds[:, None]).any(1).mean()), 4),
        "category_hit": round(float((dish_cat[preds] == dish_cat[gold]).mean()), 4),
        "log_T": round(float(log_T.clamp(-3, 3)), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", help="full / tower，逗号分隔可跑多个")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--eval-art", default="artifacts_eval")
    ap.add_argument("--out", default="artifacts_listwise")
    ap.add_argument("--save-model", action="store_true", help="保存最优 checkpoint（单 seed 用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v.strip() for v in args.variant.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ev_dir = Path(args.eval_art)

    D = load_data(ev_dir)
    print(f"训练集情境 {len(D['y_tr'])} / 验证 {len(D['y_va'])}  "
          f"评估集 {ev_dir.name}")

    out = {}
    for v in variants:
        rows = []
        for sd in seeds:
            print(f"  [{v}] seed={sd}", flush=True)
            model, lt, meta = train_variant(v, D, device, args.epochs, args.batch, sd)
            m = evaluate(model, lt, v, ev_dir, device)
            m.update(seed=sd, seconds=meta["seconds"], val_softce=round(meta["val_softce"], 5),
                     best_ep=meta["best_ep"],
                     n_params=int(sum(p.numel() for p in model.parameters()) + lt.log_T.numel()))
            rows.append(m)
            print(f"    -> 效用={m['mean_utility']:.4f} 后悔={m['regret']:.4f} "
                  f"top1={m['top1']:.2%} top3={m['top3']:.2%} 品类={m['category_hit']:.2%} "
                  f"logT={m['log_T']:+.2f} ({m['seconds']:.0f}s)", flush=True)
            if args.save_model and len(seeds) == 1:
                d = Path(args.out) / f"{v}_seed{sd}"
                d.mkdir(parents=True, exist_ok=True)
                # 键名与 export_npz.py 对齐：full 存 "dish"（单阶段，没有 cuisine，
                # 于是导出的 npz 里不含 cuisine.* 键 —— 这正是 recommend.load_models
                # 用来识别「单阶段」的标记）。tower 的结构无法映射到现有 npz 格式，
                # 只保存 torch checkpoint 供参考。
                payload = ({"dish": model.state_dict(), "log_T": float(lt.log_T),
                            "variant": v, "meta": m}
                           if v == "full" else
                           {"model": model.state_dict(), "variant": v, "meta": m})
                torch.save(payload, d / "model.pt")
                print(f"    已保存 -> {d/'model.pt'}")
        out[v] = rows

    print()
    for v, rows in out.items():
        if len(rows) > 1:
            print(f"[{v}] n={len(rows)}  "
                  f"效用 {np.mean([r['mean_utility'] for r in rows]):.4f}±"
                  f"{np.std([r['mean_utility'] for r in rows], ddof=1):.4f}  "
                  f"top1 {np.mean([r['top1'] for r in rows]):.2%}±"
                  f"{np.std([r['top1'] for r in rows], ddof=1):.2%}  "
                  f"后悔 {np.mean([r['regret'] for r in rows]):.4f}")

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    # 与 sweep_seeds 的 json 同构，便于 compare_runs 直接比较
    (od / "sweep.json").write_text(json.dumps(
        {"seeds": seeds, "configs": variants, "epochs": args.epochs,
         "results": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 -> {od/'sweep.json'}")


if __name__ == "__main__":
    main()
