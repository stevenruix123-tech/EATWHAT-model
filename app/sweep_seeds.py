# -*- coding: utf-8 -*-
"""
sweep_seeds.py —— 多 seed 复测，确认「h96 比 h48 好」不是单 seed 的噪声

为什么必须做：上一轮我基于**单 seed + 8 个配置**得出「容量已无提升空间」，
后来被 h96 的结果推翻。根因有两个：
  1. 单 seed —— 初始化方差没有估计，任何 <2×sd 的差异都可能只是运气；
  2. 验证集只有 284 个情境 —— 1 个百分点 ≈ 2.8 个样本。

本脚本对每组配置跑 N 个 seed，报告「均值 ± 标准差」与配对差，
并用配对 t 检验判断差异是否显著。

用法：
    python sweep_seeds.py --seeds 1,2,3,4,5 --configs h48,h96
    python sweep_seeds.py --seeds 7 --configs h48,h96 --epochs 60   # 单 seed 快查
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_torch as TT
from features import ART, DISH_CAT_IDX

# 配置表：(hid_dish, d_ctx, id_dim, hid_ctx)
CONFIGS = {
    "h48": (48, 24, 8, 96),
    "h96": (96, 24, 8, 96),
    "h192": (192, 24, 8, 96),
}


def set_config(cfg: str):
    hid_dish, d_ctx, id_dim, hid_ctx = CONFIGS[cfg]
    TT.DishScorer.__init__.__defaults__ = (hid_dish, d_ctx, id_dim)
    TT.CuisineNet.__init__.__defaults__ = (hid_ctx, d_ctx)
    TT.ContextEncoder.__init__.__defaults__ = (d_ctx, hid_ctx)


@torch.no_grad()
def eval_test(cm, dm, sc, te, Ute, U, device):
    """与 eval_torch.py 完全相同的测试集口径。"""
    import features as feat
    from features import DISH_CAT_IDX as DCI, N_DISH

    c, n = feat.encode_context(te)
    ck = torch.as_tensor(np.ascontiguousarray(c), device=device)
    nk = torch.as_tensor(np.array(n, dtype=np.float32, copy=True), device=device)
    win = cm(ck, nk).argmax(1).cpu().numpy()

    ids, dc, dn = TT.dish_batch_tensors(device)
    scores = dm(ck, nk, ids, dc, dn).float().cpu().numpy()
    dish_cat = DCI[:, 0]
    tops = np.zeros((len(te), 3), dtype=int)
    for i in range(len(te)):
        cand = np.where(dish_cat == win[i])[0]
        if len(cand) == 0:
            cand = np.arange(N_DISH)
        tops[i, :3] = cand[np.argsort(-scores[i, cand])][:3]

    gold = Ute.argmax(1)
    gold_cat = DCI[gold, 0]
    idx = np.arange(len(te))
    u_model = float(Ute[idx, tops[:, 0]].mean())
    u_best = float(Ute[idx, gold].mean())
    return {
        "mean_utility": round(u_model, 4),
        "regret": round(u_best - u_model, 4),
        "top1": round(float((tops[:, 0] == gold).mean()), 4),
        "top3": round(float((tops[:, :3] == gold[:, None]).any(1).mean()), 4),
        "category_hit": round(float((win == gold_cat).mean()), 4),
    }


def paired_ttest(a: list[float], b: list[float]):
    """配对 t 检验：返回 (均值差, t 值, 近似双尾 p)。a、b 等长。"""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    n = len(d)
    if n < 2:
        return float(d.mean()), float("nan"), float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), float("inf"), 0.0
    t = float(d.mean() / (sd / math.sqrt(n)))
    # 用正态近似给 p（n 小，够用来判断量级）
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(d.mean()), t, float(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--configs", default="h48,h96")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="artifacts_sweep")
    ap.add_argument("--eval-art", default=None,
                    help="在另一份数据集的留出情境上评估（模型仍用 ART 训练）。"
                         "跨数据集/跨数据量的比较必须用它：各数据集自带测试集时，"
                         "绝对指标是在不同样本上算的，不可直接比。")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]

    scen = pd.read_csv(ART / "scenarios.csv")
    samples = pd.read_csv(ART / "samples.csv")
    for df in (scen, samples):
        df["taboo"] = df["taboo"].fillna("").astype(str)
    U = np.load(ART / "scenario_utility.npy")
    if args.eval_art:
        # 公共留出集：全部情境都算（这份数据从未参与训练，不需要再切 split）
        ev = Path(args.eval_art)
        te = pd.read_csv(ev / "scenarios.csv").reset_index(drop=True)
        te["taboo"] = te["taboo"].fillna("").astype(str)
        Ue = np.load(ev / "scenario_utility.npy")
        Ute = Ue[te.scenario_id.values.astype(int)]
        print(f"公共留出集: {ev}  ({len(te)} 个情境)")
    else:
        te = scen[scen.split == "test"].reset_index(drop=True)
        Ute = U[te.scenario_id.values.astype(int)]

    print("=" * 78)
    print(f"多 seed 复测  device={device}  seeds={seeds}  configs={cfgs}  epochs={args.epochs}")
    print("=" * 78)

    out = {}
    for cfg in cfgs:
        set_config(cfg)
        rows = []
        for sd in seeds:
            TT.SEED = sd
            t0 = time.perf_counter()
            cm, _ = TT.train_cuisine_torch(scen, device, args.epochs, args.batch, verbose=False)
            dm, _, _ = TT.train_dish_torch(samples, device, args.epochs, args.batch, verbose=False)
            m = eval_test(cm, dm, scen, te, Ute, U, device)
            m["seed"] = sd
            m["seconds"] = round(time.perf_counter() - t0, 1)
            n_par = sum(p.numel() for p in dm.parameters()) + sum(p.numel() for p in cm.parameters())
            m["n_params"] = int(n_par)
            rows.append(m)
            print(f"  {cfg:<5} seed={sd}  效用={m['mean_utility']:.4f} 后悔={m['regret']:.4f} "
                  f"top1={m['top1']:.2%} top3={m['top3']:.2%} 品类={m['category_hit']:.2%} "
                  f"({m['seconds']:.0f}s)", flush=True)
        out[cfg] = rows

    # ---- 汇总 ----
    def agg(rows, key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    keys = ["mean_utility", "regret", "top1", "top3", "category_hit"]
    fixed = {
        "mean_utility": lambda x: f"{x:.4f}",
        "regret": lambda x: f"{x:.4f}",
        "top1": lambda x: f"{x:.2%}",
        "top3": lambda x: f"{x:.2%}",
        "category_hit": lambda x: f"{x:.2%}",
    }
    print()
    print("=" * 78)
    print(f"汇总（均值 ± 总体标准差，n={len(seeds)} seed）")
    print("=" * 78)
    names = {"mean_utility": "平均效用", "regret": "后悔值", "top1": "top-1",
             "top3": "top-3", "category_hit": "品类命中"}
    hdr = f"{'指标':<10}" + "".join(f"{c:>22}" for c in cfgs)
    print(hdr)
    for k in keys:
        line = f"{names[k]:<10}"
        for c in cfgs:
            mu, sd = agg(out[c], k)
            line += f"{fixed[k](mu) + ' ± ' + fixed[k](sd):>22}"
        print(line)

    # ---- 配对比较（以第一个配置为基线）----
    if len(cfgs) >= 2:
        base = cfgs[0]
        print()
        print(f"配对比较（基线 {base}）")
        for c in cfgs[1:]:
            print(f"  {c} vs {base}:")
            for k in ["top1", "top3", "regret"]:
                a = [r[k] for r in out[c]]
                b = [r[k] for r in out[base]]
                dmu, t, p = paired_ttest(a, b)
                verdict = "显著" if (not math.isnan(p) and p < 0.05) else "不显著"
                print(f"    {names[k]:<8} 均值差 {dmu:+.4f}   t={t:+.2f}  p≈{p:.3f}  -> {verdict}")

    # 每个指标在多少个 seed 上占优
    if len(cfgs) >= 2:
        base = cfgs[0]
        c = cfgs[1]
        print()
        for k in ["top1", "top3", "regret"]:
            a = [r[k] for r in out[c]]
            b = [r[k] for r in out[base]]
            if k == "regret":     # 后悔值越低越好
                wins = sum(1 for x, y in zip(a, b) if x < y)
            else:
                wins = sum(1 for x, y in zip(a, b) if x > y)
            print(f"  {c} 在 {wins}/{len(seeds)} 个 seed 上 {names[k]} 优于 {base}")

    outdir = HERE / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sweep.json").write_text(
        json.dumps({"seeds": seeds, "configs": cfgs, "epochs": args.epochs,
                    "results": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 -> {outdir/'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
