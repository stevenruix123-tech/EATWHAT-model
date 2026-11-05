# -*- coding: utf-8 -*-
"""
probe_capacity.py —— 用 PyTorch 快速验证「容量是否还有提升空间」

背景：numpy 版训练要 740s，做一次容量对比就要十几分钟，所以上一轮的
容量扫描只跑了 8 个配置、单 seed，结论偏弱。PyTorch 版约 117s/次，
同样的工作量现在几分钟就能做完，而且能直接上更宽的模型。

本脚本在**不改动 train_torch.py 默认结构**的前提下，用环境变量式覆盖
（直接改类的默认参数）扫几组容量，输出验证集 top-1 / top-3 / rank。

用法：
    python probe_capacity.py                        # 默认扫 3 个配置
    python probe_capacity.py --configs h48,h96,h192 \
        --epochs 60 --batch 64 --out artifacts_probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features as feat
import train as NP
import train_torch as TT
from features import ART, DISH_CAT_IDX, N_CAT, N_DISH


def run_config(tag, d_ctx, hid_ctx, hid_dish, id_dim, scen, samples, epochs, batch, device):
    """按给定容量训练两层，返回验证指标。"""
    # 覆盖默认超参（构造函数的默认参数）
    TT.DishScorer.__init__.__defaults__ = (hid_dish, d_ctx, id_dim)
    TT.CuisineNet.__init__.__defaults__ = (hid_ctx, d_ctx)
    TT.ContextEncoder.__init__.__defaults__ = (d_ctx, hid_ctx)

    t0 = time.perf_counter()
    cm, _ = TT.train_cuisine_torch(scen, device, epochs, batch, verbose=False)
    dm, hist, _ = TT.train_dish_torch(samples, device, epochs, batch, verbose=False)

    P, V, _, _ = TT.build_dish_data(samples, device)
    ids, dc, dn = TT.dish_batch_tensors(device)
    dish_cat_of = torch.as_tensor(np.ascontiguousarray(DISH_CAT_IDX[:, 0]), device=device)
    umat = TT._masked_utilities().to(device)
    v_rank = TT.eval_rank_torch(dm, V, ids, dc, dn, dish_cat_of, umat)
    v_top1 = TT.eval_top1_torch(dm, V, ids, dc, dn, dish_cat_of)
    v_top3 = TT.eval_val_topk(dm, V, ids, dc, dn, dish_cat_of, k=3)
    dt = time.perf_counter() - t0
    n_params = sum(p.numel() for p in dm.parameters()) + sum(p.numel() for p in cm.parameters())

    return {
        "tag": tag, "d_ctx": d_ctx, "hid_ctx": hid_ctx, "hid_dish": hid_dish,
        "id_dim": id_dim, "n_params": int(n_params),
        "val_rank": round(float(v_rank), 5),
        "val_top1": round(float(v_top1), 5),
        "val_top3": round(float(v_top3), 5),
        "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="h48,h96,h192")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--amp", default="none")
    ap.add_argument("--out", default="artifacts_probe")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scen = pd.read_csv(ART / "scenarios.csv")
    samples = pd.read_csv(ART / "samples.csv")
    for df in (scen, samples):
        df["taboo"] = df["taboo"].fillna("").astype(str)

    # 配置表：(标签, d_ctx, hid_ctx, hid_dish, id_dim)
    # 默认 h48 = 当前线上模型；h96/h192 只放大第二层 MLP 隐层
    table = {
        "h48":  ("b", 24, 96, 48, 8),
        "h96":  ("b", 24, 96, 96, 8),
        "h192": ("b", 24, 96, 192, 8),
        "ctx48": ("b", 48, 128, 48, 8),
        "id16": ("b", 24, 96, 48, 16),
    }
    tags = [t.strip() for t in args.configs.split(",") if t.strip()]

    print("=" * 74)
    print(f"容量探针（device={device}, epochs={args.epochs}, batch={args.batch}, amp={args.amp}）")
    print("=" * 74)
    results = []
    for tag in tags:
        if tag not in table:
            print(f"  [跳过] 未知配置 {tag}")
            continue
        _, d_ctx, hid_ctx, hid_dish, id_dim = table[tag]
        r = run_config(tag, d_ctx, hid_ctx, hid_dish, id_dim,
                       scen, samples, args.epochs, args.batch, device)
        results.append(r)
        print(f"  {tag:<6} 参数 {r['n_params']:>7,}  "
              f"val_rank={r['val_rank']:.4f}  top1={r['val_top1']:.2%}  "
              f"top3={r['val_top3']:.2%}  ({r['seconds']:.0f}s)", flush=True)

    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if results:
        base = next((r for r in results if r["tag"] == "h48"), results[0])
        print()
        print(f"  基线 {base['tag']}: top1={base['val_top1']:.2%}  top3={base['val_top3']:.2%}")
        for r in results:
            if r is base:
                continue
            print(f"  {r['tag']:<6} Δtop1={r['val_top1']-base['val_top1']:+.2%}  "
                  f"Δtop3={r['val_top3']-base['val_top3']:+.2%}  "
                  f"参数×{r['n_params']/base['n_params']:.2f}")
    print(f"\n已写入 -> {out/'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
