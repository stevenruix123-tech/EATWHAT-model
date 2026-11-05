# -*- coding: utf-8 -*-
"""
eval_torch.py —— 用与 evaluate.py 完全相同的口径评估 PyTorch 模型

为什么需要单独写：evaluate.py 依赖 numpy 版 train.py 的模型类与
predict_row()。要跟 numpy 版做同口径对比，必须复刻它的三件事：
  1. 第一层用 CuisineNet 取 argmax 品类
  2. 第二层在**预测出的品类内**打分，取 top-1 / top-3
  3. 「上帝视角」用 scenario_utility.npy 里的真实效用（不是重新推）
以及同样的基线：随机 / 全库最热门 / 按时段最热门。

用法：
    python eval_torch.py --model artifacts_torch/model_torch.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features as feat
import train as NP
import train_torch as TT
from features import ART, DISH_CAT_IDX, DISH_NAMES, N_DISH, N_CAT, CAT_TO_ID


@torch.no_grad()
def predict_all(cm, dm, df, device):
    """返回每个情境的：预测品类、品类内 top-3 菜品 id。"""
    c, n = feat.encode_context(df)
    ck = torch.as_tensor(np.ascontiguousarray(c), device=device)
    nk = torch.as_tensor(np.array(n, dtype=np.float32, copy=True), device=device)
    win = cm(ck, nk).argmax(1).cpu().numpy()

    ids, dc, dn = TT.dish_batch_tensors(device)
    scores = dm(ck, nk, ids, dc, dn).float().cpu().numpy()      # [N, K]
    dish_cat = DISH_CAT_IDX[:, 0]
    tops = np.zeros((len(df), 3), dtype=int)
    for i in range(len(df)):
        cand = np.where(dish_cat == win[i])[0]
        if len(cand) == 0:
            cand = np.arange(N_DISH)
        order = cand[np.argsort(-scores[i, cand])][:3]
        tops[i, :len(order)] = order
    return win, tops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts_torch/model_torch.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_s)

    ckpt = torch.load(HERE / args.model, map_location=device, weights_only=False)
    # 从 checkpoint 的 meta 还原结构（容量探针会改 hid_dish / d_ctx / id_dim）
    m = ckpt.get("meta", {})
    TT.DishScorer.__init__.__defaults__ = (
        int(m.get("hid_dish", TT.HID_DISH)),
        int(m.get("d_ctx", TT.D_CTX)),
        int(m.get("id_dim", TT.ID_DIM)),
    )
    TT.CuisineNet.__init__.__defaults__ = (
        int(m.get("hid_ctx", TT.HID_CTX)), int(m.get("d_ctx", TT.D_CTX)))
    cm = TT.CuisineNet().to(device); cm.load_state_dict(ckpt["cuisine"]); cm.eval()
    dm = TT.DishScorer().to(device); dm.load_state_dict(ckpt["dish"]); dm.eval()

    samples = pd.read_csv(ART / "samples.csv")
    samples["taboo"] = samples["taboo"].fillna("").astype(str)
    scen = pd.read_csv(ART / "scenarios.csv")
    te = scen[scen.split == "test"].reset_index(drop=True)
    n = len(te)

    # ---- 上帝视角：真实效用矩阵（与 build_dataset 同源）----
    U = np.load(ART / "scenario_utility.npy")
    sid = te.scenario_id.values.astype(int)
    Ute = U[sid]                                     # [N, N_DISH]
    gold = Ute.argmax(1)
    gold_cat = DISH_CAT_IDX[gold, 0]

    win, tops = predict_all(cm, dm, te, device)

    # ---- 四个指标（口径同 evaluate.py）----
    u_best = Ute[np.arange(n), gold]
    u_model = Ute[np.arange(n), tops[:, 0]]
    top1_hit = float((tops[:, 0] == gold).mean())
    top3_hit = float((tops[:, :3] == gold[:, None]).any(1).mean())
    cat_hit = float((win == gold_cat).mean())
    regret = float((u_best - u_model).mean())

    # ---- 基线 ----
    rng = np.random.default_rng(0)
    u_rand = float(np.mean([Ute[i][rng.integers(N_DISH)] for i in range(n)]))
    # 全库最热门（训练集里被选次数最多的菜）
    tr = samples[samples.split == "train"]
    pop = int(tr.dish_id.value_counts().idxmax())
    u_pop = float(Ute[:, pop].mean())
    # 按时段最热门
    slot_pop, u_slot = {}, 0.0
    for slot, g in tr.groupby("time_slot"):
        slot_pop[slot] = int(g.dish_id.value_counts().idxmax())
    for i, row in te.iterrows():
        u_slot += Ute[i][slot_pop.get(row.time_slot, pop)]
    u_slot /= n

    res = {
        "n_test_scenarios": n,
        "mean_utility_model": round(float(u_model.mean()), 4),
        "mean_utility_best": round(float(u_best.mean()), 4),
        "relative_to_optimum": round(float(u_model.mean() / u_best.mean()), 4),
        "mean_regret": round(regret, 4),
        "category_hit": round(cat_hit, 4),
        "top1_hit": round(top1_hit, 4),
        "top3_hit": round(top3_hit, 4),
        "baselines": {
            "random": round(u_rand, 4),
            "most_popular": round(u_pop, 4),
            "most_popular_dish": DISH_NAMES[pop],
            "per_slot_popular": round(float(u_slot), 4),
        },
    }

    print("=" * 66)
    print(f"PyTorch 模型同口径评估（测试集 {n} 个情境）")
    print("=" * 66)
    print(f"  平均效用        : {res['mean_utility_model']}   （理论最优 {res['mean_utility_best']}，"
          f"达成 {res['relative_to_optimum']:.1%}）")
    print(f"  平均后悔值      : {res['mean_regret']}")
    print(f"  品类命中率      : {res['category_hit']:.2%}")
    print(f"  top-1 命中全库最优 : {res['top1_hit']:.2%}")
    print(f"  top-3 命中全库最优 : {res['top3_hit']:.2%}")
    print(f"  基线：随机 {res['baselines']['random']}  "
          f"最热门 {res['baselines']['most_popular']}（{res['baselines']['most_popular_dish']}）  "
          f"按时段热门 {res['baselines']['per_slot_popular']}")
    print(f"  比最热门菜基线高 : "
          f"{(res['mean_utility_model']/res['baselines']['most_popular']-1):+.1%}")

    out = Path(args.out) if args.out else (HERE / "artifacts_torch" / "eval_torch.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
