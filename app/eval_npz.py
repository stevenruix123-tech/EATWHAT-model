# -*- coding: utf-8 -*-
"""
eval_npz.py —— 用 numpy 线上权重（model.npz）在指定留出集上评估

为什么需要它：`evaluate.py` 把测试集绑在 `features.ART` 上，所以它只能评估
「和这份数据集同一目录」的模型。可要回答「旧数据练的模型 vs 新数据练的模型
谁更好」，必须让两个模型打**同一套题** —— 否则比较的是两份不同试卷的分数
（交接文档 §8 注意事项 2 说的就是这个坑）。

本脚本是 `eval_torch.py --eval-art` 的 numpy 对应物：
    模型来自 --model（任意路径的 .npz），题目来自 --eval-art（任意数据集目录）。
口径与 sweep_seeds.eval_test 完全一致。

用法：
    python eval_npz.py --model artifacts_orig/model.npz --eval-art artifacts_eval
    python eval_npz.py --model artifacts_v2/model.npz   --eval-art artifacts_eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as feat
from features import ART, DISH_CAT_IDX, N_DISH
import train as T


def load_npz(path: Path):
    """按 train.py 的结构还原模型并把 npz 权重灌进去。

    含 `cuisine.*` 键 = 两层模型；不含 = 单阶段（整目录 softmax，§7.4）。
    """
    dm = T.DishScorer(np.random.default_rng(T.SEED + 1))
    z = np.load(path)

    cm = None
    if any(k.startswith("cuisine.") for k in z.files):
        cm = T.CuisineNet(np.random.default_rng(T.SEED))
        missing = [k for k in cm.params() if f"cuisine.{k}" not in z]
        if missing:
            raise SystemExit(f"{path} 与当前结构不匹配，缺少 {missing[:6]}…")
        for k, v in cm.params().items():
            v[...] = z[f"cuisine.{k}"]

    missing = [k for k in dm.params() if f"dish.{k}" not in z]
    if missing:
        raise SystemExit(f"{path} 与当前结构不匹配，缺少 {missing[:6]}…")
    for k, v in dm.params().items():
        v[...] = z[f"dish.{k}"]
    return cm, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.npz 路径")
    ap.add_argument("--eval-art", default=None,
                    help="留出集目录（默认用 ART 的 test split）")
    ap.add_argument("--label", default=None, help="结果里显示的名字")
    args = ap.parse_args()

    model_path = Path(args.model)
    cm, dm = load_npz(model_path)

    if args.eval_art:
        ev_dir = Path(args.eval_art)
        ev = pd.read_csv(ev_dir / "scenarios.csv")
        ev["taboo"] = ev["taboo"].fillna("").astype(str)
        ev = ev.reset_index(drop=True)
        U = np.load(ev_dir / "scenario_utility.npy")
        Ute = U[ev.scenario_id.values.astype(int)]
        tag = f"{ev_dir.name}（全部 {len(ev)} 个情境）"
    else:
        sc = pd.read_csv(ART / "scenarios.csv")
        sc["taboo"] = sc["taboo"].fillna("").astype(str)
        ev = sc[sc.split == "test"].reset_index(drop=True)
        U = np.load(ART / "scenario_utility.npy")
        Ute = U[ev.scenario_id.values.astype(int)]
        tag = f"{ART.name}/test（{len(ev)} 个情境）"

    c, n = feat.encode_context(ev)
    # 两层模型：先按第一层选品类；单阶段（cm is None）：直接在全部菜里排序
    win = None if cm is None else cm.forward(c, n).argmax(1)

    dish_cat = DISH_CAT_IDX[:, 0]
    tops = np.zeros((len(ev), 3), dtype=int)
    for i in range(len(ev)):
        if win is None:
            cand = np.arange(N_DISH)
        else:
            cand = np.where(dish_cat == win[i])[0]
            if len(cand) == 0:
                cand = np.arange(N_DISH)
        ids, dc, dn = T.tile_dishes(cand[None, :], 1)
        s = dm.forward(c[i:i + 1], n[i:i + 1], ids, dc, dn)[0]
        tops[i, :3] = cand[np.argsort(-s, kind="stable")][:3]

    gold = Ute.argmax(1)
    gold_cat = dish_cat[gold]
    # 品类命中一律按「最终选中那道菜的品类」算，两种结构口径才一致
    pred_cat = dish_cat[tops[:, 0]]
    idx = np.arange(len(ev))
    u_model = float(Ute[idx, tops[:, 0]].mean())
    u_best = float(Ute[idx, gold].mean())

    name = args.label or f"{model_path.parent.name}/{model_path.name}"
    print(f"[{name}]  @ {tag}")
    print(f"  平均效用  {u_model:.4f}   (理论最优 {u_best:.4f}，达 {u_model / u_best:.1%})")
    print(f"  后悔值    {u_best - u_model:.4f}")
    print(f"  top-1     {float((tops[:, 0] == gold).mean()):.2%}")
    print(f"  top-3     {float((tops[:, :3] == gold[:, None]).any(1).mean()):.2%}")
    print(f"  品类命中  {float((pred_cat == gold_cat).mean()):.2%}")


if __name__ == "__main__":
    main()
