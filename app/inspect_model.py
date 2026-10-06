# -*- coding: utf-8 -*-
"""
inspect_model.py —— 模型诊断：弄清指标背后的真实情况

关心的问题：
  1. 品类的「天花板」是多少？（同一时段最优品类本来就集中，8 分类准确率不可比）
  2. 给定选对品类，菜品的 top-1 命中率是多少？（这才是第二层的真实能力）
  3. 后悔值 0.0624 里，有多少来自「品类选错」，多少来自「品类内选错菜品」？
  4. 推荐菜的绝对效用是否明显高于随机/热门基线？
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import ART, CATEGORIES, CAT_TO_ID, CTX_CAT_COLS, DISH_CAT_IDX, DISH_NAMES, N_DISH
import train as T


def main():
    samples = pd.read_csv(ART / "samples.csv")
    scenarios = pd.read_csv(ART / "scenarios.csv")
    for df in (samples, scenarios):
        df["taboo"] = df["taboo"].fillna("").astype(str)

    name2id = {n: i for i, n in enumerate(DISH_NAMES)}
    ev = scenarios[scenarios.split == "test"].reset_index(drop=True)
    gold_ids = ev.gold_dish.map(name2id).values

    # 为每个测试情境算真实效用向量（上帝视角）
    U = np.stack([T.utility_vector(r) for _, r in ev.iterrows()])            # [N, N_DISH]
    gold_best_all = U.argmax(1)                                              # 全库最优
    gold_cat_best = np.array([U[i][DISH_CAT_IDX[:, 0] == DISH_CAT_IDX[g, 0]].argmax()
                              for i, g in enumerate(gold_best_all)])
    # 品类内最优菜品 id
    cat_best_ids = []
    for i, g in enumerate(gold_best_all):
        c = DISH_CAT_IDX[g, 0]
        cand = np.where(DISH_CAT_IDX[:, 0] == c)[0]
        cat_best_ids.append(cand[U[i][cand].argmax()])
    cat_best_ids = np.array(cat_best_ids)

    print("=" * 74)
    print("诊断报告（测试集 n=%d）" % len(ev))
    print("=" * 74)

    # ---- 1. 品类基线：每个时段最常见的最优品类 ----
    tr = scenarios[scenarios.split == "train"]
    slot_majority = tr.groupby("time_slot").gold_category.agg(lambda s: s.mode().iat[0]).to_dict()
    maj_acc = np.mean([slot_majority[r.time_slot] == r.gold_category for _, r in ev.iterrows()])

    # ---- 2. 给定品类正确时的菜品命中率 ----
    cm = T.CuisineNet(np.random.default_rng(T.SEED))
    dm = T.DishScorer(np.random.default_rng(T.SEED + 1))
    z = np.load(ART / "model.npz")
    cm_params, dm_params = cm.params(), dm.params()
    for k in cm_params:
        cm_params[k][...] = z[f"cuisine.{k}"]
    for k in dm_params:
        dm_params[k][...] = z[f"dish.{k}"]

    rows = []
    for i, r in ev.iterrows():
        top, sc, w = T.predict_row(r, cm, dm, topk=3, use_filter=True)
        pred_cat = int(DISH_CAT_IDX[top[0], 0])
        gold_cat = int(CAT_TO_ID[r.gold_category])
        rows.append({
            "cat_ok": pred_cat == gold_cat,
            "top1": top[0], "top3": list(top),
            "hit_best_all": top[0] == gold_best_all[i],
            "hit_best_in_cat": top[0] == cat_best_ids[i],
            "best_in_cat_in_top3": cat_best_ids[i] in top,
            "u_model": U[i][top[0]],
            "u_best_all": U[i][gold_best_all[i]],
            "u_best_same_cat": U[i][cat_best_ids[i]],
            "cat": pred_cat,
        })
    D = pd.DataFrame(rows)

    print(f"\n[1] 品类预测的天花板")
    print(f"    测试集上「按时段直接猜最常见品类」的准确率 : {maj_acc:.2%}")
    print(f"    模型的品类准确率（第一层）                 : {D.cat_ok.mean():.2%}")
    print(f"    -> 时段本身已决定一部分答案，8 分类准确率要跟这个基线比")
    print(f"    各时段最常见品类: {slot_majority}")

    print(f"\n[2] 菜品选择的真实能力")
    print(f"    品类选对的比例                     : {D.cat_ok.mean():.2%}")
    print(f"    整体 top-1 命中「全库最优」        : {D.hit_best_all.mean():.2%}")
    print(f"    ** 品类选对时 ** top-1 命中品类内最优 : "
          f"{D.loc[D.cat_ok, 'hit_best_in_cat'].mean():.2%}")
    print(f"    ** 品类选对时 ** 品类内最优进 top-3   : "
          f"{D.loc[D.cat_ok, 'best_in_cat_in_top3'].mean():.2%}")

    print(f"\n[3] 后悔值分解（越低越好）")
    r_cat = (D.u_best_all - D.u_best_same_cat).mean()
    r_dish = (D.u_best_same_cat - D.u_model).mean()
    print(f"    总后悔值                          : {(D.u_best_all - D.u_model).mean():.4f}")
    print(f"      其中「品类选错」贡献            : {r_cat:.4f}  ({r_cat / (r_cat + r_dish):.0%})")
    print(f"      其中「品类内选错菜品」贡献      : {r_dish:.4f}  ({r_dish / (r_cat + r_dish):.0%})")

    print(f"\n[4] 推荐质量对比")
    rng = np.random.default_rng(0)
    rand_u = np.mean([U[i][rng.integers(N_DISH)] for i in range(len(ev))])
    pop = pd.read_csv(ART / "samples.csv").query("split == 'train'").dish_id.value_counts().idxmax()
    pop_u = np.mean([U[i][pop] for i in range(len(ev))])
    print(f"    模型推荐的平均效用 : {D.u_model.mean():.4f}")
    print(f"    随机选菜的平均效用 : {rand_u:.4f}")
    print(f"    永远选最热门菜     : {pop_u:.4f}  ({DISH_NAMES[pop]})")
    print(f"    理论最优（上帝）   : {D.u_best_all.mean():.4f}")
    print(f"    -> 模型达到最优的 {D.u_model.mean() / D.u_best_all.mean():.1%}")

    print(f"\n[5] 品类内最优菜品的效用差距（说明第二层的难度）")
    rng2 = np.random.default_rng(1)
    top2gap = []
    for i in range(len(ev)):
        c = DISH_CAT_IDX[cat_best_ids[i], 0]
        u = np.sort(U[i][DISH_CAT_IDX[:, 0] == c])
        top2gap.append(u[-1] - u[-2])
    print(f"    品类内 最优 与 次优 的平均效用差 : {np.mean(top2gap):.4f}")
    print(f"    -> 品类内菜品彼此高度相似：选错也几乎不掉分，"
          f"所以 top-1 精确命中天然偏低，应看效用/后悔值")

    print(f"\n[6] 更强基线：按时段选该时段最热门的菜")
    slot_pop = (samples.query("split == 'train'")
                .groupby("time_slot").dish_id.agg(lambda s: s.mode().iat[0]).to_dict())
    sp_u = np.mean([U[i][slot_pop[r.time_slot]] for i, r in ev.iterrows()])
    print(f"    按时段热门菜的平均效用 : {sp_u:.4f}   (模型 {D.u_model.mean():.4f})")
    print(f"    模型相对提升           : {D.u_model.mean() - sp_u:+.4f}"
          f"  ({(D.u_model.mean() / sp_u - 1):+.1%})")
    return D, U


if __name__ == "__main__":
    main()
