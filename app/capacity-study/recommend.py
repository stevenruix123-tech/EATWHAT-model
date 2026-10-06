# -*- coding: utf-8 -*-
"""
recommend.py —— 用训练好的两层模型推荐「今天该吃什么」

用法示例：
    python recommend.py                                     # 跑 10 个预设场景
    python recommend.py --slot 晚餐 --hunger 4 --time 40 ^
        --mood 疲惫 --weather 冷 --companion 一个人 ^
        --health 想健康 --spicy 微辣 --budget 正常
    python recommend.py --interactive                      # 交互式逐项输入
    python recommend.py --preset 深夜加班

模型加载自 artifacts/model.npz（由 train.py 产出），不重新训练。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from features import (
    ART, CATEGORIES, CAT_TO_ID, CTX_CAT_COLS, CTX_VOCAB, DISH_CAT_IDX,
    DISH_NAMES, N_DISH,
)

DISH_TABLE = pd.read_csv(ART / "dishes.csv")   # 只读一次，用于生成推荐理由
import train as T

# 字段 -> 命令行参数名 / 中文提示
FIELD_INFO = [
    ("time_slot", "slot", "时段", CTX_VOCAB["time_slot"]),
    ("hunger", "hunger", "饥饿程度 1-5", ["1", "2", "3", "4", "5"]),
    ("available_time", "time", "可用时间(分钟)", None),
    ("budget", "budget", "预算", CTX_VOCAB["budget"]),
    ("mood", "mood", "心情", CTX_VOCAB["mood"]),
    ("weather", "weather", "天气", CTX_VOCAB["weather"]),
    ("companion", "companion", "和谁吃", CTX_VOCAB["companion"]),
    ("health_goal", "health", "健康诉求", CTX_VOCAB["health_goal"]),
    ("spicy_tol", "spicy", "吃辣能力", CTX_VOCAB["spicy_tol"]),
    ("taboo", "taboo", "忌口(可留空)", CTX_VOCAB["taboo"]),
]

PRESETS = {
    "深夜加班": dict(time_slot="夜宵", hunger=5, available_time=25, budget="正常",
                 mood="疲惫", weather="冷", companion="一个人",
                 health_goal="不关心", spicy_tol="中辣", taboo=""),
    "工作日午餐": dict(time_slot="午餐", hunger=4, available_time=40, budget="正常",
                  mood="一般", weather="晴", companion="同事",
                  health_goal="随便", spicy_tol="微辣", taboo=""),
    "周末朋友聚餐": dict(time_slot="晚餐", hunger=5, available_time=100, budget="想吃好点",
                   mood="开心", weather="晴", companion="朋友",
                   health_goal="不关心", spicy_tol="中辣", taboo=""),
    "减脂期午餐": dict(time_slot="午餐", hunger=3, available_time=30, budget="正常",
                  mood="一般", weather="热", companion="一个人",
                  health_goal="严格控制", spicy_tol="微辣", taboo=""),
    "约会晚餐": dict(time_slot="晚餐", hunger=4, available_time=90, budget="不差钱",
                 mood="开心", weather="晴", companion="约会",
                 health_goal="不关心", spicy_tol="不吃辣", taboo=""),
    "预算紧张": dict(time_slot="午餐", hunger=4, available_time=30, budget="省着点",
                 mood="低落", weather="阴", companion="一个人",
                 health_goal="不关心", spicy_tol="不吃辣", taboo=""),
    "感冒没胃口": dict(time_slot="晚餐", hunger=2, available_time=60, budget="正常",
                  mood="疲惫", weather="冷", companion="家人",
                  health_goal="想健康", spicy_tol="不吃辣", taboo=""),
    "下午犯困": dict(time_slot="下午加餐", hunger=2, available_time=15, budget="正常",
                 mood="疲惫", weather="阴", companion="一个人",
                 health_goal="想健康", spicy_tol="微辣", taboo=""),
    "早起赶时间": dict(time_slot="早餐", hunger=3, available_time=15, budget="省着点",
                  mood="一般", weather="冷", companion="一个人",
                  health_goal="不关心", spicy_tol="不吃辣", taboo=""),
    "海鲜过敏": dict(time_slot="晚餐", hunger=4, available_time=60, budget="想吃好点",
                 mood="开心", weather="雨", companion="家人",
                 health_goal="随便", spicy_tol="中辣", taboo="海鲜"),
}


# ---------------------------------------------------------------------------
def load_models():
    cm = T.CuisineNet(np.random.default_rng(T.SEED))
    dm = T.DishScorer(np.random.default_rng(T.SEED + 1))
    z = np.load(ART / "model.npz")
    cp, dp = cm.params(), dm.params()
    missing = [k for k in cp if f"cuisine.{k}" not in z]
    if missing:
        raise SystemExit(f"模型文件不匹配，缺少 {missing}；请重新运行 train.py")
    for k, v in cp.items():
        v[...] = z[f"cuisine.{k}"]
    for k, v in dp.items():
        v[...] = z[f"dish.{k}"]
    return cm, dm


def normalize_scenario(raw: dict) -> dict:
    sc = {}
    for field, _, _, vocab in FIELD_INFO:
        v = raw.get(field)
        if field == "hunger":
            sc[field] = int(v)
        elif field == "available_time":
            sc[field] = int(v)
        else:
            v = "" if v is None else str(v).strip()
            if vocab and field != "taboo" and v not in vocab:
                raise SystemExit(f"字段 {field} 的值 '{v}' 不合法，可选：{vocab}")
            sc[field] = v
    return sc


def why(dish_row, sc: dict) -> str:
    """用「菜品属性 vs 情境」对比生成一句人话解释（取自数据集的客观属性，
    不是模型内部权重——目的是让人能一眼判断推荐合不合理）。"""
    d = dish_row
    bits = []
    if d.time_cost <= 15:
        bits.append(f"快({int(d.time_cost)}分钟)")
    elif d.time_cost >= 45:
        bits.append(f"要等({int(d.time_cost)}分钟)")
    if d.temperature == 2:
        bits.append("热乎")
    elif d.temperature == 0:
        bits.append("清爽")
    if d.spicy >= 2:
        bits.append(f"辣度{d.spicy}")
    if sc["health_goal"] in ("想健康", "严格控制") and d.healthy >= 4:
        bits.append("够健康")
    if d.heavy >= 4 and sc["hunger"] >= 4:
        bits.append("顶饱")
    if d.price <= 20:
        bits.append(f"便宜({int(d.price)}元)")
    elif d.price >= 60:
        bits.append(f"偏贵({int(d.price)}元)")
    if sc["companion"] in ("朋友", "家人", "约会") and d.heavy >= 3:
        bits.append("适合一起吃")
    if sc["weather"] == "雪" and d.temperature == 2 and d.time_cost <= 25:
        bits.append("不用出门太久")
    return "、".join(bits) if bits else "综合得分最高"


def recommend(sc: dict, cm, dm, topk: int = 3, verbose: bool = True):
    row = pd.DataFrame([sc])
    top, scores, w = T.predict_row(row.iloc[0], cm, dm, topk=topk, use_filter=True)
    # 第一层的品类概率（给出「为什么是这个品类」）
    c, n = T.encode_context(row)
    logits = cm.forward(c, n)[0]
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    order = np.argsort(-p)
    cat_rank = [(CATEGORIES[i], float(p[i])) for i in order[:3]]

    if verbose:
        print("情境：" + "  ".join(
            f"{lbl}={sc[f]}" for f, _, lbl, _ in FIELD_INFO if f != "available_time"
        ) + f"  可用时间={sc['available_time']}分钟")
        print(f"第一层品类判断： " + "  ".join(f"{c}({v:.0%})" for c, v in cat_rank))
        print(f"第二层菜品排序（候选 {len(np.where(DISH_CAT_IDX[:, 0] == w)[0])} 道）：")
        for rank, (did, s) in enumerate(zip(top, scores), 1):
            d = DISH_TABLE.iloc[int(did)]
            print(f"   {rank}. {DISH_NAMES[int(did)]:<16s} [{d.category}] "
                  f"评分 {s:.4f}   {why(d, sc)}")
        print()
    return top, scores, cat_rank


def interactive(cm, dm):
    print("交互式输入（直接回车用括号里的默认值）\n")
    sc = dict(PRESETS["工作日午餐"])
    for field, _, label, vocab in FIELD_INFO:
        hint = f"{vocab}" if vocab else "数字"
        cur = sc[field]
        v = input(f"{label} {hint} [{cur}]: ").strip()
        if v:
            sc[field] = int(v) if field in ("hunger", "available_time") else v
    print()
    recommend(sc, dm=dm, cm=cm)


def build_parser():
    p = argparse.ArgumentParser(
        description="今天该吃什么 —— 两层决策模型推荐",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS), help="使用预设场景")
    p.add_argument("--all-presets", action="store_true", help="依次跑完所有预设场景")
    p.add_argument("--interactive", action="store_true", help="交互式输入")
    p.add_argument("--topk", type=int, default=3, help="推荐前几名，默认 3")
    p.add_argument("--slot", choices=CTX_VOCAB["time_slot"])
    p.add_argument("--hunger", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--time", type=int, dest="available_time", help="可用时间(分钟)")
    p.add_argument("--budget", choices=CTX_VOCAB["budget"])
    p.add_argument("--mood", choices=CTX_VOCAB["mood"])
    p.add_argument("--weather", choices=CTX_VOCAB["weather"])
    p.add_argument("--companion", choices=CTX_VOCAB["companion"])
    p.add_argument("--health", choices=CTX_VOCAB["health_goal"], dest="health_goal")
    p.add_argument("--spicy", choices=CTX_VOCAB["spicy_tol"], dest="spicy_tol")
    p.add_argument("--taboo", default="", help="忌口，可留空")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cm, dm = load_models()
    print("=" * 68)
    print("今天该吃什么 —— 两层决策模型")
    print("=" * 68 + "\n")

    if args.interactive:
        interactive(cm, dm)
        return 0

    if args.all_presets:
        for name, sc in PRESETS.items():
            print(f"### {name}")
            recommend(sc, cm, dm, topk=args.topk)
        return 0

    if args.preset:
        sc = dict(PRESETS[args.preset])
    else:
        base = dict(PRESETS["工作日午餐"])
        overrides = {k: v for k, v in vars(args).items()
                     if k in [f[0] for f in FIELD_INFO] and v is not None}
        if not overrides and not sys.stdin.isatty():
            # 无参数且非交互：展示所有预设
            for name, s in PRESETS.items():
                print(f"### {name}")
                recommend(s, cm, dm, topk=args.topk)
            return 0
        base.update(overrides)
        sc = base
    sc = normalize_scenario(sc)
    recommend(sc, cm, dm, topk=args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
