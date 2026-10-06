# -*- coding: utf-8 -*-
"""
features.py —— 特征编码（数据 -> 模型输入）

约定：
    情境（context）：类别列 -> 整数索引；数值列 -> 归一化到 0~1
    菜品（dish）   ：额外带 dish_id（模型自己学每道菜的偏好偏置）
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"

_dishes = pd.read_csv(ART / "dishes.csv")
DISHES = _dishes
N_DISH = len(_dishes)
CATEGORIES = sorted(_dishes.category.unique().tolist())
N_CAT = len(CATEGORIES)
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}
DISH_ID = _dishes.dish_id.values
DISH_NAME = _dishes.dish_name.tolist()
DISH_NAMES = DISH_NAME
DISH_CATEGORY_ID = _dishes.category.map(CAT_TO_ID).values

# ---- 列定义 ----
CTX_CAT_COLS = ["time_slot", "mood", "budget", "weather", "companion",
                "health_goal", "spicy_tol", "taboo"]
CTX_NUM_COLS = ["available_time", "hunger"]
DISH_CAT_COLS = ["category", "spicy", "heavy", "healthy", "temperature", "carb", "protein"]
DISH_NUM_COLS = ["price", "time_cost"]

# ---- 类别取值表：固定顺序，保证训练/推理一致 ----
CTX_VOCAB = {
    "time_slot":  ["早餐", "午餐", "下午加餐", "晚餐", "夜宵"],
    "mood":       ["开心", "一般", "疲惫", "压力大", "低落", "兴奋"],
    "budget":     ["省着点", "正常", "想吃好点", "不差钱"],
    "weather":    ["晴", "阴", "雨", "雪", "热", "冷"],
    "companion":  ["一个人", "同事", "朋友", "家人", "约会"],
    "health_goal": ["不关心", "随便", "想健康", "严格控制"],
    "spicy_tol":  ["不吃辣", "微辣", "中辣", "重辣"],
    # taboo 的取值必须与 artifacts/dishes.csv 里实际出现的过敏原一致。
    # 注意：「素食」是饮食约束而非过敏原，它不写在 dish.taboo 里，
    #       而是由 dishes.csv 的 is_veg 列表达（见 build_dataset.utility 第 10/11 条）。
    "taboo":      ["", "海鲜", "乳制品", "麸质", "花生", "海鲜,乳制品", "素食"],
}
DISH_VOCAB = {
    "category":    CATEGORIES,
    "spicy":       [0, 1, 2, 3],
    "heavy":       [1, 2, 3, 4, 5],
    "healthy":     [1, 2, 3, 4, 5],
    "temperature": [0, 1, 2],
    "carb":        [1, 2, 3, 4, 5],
    "protein":     [1, 2, 3, 4, 5],
}

# 数值归一化区间（来自数据集实际取值范围）
CTX_NUM_RANGE = {"available_time": (0.0, 150.0), "hunger": (1.0, 5.0)}
# 菜品库扩展到 317 道后，price 实测 4~108、time_cost 实测 3~90，
# 旧的 (0,100)/(0,70) 上界会把部分菜品归一化成 >1，这里按实测上界放宽。
DISH_NUM_RANGE = {"price": (0.0, 120.0), "time_cost": (0.0, 100.0)}


# embedding 维度上限。原值 12 是按「47 道菜 / 8 个品类」标定的：
# 那时品类内候选池只有 5.9 道菜，12 维足够区分。菜品库扩到 317 道 / 18 品类后，
# 品类内候选池涨到平均 17.6 道（最多 30 道），12 维装不下「同品类内哪道菜更好」
# 所需的区分度 —— 实测第二层验证 top-1 掉到 4.83%（≈ 随机 1/17.6=5.7%）。
# 用环境变量 EMB_DIM_CAP 覆盖，便于做容量对比实验。
EMB_DIM_CAP = int(os.environ.get("EMB_DIM_CAP", "12"))


def emb_dim(name: str, n_unique: int) -> int:
    """embedding 维度：类别少就给小维度，避免小数据过拟合。"""
    return int(min(EMB_DIM_CAP, max(3, round(1.6 * np.sqrt(n_unique)))))


def _cat_index(df: pd.DataFrame, col: str, vocab: list) -> np.ndarray:
    m = {v: i for i, v in enumerate(vocab)}
    s = df[col]
    if col == "taboo":
        s = s.fillna("").astype(str)
    return s.map(lambda x: m.get(x, 0)).to_numpy(dtype=np.int64)


def _num_norm(df: pd.DataFrame, col: str, rng: tuple) -> np.ndarray:
    lo, hi = rng
    return ((df[col].to_numpy(dtype=float) - lo) / (hi - lo)).astype(np.float32)


def encode_context(df: pd.DataFrame):
    """-> (cat_idx [B, len(CTX_CAT_COLS)], num [B, len(CTX_NUM_COLS)])"""
    cat = np.stack([_cat_index(df, c, CTX_VOCAB[c]) for c in CTX_CAT_COLS], axis=1)
    num = np.stack([_num_norm(df, c, CTX_NUM_RANGE[c]) for c in CTX_NUM_COLS], axis=1)
    return cat, num


def encode_dishes():
    """-> (cat_idx [N_DISH, len(DISH_CAT_COLS)], num [N_DISH, len(DISH_NUM_COLS)])"""
    cat = np.stack([_cat_index(_dishes, c, DISH_VOCAB[c]) for c in DISH_CAT_COLS], axis=1)
    num = np.stack([_num_norm(_dishes, c, DISH_NUM_RANGE[c]) for c in DISH_NUM_COLS], axis=1)
    return cat, num


# 预先算好，训练与推理共用，避免重复计算
DISH_CAT_IDX, DISH_NUM_FEAT = encode_dishes()
