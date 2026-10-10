# -*- coding: utf-8 -*-
"""
合并两个菜品库，产出最终的 dishes.csv：

  1. 模型原有的 47 道菜（app/artifacts/dishes_47原始备份.csv）
     —— 保留它，因为它带着当初为建模选定的品类与属性，且模型已在其上训练过
  2. 扩充后的中国菜库（dish-library/dishes_cn3_regions.csv，808 道）
     —— 长三角/珠三角 448 + 本次新增 湘/东北/川 364，去重后 808

合并口径：
  - 菜名去重，**原有 47 道优先**（保留其手工调过的属性与旧品类归属）
  - taboo 分隔符统一为逗号（旧库是逗号，新库是 "!"），与 features.py 一致
  - dish_id 重新连续编号
  - 输出前做完整性自检

用法：
    python merge_final.py
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"
sys_path = str(HERE)
import sys
sys.path.insert(0, sys_path)

COLS = ["dish_name", "category", "time_cost", "price", "spicy", "heavy",
        "healthy", "temperature", "carb", "protein", "taboo"]

# 旧库 47 道菜的 region / is_veg 人工映射（region 仅作参考列，不参与特征编码）
OLD_META = {
    "番茄牛腩饭": ("长三角", 0), "黄焖鸡米饭": ("共通", 0), "青椒肉丝盖饭": ("长三角", 0),
    "台式卤肉饭": ("共通", 0), "咖喱鸡排饭": ("共通", 0), "梅菜扣肉饭": ("珠三角", 0),
    "兰州牛肉面": ("共通", 0), "重庆小面": ("共通", 0), "番茄鸡蛋面": ("共通", 0),
    "炸酱面": ("共通", 0), "螺蛳粉": ("共通", 0), "云南小锅米线": ("共通", 0),
    "西红柿炒蛋+米饭": ("共通", 0), "麻婆豆腐+米饭": ("共通", 0),
    "清蒸鲈鱼+米饭": ("珠三角", 0), "宫保鸡丁+米饭": ("共通", 0),
    "红烧肉+米饭": ("长三角", 0), "蒜蓉西兰花+米饭": ("共通", 1),
    "地三鲜+米饭": ("共通", 1),
    "麻辣烫": ("共通", 0), "清汤火锅": ("共通", 0), "牛油麻辣火锅": ("共通", 0),
    "关东煮": ("共通", 0), "冒菜": ("共通", 0), "菌汤火锅": ("共通", 0),
    "烤羊肉串": ("共通", 0), "烤鸡翅": ("共通", 0), "烤茄子": ("共通", 1),
    "烤冷面": ("共通", 0), "韩式炸鸡": ("共通", 0), "铁板豆腐": ("共通", 1),
    "鸡胸肉沙拉": ("共通", 0), "藜麦牛油果碗": ("共通", 1), "三文鱼沙拉": ("共通", 0),
    "全麦鸡肉三明治": ("共通", 0), "水煮时蔬+溏心蛋": ("共通", 0),
    "希腊酸奶坚果碗": ("共通", 0),
    "番茄肉酱意面": ("共通", 0), "香煎牛排": ("共通", 0), "芝士汉堡+薯条": ("共通", 0),
    "奶油蘑菇汤+面包": ("共通", 0), "玛格丽特披萨": ("共通", 0),
    "烤鸡胸配土豆泥": ("共通", 0),
    "皮蛋瘦肉粥+油条": ("长三角", 0), "小笼包+豆浆": ("长三角", 0),
    "杂粮煎饼果子": ("共通", 0), "燕麦牛奶粥": ("共通", 0),
}


def load_old() -> pd.DataFrame:
    p = APP / "artifacts" / "dishes_47原始备份.csv"
    if not p.exists():
        raise SystemExit(f"找不到旧库备份：{p}")
    df = pd.read_csv(p)
    df["taboo"] = df.taboo.fillna("").astype(str).str.replace("!", ",", regex=False)
    df["region"] = df.dish_name.map(lambda n: OLD_META.get(n, ("共通", 0))[0])
    df["is_veg"] = df.dish_name.map(lambda n: OLD_META.get(n, ("共通", 0))[1])
    df["frequency"] = 3
    df["meal_slots"] = "午餐/晚餐"
    return df


def load_new() -> pd.DataFrame:
    p = HERE / "dishes_cn3_regions.csv"
    if not p.exists():
        raise SystemExit(f"找不到扩充库：{p}\n请先运行 python _gen_dishes_cn3.py")
    df = pd.read_csv(p)
    df["taboo"] = df.taboo.fillna("").astype(str).str.replace("!", ",", regex=False)
    return df


def main() -> pd.DataFrame:
    old = load_old()
    new = load_new()

    # ---- 1) 旧库里的「同菜异名」归一 -------------------------------
    # 旧库把同一道菜写成「X+米饭」「X(快餐)」，与扩充库的标准名重复。
    # 统一改名后由下面的去重丢弃旧的冗余条目（旧库优先保留属性）。
    old["dish_name"] = old.dish_name.replace({
        "西红柿炒蛋+米饭": "番茄炒蛋",
        "麻婆豆腐+米饭": "麻婆豆腐",
        "清蒸鲈鱼+米饭": "清蒸鲈鱼",
        "宫保鸡丁+米饭": "宫保鸡丁",
        "红烧肉+米饭": "上海红烧肉",
        "蒜蓉西兰花+米饭": "蒜蓉西兰花",
        "地三鲜+米饭": "地三鲜",
        "皮蛋瘦肉粥+油条": "皮蛋瘦肉粥",
        "青椒肉丝盖饭": "青椒肉丝",
    })
    new["dish_name"] = new.dish_name.replace({
        "隆江猪脚饭(快餐)": "隆江猪脚饭",
    })

    # ---- 2) 旧库的窄品类并入同类大品类 -----------------------------
    # 旧库只有 8 个品类，其中两个（各 6 道）与扩充库的品类是同一概念，
    # 不合并会把同类菜拆成两处、并让这两个品类只有 6 道候选（第二层在品类内排序）。
    cat_map = {
        "中式快餐盖饭": "煲仔饭碟头饭",   # 盖饭/碟头饭本就是一类
        "面食粉面": "长三角粉面",         # 兰州拉面/螺蛳粉/米线等归入面食
    }
    old["category"] = old.category.replace(cat_map)
    new["category"] = new.category.replace(cat_map)

    extra = ["region", "is_veg", "frequency", "meal_slots"]
    merged = pd.concat([old[COLS + extra], new[COLS + extra]], ignore_index=True)

    n_old, n_new = len(old), len(new)

    merged = merged.drop_duplicates("dish_name", keep="first").reset_index(drop=True)
    merged.insert(0, "dish_id", range(len(merged)))

    # 检查：新库里有多少菜是旧库没有的（净增），以及同名的有多少
    old_names = set(old.dish_name)
    new_names = set(new.dish_name)
    both = sorted(old_names & new_names)
    net_new = len(new_names - old_names)

    s = merged.taboo.fillna("").astype(str)
    bad = []
    if merged.dish_name.duplicated().any():
        bad.append("重名")
    for c, lo, hi in [("spicy", 0, 3), ("heavy", 1, 5), ("healthy", 1, 5),
                      ("temperature", 0, 2), ("carb", 1, 5), ("protein", 1, 5),
                      ("is_veg", 0, 1), ("frequency", 1, 3)]:
        if not merged[c].between(lo, hi).all():
            bad.append(f"{c} 越界")
    leak = merged[(merged.is_veg == 1) & (s.str.contains("海鲜") | s.str.contains("乳制品"))]
    if len(leak):
        bad.append(f"素食带荤标签: {leak.dish_name.tolist()[:5]}")
    # taboo 词表必须都是 features.py 认识的
    toks = sorted(set(s.str.split(",").explode()) - {""})
    allowed = {"海鲜", "乳制品", "麸质", "花生"}
    unknown = [t for t in toks if t not in allowed]
    if unknown:
        bad.append(f"taboo 出现未知取值: {unknown}")

    out = APP / "artifacts" / "dishes.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"旧库(模型原有 47 道)   : {n_old} 道  -> 归一化+并入大品类")
    print(f"扩充库(中国菜)         : {n_new} 道")
    print(f"两库同名(旧库优先保留)  : {len(both)} 道")
    print(f"扩充库净新增           : {net_new} 道")
    print(f"合并去重后合计          : {len(merged)} 道 / {merged.category.nunique()} 品类")
    print("=" * 72)
    print()
    print(merged.groupby("category").size().sort_values(ascending=False).to_string())
    print()
    print("按 region：")
    print(merged.region.value_counts().to_string())
    print()
    print("按 frequency：", merged.frequency.value_counts().sort_index().to_dict())
    print("纯素：", int(merged.is_veg.sum()), "道")
    print("taboo 取值：", toks)
    print()
    print(f"自检: {'通过' if not bad else '失败 -> ' + '; '.join(bad)}")
    print(f"已写入 -> {out}")
    return merged


if __name__ == "__main__":
    main()
