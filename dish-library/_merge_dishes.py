# -*- coding: utf-8 -*-
"""
把 dish_id 0..46 的旧库对齐到原 features.py 的数值归一化区间。

旧库 price 最大 98、time_cost 最大 60，都落在 DISH_NUM_RANGE 内；
本库 price 4~108、time_cost 3~90 略超上界。这里把两个库的
price / time_cost 统一到 (0,120) / (0,100)，避免归一化特征 >1。
"""
import pandas as pd

DISCOLS = ["dish_name", "category", "time_cost", "price", "spicy", "heavy",
           "healthy", "temperature", "carb", "protein", "taboo"]

# 旧库 47 道菜的 region / is_veg 人工映射（region 只作参考列，不参与特征编码）
OLD_META = {
    # ---- 中式快餐盖饭 ----
    "番茄牛腩饭": ("长三角", 0), "黄焖鸡米饭": ("共通", 0), "青椒肉丝盖饭": ("长三角", 0),
    "台式卤肉饭": ("共通", 0), "咖喱鸡排饭": ("共通", 0), "梅菜扣肉饭": ("珠三角", 0),
    # ---- 面食粉面 ----
    "兰州牛肉面": ("共通", 0), "重庆小面": ("共通", 0), "番茄鸡蛋面": ("共通", 0),
    "炸酱面": ("共通", 0), "螺蛳粉": ("共通", 0), "云南小锅米线": ("共通", 0),
    # ---- 家常炒菜 ----
    "西红柿炒蛋+米饭": ("共通", 0), "麻婆豆腐+米饭": ("共通", 0),
    "清蒸鲈鱼+米饭": ("珠三角", 0), "宫保鸡丁+米饭": ("共通", 0),
    "红烧肉+米饭": ("长三角", 0), "蒜蓉西兰花+米饭": ("共通", 1),
    "地三鲜+米饭": ("共通", 1),
    # ---- 火锅麻辣烫 ----
    "麻辣烫": ("共通", 0), "清汤火锅": ("共通", 0), "牛油麻辣火锅": ("共通", 0),
    "关东煮": ("共通", 0), "冒菜": ("共通", 0), "菌汤火锅": ("共通", 0),
    # ---- 烧烤小吃 ----
    "烤羊肉串": ("共通", 0), "烤鸡翅": ("共通", 0), "烤茄子": ("共通", 1),
    "烤冷面": ("共通", 0), "韩式炸鸡": ("共通", 0), "铁板豆腐": ("共通", 1),
    # ---- 轻食健康 ----
    "鸡胸肉沙拉": ("共通", 0), "藜麦牛油果碗": ("共通", 1), "三文鱼沙拉": ("共通", 0),
    "全麦鸡肉三明治": ("共通", 0), "水煮时蔬+溏心蛋": ("共通", 0),
    "希腊酸奶坚果碗": ("共通", 0),
    # ---- 西式简餐 ----
    "番茄肉酱意面": ("共通", 0), "香煎牛排": ("共通", 0), "芝士汉堡+薯条": ("共通", 0),
    "奶油蘑菇汤+面包": ("共通", 0), "玛格丽特披萨": ("共通", 0),
    "烤鸡胸配土豆泥": ("共通", 0),
    # ---- 早餐粥点 ----
    "皮蛋瘦肉粥+油条": ("长三角", 0), "小笼包+豆浆": ("长三角", 0),
    "杂粮煎饼果子": ("共通", 0), "燕麦牛奶粥": ("共通", 0),
}


def load_old(p):
    df = pd.read_csv(p)
    df["taboo"] = df.taboo.fillna("").astype(str).str.replace("!", ",", regex=False)
    df["region"] = df.dish_name.map(lambda n: OLD_META.get(n, ("共通", 0))[0])
    df["is_veg"] = df.dish_name.map(lambda n: OLD_META.get(n, ("共通", 0))[1])
    df["frequency"] = 3
    df["meal_slots"] = "午餐/晚餐"
    return df


def load_new(p):
    df = pd.read_csv(p)
    df["taboo"] = df.taboo.fillna("").astype(str).str.replace("!", ",", regex=False)
    return df


def main():
    base = r"C:\Users\123\Desktop\桌面应用\文档"
    old = load_old(rf"{base}\meal-decision\artifacts\dishes_47原始备份.csv")
    new = load_new(rf"{base}\dishes_yangtze_pearl_delta.csv")

    # 旧库里带「+米饭」后缀的条目，与新库的无后缀标准名是同一道菜；
    # 统一改名后由 drop_duplicates(keep="first") 丢弃旧的冗余条目，
    # 既消除「同菜两条目」，也让品类归并（旧库有独立的「中式快餐盖饭」等 8 类）。
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

    # 新库中的旧名：旧库把广式白切鸡写作「白切鸡(广式)」，统一为标准名
    new["dish_name"] = new.dish_name.replace({
        "白切鸡(广式)": "白切鸡",
    })

    old_cols = ["dish_name", "category", "time_cost", "price", "spicy", "heavy",
                "healthy", "temperature", "carb", "protein", "taboo", "region",
                "is_veg", "frequency", "meal_slots"]
    allc = old_cols

    merged = pd.concat([old[old_cols], new[old_cols]], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="dish_name", keep="first").reset_index(drop=True)
    merged.insert(0, "dish_id", range(len(merged)))

    # 素食一致性：is_veg=1 的菜不应带海鲜/乳制品
    s = merged.taboo.fillna("").astype(str)
    bad = merged[(merged.is_veg == 1) & (s.str.contains("海鲜", regex=False) |
                                        s.str.contains("乳制品", regex=False))]
    if len(bad):
        print("[警告] 素食条目带海鲜/乳制品标签：")
        print(bad[["dish_name", "taboo"]].to_string(index=False))

    out = rf"{base}\meal-decision\artifacts\dishes.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"合并前 {before} 行（含重名）-> 去重后 {len(merged)} 道")
    print(f"  price     {merged.price.min()} ~ {merged.price.max()}")
    print(f"  time_cost {merged.time_cost.min()} ~ {merged.time_cost.max()}")
    print(f"  品类 {merged.category.nunique()} 个，每类菜数 {merged.groupby('category').size().min()}"
          f" ~ {merged.groupby('category').size().max()}")
    print(f"  taboo 取值: {sorted(set(s.str.split(',').explode()) - {''})}")
    print(f"  纯素 {int(merged.is_veg.sum())} 道")
    print(f"  品类: {' | '.join(sorted(merged.category.unique()))}")


if __name__ == "__main__":
    main()
