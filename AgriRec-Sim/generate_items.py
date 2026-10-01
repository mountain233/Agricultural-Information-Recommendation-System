# -*- coding: utf-8 -*-
"""
生成农产品数据
"""
import csv
import random
import numpy as np
from config import (NUM_ITEMS, ITEMS_FILE, RANDOM_SEED,
                    ITEM_CATEGORIES, ITEM_CATEGORY_WEIGHTS, PROVINCES)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 各品类的具体农产品名称库
ITEM_NAMES = {
    "水果": [
        "红富士苹果", "烟台苹果", "赣南脐橙", "砂糖橘", "沃柑", "丑橘",
        "猕猴桃", "红心猕猴桃", "芒果", "贵妃芒", "小台芒", "石榴",
        "软籽石榴", "草莓", "红颜草莓", "樱桃", "大樱桃", "车厘子",
        "葡萄", "阳光玫瑰", "巨峰葡萄", "西瓜", "麒麟瓜", "哈密瓜",
        "香瓜", "火龙果", "红心火龙果", "蓝莓", "百香果", "柠檬",
        "黄桃", "水蜜桃", "油桃", "李子", "杏子", "梨", "皇冠梨",
        "雪梨", "柚子", "红心柚", "香蕉", "菠萝", "凤梨", "荔枝",
        "龙眼", "杨梅", "枇杷", "青提", "红提",
    ],
    "蔬菜": [
        "大白菜", "小白菜", "娃娃菜", "菠菜", "生菜", "油麦菜",
        "芹菜", "韭菜", "香菜", "番茄", "西红柿", "圣女果",
        "黄瓜", "水果黄瓜", "丝瓜", "苦瓜", "冬瓜", "南瓜",
        "茄子", "紫茄子", "辣椒", "螺丝椒", "小米辣", "青椒",
        "彩椒", "土豆", "马铃薯", "红薯", "紫薯", "山药",
        "莲藕", "茭白", "竹笋", "芦笋", "西兰花", "花菜",
        "胡萝卜", "白萝卜", "青萝卜", "洋葱", "大蒜", "生姜",
        "大葱", "小葱", "豆角", "四季豆", "豌豆", "荷兰豆",
    ],
    "禽肉蛋品": [
        "土鸡蛋", "洋鸡蛋", "乌鸡蛋", "鹌鹑蛋", "鸭蛋", "鹅蛋",
        "鸡肉", "土鸡", "老母鸡", "三黄鸡", "乌鸡", "鸭肉",
        "老鸭", "樱桃谷鸭", "鹅肉", "猪肉", "五花肉", "里脊肉",
        "排骨", "猪蹄", "牛肉", "牛腱子", "牛腩", "羊肉",
        "羊排", "羊腿", "兔肉", "鸽子肉", "鹌鹑肉",
    ],
    "粮油": [
        "五常大米", "东北大米", "泰国香米", "籼米", "糯米",
        "小米", "黄小米", "黑米", "紫米", "面粉", "高筋面粉",
        "低筋面粉", "全麦面粉", "玉米面", "荞麦面", "菜籽油",
        "花生油", "大豆油", "玉米油", "橄榄油", "芝麻油",
        "葵花籽油", "亚麻籽油", "核桃油", "玉米", "甜玉米",
        "糯玉米", "红薯粉", "土豆粉", "绿豆", "红豆", "黄豆",
    ],
    "农资": [
        "尿素", "复合肥", "磷酸二铵", "氯化钾", "硫酸钾",
        "有机肥", "生物菌肥", "叶面肥", "冲施肥", "水溶肥",
        "吡虫啉", "啶虫脒", "阿维菌素", "甲维盐", "氯虫苯甲酰胺",
        "多菌灵", "代森锰锌", "百菌清", "甲基硫菌灵", "戊唑醇",
        "草甘膦", "草铵膦", "乙草胺", "莠去津", "烟嘧磺隆",
        "水稻种子", "玉米种子", "小麦种子", "蔬菜种子", "水果苗木",
        "农膜", "地膜", "大棚膜", "遮阳网", "防虫网",
    ],
}

# 子类别映射
SUB_CATEGORY = {
    "水果": ["仁果类", "柑橘类", "浆果类", "核果类", "热带水果", "瓜果类"],
    "蔬菜": ["叶菜类", "茄果类", "瓜类", "根茎类", "豆类", "花菜类"],
    "禽肉蛋品": ["蛋类", "禽肉类", "畜肉类"],
    "粮油": ["大米", "面粉", "食用油", "杂粮"],
    "农资": ["化肥", "农药", "种子", "农膜"],
}

# 上市季节
SEASONS = ["春季", "夏季", "秋季", "冬季", "全年"]


def generate_items():
    """生成农产品数据并保存为CSV"""
    provinces = list(PROVINCES.keys())
    items = []

    for item_id in range(1, NUM_ITEMS + 1):
        category = random.choices(ITEM_CATEGORIES, weights=ITEM_CATEGORY_WEIGHTS, k=1)[0]
        item_name = random.choice(ITEM_NAMES[category])
        sub_category = random.choice(SUB_CATEGORY[category])

        # 产地：从全国省份中随机选择
        origin = random.choice(provinces)

        # 价格
        if category == "水果":
            price = round(random.uniform(3.0, 30.0), 2)
        elif category == "蔬菜":
            price = round(random.uniform(1.5, 15.0), 2)
        elif category == "禽肉蛋品":
            price = round(random.uniform(8.0, 60.0), 2)
        elif category == "粮油":
            price = round(random.uniform(5.0, 80.0), 2)
        else:  # 农资
            price = round(random.uniform(10.0, 200.0), 2)

        # 季节
        season = random.choice(SEASONS)

        # 保质期
        if category in ["水果", "蔬菜"]:
            shelf_life = random.randint(3, 30)
        elif category == "禽肉蛋品":
            shelf_life = random.randint(7, 90)
        elif category == "粮油":
            shelf_life = random.randint(180, 720)
        else:
            shelf_life = random.randint(365, 1095)

        # 长尾标记：80%的农产品为长尾（幂律分布）
        # 长尾与品类相关：农资、粮油等标准化产品中长尾更少
        if category in ["水果", "蔬菜"]:
            is_long_tail = random.random() < 0.85  # 生鲜类长尾更多
        elif category == "禽肉蛋品":
            is_long_tail = random.random() < 0.75
        elif category == "粮油":
            is_long_tail = random.random() < 0.70
        else:  # 农资
            is_long_tail = random.random() < 0.60

        items.append({
            "item_id": item_id,
            "item_name": item_name,
            "category": category,
            "sub_category": sub_category,
            "price": price,
            "origin": origin,
            "season": season,
            "shelf_life": shelf_life,
            "is_long_tail": is_long_tail,
        })

    with open(ITEMS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)

    print(f"[OK] 农产品数据已生成：{ITEMS_FILE}，共 {len(items)} 条")
    return items


if __name__ == "__main__":
    generate_items()