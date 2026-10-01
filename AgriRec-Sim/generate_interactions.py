# -*- coding: utf-8 -*-
"""
生成用户-物品交互数据
"""
import csv
import random
import numpy as np
from datetime import datetime, timedelta
from config import (NUM_USERS, NUM_ITEMS, NUM_INTERACTIONS,
                    INTERACTIONS_FILE, RANDOM_SEED,
                    BEHAVIOR_TYPES, BEHAVIOR_WEIGHTS,
                    RATING_VALUES, RATING_WEIGHTS)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_interactions(users, items):
    """生成交互数据并保存为CSV"""
    # 构建索引
    user_dict = {u["user_id"]: u for u in users}
    item_dict = {i["item_id"]: i for i in items}

    # 为每个用户生成交互列表（基于其交互数）
    interactions = []
    interaction_id = 0

    # 用户类型与品类的偏好映射
    type_category_pref = {
        "农户": {"农资": 2.5, "粮油": 1.5, "水果": 1.0, "蔬菜": 1.0, "禽肉蛋品": 0.5},
        "农技人员": {"农资": 2.0, "粮油": 1.5, "水果": 1.2, "蔬菜": 1.2, "禽肉蛋品": 0.8},
        "采购商": {"水果": 2.0, "蔬菜": 1.8, "禽肉蛋品": 1.5, "粮油": 1.5, "农资": 0.5},
        "普通消费者": {"水果": 2.0, "蔬菜": 1.8, "禽肉蛋品": 1.5, "粮油": 1.2, "农资": 0.2},
    }

    # 时间范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    total_days = (end_date - start_date).days

    for user in users:
        user_id = user["user_id"]
        user_type = user["user_type"]
        province = user["province"]
        n_interactions = user["interaction_count"]

        # 限制单个用户最大交互数
        n_interactions = min(n_interactions, 200)

        # 为每个用户生成交互
        candidate_items = random.sample(items, min(len(items), n_interactions * 5))

        selected_count = 0
        for item in candidate_items:
            if selected_count >= n_interactions:
                break

            # 计算交互概率
            prob = 0.3  # 基础概率

            # 用户类型与品类偏好
            prob *= type_category_pref[user_type].get(item["category"], 1.0)

            # 地区匹配度
            if user["province"] == item["origin"]:
                prob *= 1.5

            # 长尾惩罚
            if item["is_long_tail"]:
                prob *= 0.03  # 从0.1降低至0.03，进一步强化长尾稀疏性

            # 归一化到[0,1]
            prob = min(prob, 1.0)

            if random.random() < prob:
                # 行为类型
                behavior = random.choices(BEHAVIOR_TYPES,
                                          weights=BEHAVIOR_WEIGHTS, k=1)[0]

                # 评分（仅购买和评分行为有）
                if behavior in ["购买", "评分"]:
                    rating = random.choices(RATING_VALUES,
                                            weights=RATING_WEIGHTS, k=1)[0]
                else:
                    rating = 0

                # 时间戳
                days_offset = random.randint(0, total_days)
                timestamp = start_date + timedelta(days=days_offset)
                timestamp = timestamp.replace(
                    hour=random.randint(0, 23),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )

                interaction_id += 1
                interactions.append({
                    "interaction_id": interaction_id,
                    "user_id": user_id,
                    "item_id": item["item_id"],
                    "behavior_type": behavior,
                    "rating": rating,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                })
                selected_count += 1

    # 如果交互数不足目标值，随机补充
    while len(interactions) < NUM_INTERACTIONS:
        user = random.choice(users)
        item = random.choice(items)
        behavior = random.choices(BEHAVIOR_TYPES, weights=BEHAVIOR_WEIGHTS, k=1)[0]
        rating = random.choices(RATING_VALUES, weights=RATING_WEIGHTS, k=1)[0] \
            if behavior in ["购买", "评分"] else 0
        days_offset = random.randint(0, total_days)
        timestamp = start_date + timedelta(days=days_offset)

        interaction_id += 1
        interactions.append({
            "interaction_id": interaction_id,
            "user_id": user["user_id"],
            "item_id": item["item_id"],
            "behavior_type": behavior,
            "rating": rating,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # 按时间排序
    interactions.sort(key=lambda x: x["timestamp"])

    with open(INTERACTIONS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=interactions[0].keys())
        writer.writeheader()
        writer.writerows(interactions)

    print(f"[OK] 交互数据已生成：{INTERACTIONS_FILE}，共 {len(interactions)} 条")
    return interactions


if __name__ == "__main__":
    from generate_users import generate_users
    from generate_items import generate_items
    users = generate_users()
    items = generate_items()
    generate_interactions(users, items)
    