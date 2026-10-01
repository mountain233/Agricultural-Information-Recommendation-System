# -*- coding: utf-8 -*-
"""
生成社交关系数据
"""
import csv
import random
import numpy as np
from config import (NUM_USERS, NUM_SOCIAL_EDGES, SOCIAL_FILE,
                    RANDOM_SEED, RELATION_TYPES, RELATION_WEIGHTS)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_social(users):
    """生成社交关系数据并保存为CSV"""
    user_dict = {u["user_id"]: u for u in users}
    social_edges = []
    edge_set = set()  # 避免重复边

    while len(social_edges) < NUM_SOCIAL_EDGES:
        user = random.choice(users)
        friend = random.choice(users)

        if user["user_id"] == friend["user_id"]:
            continue

        # 避免重复（无向边）
        edge_key = tuple(sorted([user["user_id"], friend["user_id"]]))
        if edge_key in edge_set:
            continue

        edge_set.add(edge_key)

        # 关系类型
        relation = random.choices(RELATION_TYPES,
                                  weights=RELATION_WEIGHTS, k=1)[0]

        # 信任度
        if relation == "噪声边":
            trust = round(random.uniform(0.1, 0.3), 2)
            is_noise = True
        else:
            # 真实关系：基于地区、作物类型计算信任度
            base_trust = 0.4
            if user["province"] == friend["province"]:
                base_trust += 0.3
            if user["crop_type"] == friend["crop_type"]:
                base_trust += 0.2
            # 加入随机扰动
            base_trust += random.uniform(-0.1, 0.1)
            trust = round(min(max(base_trust, 0.1), 0.99), 2)
            is_noise = False

        # 关系建立时间
        year = random.randint(2020, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        create_time = f"{year}-{month:02d}-{day:02d}"

        social_edges.append({
            "user_id": user["user_id"],
            "friend_id": friend["user_id"],
            "relation_type": relation,
            "trust_level": trust,
            "is_noise": is_noise,
            "create_time": create_time,
        })

    with open(SOCIAL_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=social_edges[0].keys())
        writer.writeheader()
        writer.writerows(social_edges)

    print(f"[OK] 社交关系数据已生成：{SOCIAL_FILE}，共 {len(social_edges)} 条")
    return social_edges


if __name__ == "__main__":
    from generate_users import generate_users
    users = generate_users()
    generate_social(users)