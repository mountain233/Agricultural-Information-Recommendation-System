# -*- coding: utf-8 -*-
"""
生成用户数据
"""
import csv
import random
import numpy as np
from config import (NUM_USERS, USERS_FILE, RANDOM_SEED, USER_TYPES,
                    USER_TYPE_WEIGHTS, PROVINCES)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_users():
    """生成用户数据并保存为CSV"""
    provinces = list(PROVINCES.keys())
    province_weights = [PROVINCES[p]["weight"] for p in provinces]
    # 归一化权重
    total_w = sum(province_weights)
    province_weights = [w / total_w for w in province_weights]

    users = []
    for user_id in range(1, NUM_USERS + 1):
        user_type = random.choices(USER_TYPES, weights=USER_TYPE_WEIGHTS, k=1)[0]
        province = random.choices(provinces, weights=province_weights, k=1)[0]
        crop_type = random.choice(PROVINCES[province]["crops"])

        # 经营规模
        if user_type == "农户":
            farm_scale = random.choices(
                ["小农户", "家庭农场", "合作社"],
                weights=[0.6, 0.3, 0.1], k=1
            )[0]
        elif user_type == "农技人员":
            farm_scale = "农技服务站"
        elif user_type == "采购商":
            farm_scale = random.choices(
                ["小型采购商", "中型采购商", "大型采购商"],
                weights=[0.5, 0.35, 0.15], k=1
            )[0]
        else:
            farm_scale = "个人"

        # 注册时间
        year = random.randint(2020, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        register_time = f"{year}-{month:02d}-{day:02d}"

        # 使用混合分布：70%普通用户 + 20%活跃用户 + 10%冷启动用户
        r = random.random()
        if r < 0.10:
            # 冷启动用户：交互数 1-20
            interaction_count = random.randint(1, 20)
        elif r < 0.80:
            # 普通用户：交互数 20-80
            interaction_count = int(np.random.lognormal(mean=3.5, sigma=0.5))
            interaction_count = max(20, min(interaction_count, 80))
        else:
            # 活跃用户：交互数 80-200
            interaction_count = random.randint(80, 200)

        is_cold_start = interaction_count < 20

        users.append({
            "user_id": user_id,
            "user_type": user_type,
            "province": province,
            "crop_type": crop_type,
            "farm_scale": farm_scale,
            "register_time": register_time,
            "interaction_count": interaction_count,
            "is_cold_start": is_cold_start,
        })

    # 写入CSV
    with open(USERS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=users[0].keys())
        writer.writeheader()
        writer.writerows(users)

    print(f"[OK] 用户数据已生成：{USERS_FILE}，共 {len(users)} 条")
    return users


if __name__ == "__main__":
    generate_users()