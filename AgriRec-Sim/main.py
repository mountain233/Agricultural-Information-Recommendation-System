# -*- coding: utf-8 -*-
"""
AgriRec-Sim 主入口
一键生成完整的农业推荐系统模拟数据集
"""
import os
import time
from config import DATA_DIR


def main():
    """主函数：依次生成所有数据"""
    # 创建输出目录
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("AgriRec-Sim 农业推荐系统模拟数据集生成")
    print("=" * 60)

    start_time = time.time()

    # ========== 1. 生成用户数据 ==========
    print("\n[1/6] 生成用户数据...")
    from generate_users import generate_users
    users = generate_users()

    # ========== 2. 生成农产品数据 ==========
    print("\n[2/6] 生成农产品数据...")
    from generate_items import generate_items
    items = generate_items()

    # ========== 3. 生成交互数据 ==========
    print("\n[3/6] 生成用户-物品交互数据...")
    from generate_interactions import generate_interactions
    interactions = generate_interactions(users, items)

    # ========== 4. 生成社交关系数据 ==========
    print("\n[4/6] 生成社交关系数据...")
    from generate_social import generate_social
    social = generate_social(users)

    # ========== 5. 生成知识图谱 ==========
    print("\n[5/6] 生成农业知识图谱...")
    from generate_kg import generate_kg
    kg = generate_kg()

    # ========== 6. 生成评论文本数据 ==========
    print("\n[6/6] 生成评论文本数据...")
    from generate_comments import generate_comments
    comments = generate_comments(users, items, interactions)

    # ========== 汇总统计 ==========
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)
    print(f"用户数：        {len(users):>10,}")
    print(f"农产品数：      {len(items):>10,}")
    print(f"交互数：        {len(interactions):>10,}")
    print(f"社交关系数：    {len(social):>10,}")
    print(f"知识图谱三元组：{len(kg):>10,}")
    print(f"评论数：        {len(comments):>10,}")
    print(f"总耗时：        {elapsed:.2f} 秒")
    print("=" * 60)

    # ========== 输出数据集统计 ==========
    print("\n【数据集统计】")
    print(f"交互密度：      {len(interactions) / (len(users) * len(items)) * 100:.2f}%")
    print(f"社交密度：      {len(social) / (len(users) * (len(users) - 1) / 2) * 100:.4f}%")

    # 冷启动用户统计
    cold_start_users = [u for u in users if u["is_cold_start"]]
    print(f"冷启动用户数：  {len(cold_start_users):,} "
          f"({len(cold_start_users) / len(users) * 100:.1f}%)")

    # 长尾物品统计
    long_tail_items = [i for i in items if i["is_long_tail"]]
    print(f"长尾农产品数：  {len(long_tail_items):,} "
          f"({len(long_tail_items) / len(items) * 100:.1f}%)")

    # 噪声社交边统计
    noise_edges = [s for s in social if s["is_noise"]]
    print(f"噪声社交边数：  {len(noise_edges):,} "
          f"({len(noise_edges) / len(social) * 100:.1f}%)")

    print("\n数据集已保存至：", os.path.abspath(DATA_DIR))


if __name__ == "__main__":
    main()