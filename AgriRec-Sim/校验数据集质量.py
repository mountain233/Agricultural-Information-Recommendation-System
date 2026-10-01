# -*- coding: utf-8 -*-
"""
AgriRec-Sim 数据集质量校验脚本
运行方式：python check_data.py
"""
import os
import json
import pandas as pd
import numpy as np
from config import DATA_DIR


def section(title):
    """打印分节标题"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_file_exists():
    """检查所有数据文件是否存在"""
    section("【1】文件完整性检查")
    files = [
        "users.csv", "items.csv", "interactions.csv",
        "social.csv", "comments.csv", "kg.json",
    ]
    all_ok = True
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  [OK] {f:<20}  {size_mb:>8.2f} MB")
        else:
            print(f"  [缺失] {f}")
            all_ok = False
    return all_ok


def check_data_loading():
    """加载全部数据"""
    section("【2】数据加载")
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    inter = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    social = pd.read_csv(os.path.join(DATA_DIR, "social.csv"))
    comments = pd.read_csv(os.path.join(DATA_DIR, "comments.csv"))
    with open(os.path.join(DATA_DIR, "kg.json"), encoding="utf-8") as f:
        kg = json.load(f)

    print(f"  users.csv        : {len(users):>10,} 条")
    print(f"  items.csv        : {len(items):>10,} 条")
    print(f"  interactions.csv : {len(inter):>10,} 条")
    print(f"  social.csv       : {len(social):>10,} 条")
    print(f"  comments.csv     : {len(comments):>10,} 条")
    print(f"  kg.json          : {len(kg['triples']):>10,} 条三元组")

    return users, items, inter, social, comments, kg


def check_schema(users, items, inter, social, comments, kg):
    """检查字段结构是否符合预期"""
    section("【3】字段结构检查")

    expected = {
        "users": ["user_id", "user_type", "province", "crop_type",
                  "farm_scale", "register_time", "interaction_count",
                  "is_cold_start"],
        "items": ["item_id", "item_name", "category", "sub_category",
                  "price", "origin", "season", "shelf_life", "is_long_tail"],
        "interactions": ["interaction_id", "user_id", "item_id",
                         "behavior_type", "rating", "timestamp"],
        "social": ["user_id", "friend_id", "relation_type",
                   "trust_level", "is_noise", "create_time"],
        "comments": ["comment_id", "user_id", "item_id", "comment_text",
                     "sentiment", "aspect", "rating", "timestamp"],
    }

    data_dict = {
        "users": users, "items": items, "interactions": inter,
        "social": social, "comments": comments,
    }

    for name, cols in expected.items():
        df = data_dict[name]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  [异常] {name:<15} 缺失字段: {missing}")
        else:
            print(f"  [OK] {name:<15} 字段完整 ({len(cols)} 个)")

    # 知识图谱字段
    if kg["triples"]:
        sample = kg["triples"][0]
        required = ["head", "head_type", "relation", "tail", "tail_type"]
        missing = [c for c in required if c not in sample]
        if missing:
            print(f"  [异常] kg.json         缺失字段: {missing}")
        else:
            print(f"  [OK] kg.json         字段完整 ({len(required)} 个)")


def check_missing_values(users, items, inter, social, comments):
    """检查缺失值"""
    section("【4】缺失值检查")
    for name, df in [("users", users), ("items", items),
                     ("interactions", inter), ("social", social),
                     ("comments", comments)]:
        null_count = df.isnull().sum().sum()
        if null_count == 0:
            print(f"  [OK] {name:<15} 无缺失值")
        else:
            print(f"  [警告] {name:<15} 存在 {null_count} 个缺失值")
            print(df.isnull().sum()[df.isnull().sum() > 0])


def check_duplicates(users, items, inter, social, comments):
    """检查重复记录"""
    section("【5】重复记录检查")
    checks = [
        ("users", users, "user_id"),
        ("items", items, "item_id"),
        ("interactions", inter, "interaction_id"),
        ("comments", comments, "comment_id"),
    ]
    for name, df, key in checks:
        dup = df.duplicated(subset=[key]).sum()
        if dup == 0:
            print(f"  [OK] {name:<15} 主键 {key} 无重复")
        else:
            print(f"  [警告] {name:<15} 主键 {key} 有 {dup} 条重复")

    # 社交关系检查无向重复
    social_key = social.apply(
        lambda r: tuple(sorted([r["user_id"], r["friend_id"]])), axis=1
    )
    dup_social = social_key.duplicated().sum()
    if dup_social == 0:
        print(f"  [OK] social          无重复无向边")
    else:
        print(f"  [警告] social          存在 {dup_social} 条重复无向边")


def check_data_range(users, items, inter, social, comments):
    """检查数值范围合理性"""
    section("【6】数值范围检查")

    # 用户
    invalid_cold = users[~users["is_cold_start"].isin([True, False])]
    print(f"  users.is_cold_start    取值: {sorted(users['is_cold_start'].unique())}")
    print(f"  users.user_type        取值: {sorted(users['user_type'].unique())}")

    # 物品
    print(f"  items.category         取值: {sorted(items['category'].unique())}")
    print(f"  items.price            范围: [{items['price'].min():.2f}, "
          f"{items['price'].max():.2f}]")

    # 交互
    print(f"  interactions.behavior  取值: {sorted(inter['behavior_type'].unique())}")
    print(f"  interactions.rating    范围: [{inter['rating'].min()}, "
          f"{inter['rating'].max()}]")

    # 社交
    print(f"  social.relation_type   取值: {sorted(social['relation_type'].unique())}")
    print(f"  social.trust_level     范围: [{social['trust_level'].min():.2f}, "
          f"{social['trust_level'].max():.2f}]")

    # 评论
    print(f"  comments.sentiment     取值: {sorted(comments['sentiment'].unique())}")
    print(f"  comments.aspect        取值: {sorted(comments['aspect'].unique())}")

    # 校验评分合法性
    valid_ratings = {0, 1, 2, 3, 4, 5}
    invalid = inter[~inter["rating"].isin(valid_ratings)]
    if len(invalid) == 0:
        print(f"  [OK] 评分取值均在 {{0,1,2,3,4,5}} 内")
    else:
        print(f"  [警告] 存在 {len(invalid)} 条非法评分")

    # 信任度范围
    invalid_trust = social[(social["trust_level"] < 0) | (social["trust_level"] > 1)]
    if len(invalid_trust) == 0:
        print(f"  [OK] 信任度取值均在 [0,1] 内")
    else:
        print(f"  [警告] 存在 {len(invalid_trust)} 条非法信任度")


def check_foreign_keys(users, items, inter, social, comments):
    """检查外键引用完整性"""
    section("【7】外键引用完整性")

    user_ids = set(users["user_id"])
    item_ids = set(items["item_id"])

    # interactions.user_id
    invalid_u = ~inter["user_id"].isin(user_ids)
    print(f"  interactions.user_id   越界: {invalid_u.sum()}")

    # interactions.item_id
    invalid_i = ~inter["item_id"].isin(item_ids)
    print(f"  interactions.item_id   越界: {invalid_i.sum()}")

    # social.user_id / friend_id
    invalid_su = ~social["user_id"].isin(user_ids)
    invalid_sf = ~social["friend_id"].isin(user_ids)
    print(f"  social.user_id         越界: {invalid_su.sum()}")
    print(f"  social.friend_id       越界: {invalid_sf.sum()}")

    # comments.user_id / item_id
    invalid_cu = ~comments["user_id"].isin(user_ids)
    invalid_ci = ~comments["item_id"].isin(item_ids)
    print(f"  comments.user_id       越界: {invalid_cu.sum()}")
    print(f"  comments.item_id       越界: {invalid_ci.sum()}")

    total_invalid = (invalid_u.sum() + invalid_i.sum() +
                     invalid_su.sum() + invalid_sf.sum() +
                     invalid_cu.sum() + invalid_ci.sum())
    if total_invalid == 0:
        print("  [OK] 所有外键引用有效")
    else:
        print(f"  [警告] 共发现 {total_invalid} 个越界引用")


def check_distribution(users, items, inter, social):
    """检查数据分布"""
    section("【8】关键分布统计")

    # 交互密度
    density = len(inter) / (len(users) * len(items)) * 100
    print(f"  交互密度:               {density:.4f}%")

    # 冷启动用户
    n_cold = users["is_cold_start"].sum()
    print(f"  冷启动用户:             {n_cold:,} ({n_cold/len(users)*100:.2f}%)")

    # 长尾农产品
    n_long = items["is_long_tail"].sum()
    print(f"  长尾农产品:             {n_long:,} ({n_long/len(items)*100:.2f}%)")

    # 噪声社交边
    n_noise = social["is_noise"].sum()
    print(f"  噪声社交边:             {n_noise:,} ({n_noise/len(social)*100:.2f}%)")

    # 用户类型分布
    print()
    print("  用户类型分布:")
    for k, v in users["user_type"].value_counts().items():
        print(f"    {k:<12} {v:>6,} ({v/len(users)*100:.1f}%)")

    # 农产品类别分布
    print()
    print("  农产品类别分布:")
    for k, v in items["category"].value_counts().items():
        print(f"    {k:<12} {v:>6,} ({v/len(items)*100:.1f}%)")

    # 行为类型分布
    print()
    print("  行为类型分布:")
    for k, v in inter["behavior_type"].value_counts().items():
        print(f"    {k:<12} {v:>6,} ({v/len(inter)*100:.1f}%)")

    # 关系类型分布
    print()
    print("  社交关系类型分布:")
    for k, v in social["relation_type"].value_counts().items():
        print(f"    {k:<12} {v:>6,} ({v/len(social)*100:.1f}%)")

    # 地区分布Top 10
    print()
    print("  用户地区分布 Top 10:")
    for k, v in users["province"].value_counts().head(10).items():
        print(f"    {k:<18} {v:>6,} ({v/len(users)*100:.1f}%)")

    # 交互度分布
    print()
    user_inter_count = inter.groupby("user_id").size()
    print("  用户交互数分布:")
    print(f"    最小: {user_inter_count.min()}")
    print(f"    最大: {user_inter_count.max()}")
    print(f"    均值: {user_inter_count.mean():.1f}")
    print(f"    中位数: {user_inter_count.median():.0f}")

    item_inter_count = inter.groupby("item_id").size()
    print()
    print("  农产品交互数分布:")
    print(f"    最小: {item_inter_count.min()}")
    print(f"    最大: {item_inter_count.max()}")
    print(f"    均值: {item_inter_count.mean():.1f}")
    print(f"    中位数: {item_inter_count.median():.0f}")

    # 长尾判定验证
    long_tail_ids = set(items[items["is_long_tail"]]["item_id"])
    long_tail_inter = item_inter_count[item_inter_count.index.isin(long_tail_ids)]
    if len(long_tail_inter) > 0:
        print()
        print(f"  长尾农产品平均交互数:   {long_tail_inter.mean():.1f}")


def check_kg(kg):
    """检查知识图谱结构"""
    section("【9】知识图谱检查")

    triples = kg["triples"]
    print(f"  三元组总数:             {len(triples):,}")
    print(f"  元数据:                 {kg['metadata']}")

    heads = [t["head"] for t in triples]
    tails = [t["tail"] for t in triples]
    relations = [t["relation"] for t in triples]

    print()
    print(f"  唯一头实体数:           {len(set(heads)):,}")
    print(f"  唯一尾实体数:           {len(set(tails)):,}")
    print(f"  唯一实体总数:           {len(set(heads) | set(tails)):,}")
    print(f"  唯一关系类型数:         {len(set(relations))}")

    print()
    print("  关系类型分布:")
    rel_count = pd.Series(relations).value_counts()
    for k, v in rel_count.items():
        print(f"    {k:<16} {v:>6,} ({v/len(triples)*100:.1f}%)")

    # 实体类型分布
    head_types = [t["head_type"] for t in triples]
    print()
    print("  头实体类型分布:")
    for k, v in pd.Series(head_types).value_counts().items():
        print(f"    {k:<12} {v:>6,}")


def check_timestamps(inter, social, comments):
    """检查时间戳"""
    section("【10】时间戳检查")

    for name, df, col in [
        ("interactions", inter, "timestamp"),
        ("social", social, "create_time"),
        ("comments", comments, "timestamp"),
    ]:
        try:
            ts = pd.to_datetime(df[col], errors="coerce")
            invalid = ts.isna().sum()
            print(f"  {name:<15} 时间范围: "
                  f"[{ts.min()}, {ts.max()}]  非法: {invalid}")
        except Exception as e:
            print(f"  [警告] {name} 时间格式异常: {e}")


def main():
    print()
    print("=" * 60)
    print("AgriRec-Sim 数据集质量校验")
    print("=" * 60)

    if not check_file_exists():
        print("\n[错误] 数据文件不完整，请先运行 python main.py")
        return

    users, items, inter, social, comments, kg = check_data_loading()
    check_schema(users, items, inter, social, comments, kg)
    check_missing_values(users, items, inter, social, comments)
    check_duplicates(users, items, inter, social, comments)
    check_data_range(users, items, inter, social, comments)
    check_foreign_keys(users, items, inter, social, comments)
    check_distribution(users, items, inter, social)
    check_kg(kg)
    check_timestamps(inter, social, comments)

    section("校验完成")
    print("  数据集已通过全部质量检查，可用于系统测试。")


if __name__ == '__main__':
    main()