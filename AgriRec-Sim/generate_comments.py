# -*- coding: utf-8 -*-
"""
生成评论文本数据
"""
import csv
import random
from config import (NUM_COMMENTS, COMMENTS_FILE, RANDOM_SEED)

random.seed(RANDOM_SEED)

# ==================== 评论模板 ====================
# 按情感极性和方面词组织

POSITIVE_TEMPLATES = {
    "口感": [
        "口感非常好，{item}很甜/很香，家人都喜欢吃",
        "味道不错，{item}新鲜多汁，下次还会回购",
        "{item}口感极佳，吃起来很满意，推荐购买",
        "这个{item}味道特别好，孩子很爱吃",
    ],
    "外观": [
        "{item}个头大，颜色鲜艳，看着就很有食欲",
        "包装很好，{item}外观完整，没有磕碰",
        "{item}品相很好，大小均匀，很新鲜",
    ],
    "新鲜度": [
        "{item}非常新鲜，叶子/果皮翠绿，一看就是刚采摘的",
        "收到货很新鲜，{item}没有腐烂，很满意",
        "{item}新鲜度很高，比菜市场买的还好",
    ],
    "物流": [
        "物流很快，{item}包装严实，没有损坏",
        "发货速度快，{item}第二天就到了，很新鲜",
        "快递给力，{item}完好无损，赞一个",
    ],
    "价格": [
        "{item}性价比很高，比超市便宜多了",
        "价格实惠，{item}质量也不错，很划算",
        "{item}物美价廉，值得购买",
    ],
}

NEGATIVE_TEMPLATES = {
    "口感": [
        "{item}口感一般，没有想象中好吃，有点失望",
        "这个{item}味道不太好，不太新鲜",
        "{item}吃起来口感偏酸/偏涩，不太满意",
    ],
    "外观": [
        "{item}个头太小，和图片差距很大",
        "{item}外观有磕碰，品相不好",
        "收到的{item}颜色暗淡，看起来不新鲜",
    ],
    "新鲜度": [
        "{item}不新鲜，有几颗已经坏了",
        "收到时{item}已经蔫了，新鲜度不够",
        "{item}有腐烂的，挑出来扔了好几个",
    ],
    "物流": [
        "物流太慢了，{item}收到时已经不新鲜了",
        "快递过程中{item}被压坏了，包装需要改进",
        "发货慢，{item}在路上走了好几天",
    ],
    "价格": [
        "{item}价格偏贵，性价比不高",
        "这个价格买到这样的{item}，有点不值",
        "{item}价格波动太大，买贵了",
    ],
}

NEUTRAL_TEMPLATES = [
    "{item}还可以，中规中矩，没有特别惊艳",
    "一般般吧，{item}和描述基本相符",
    "{item}无功无过，凑合能用",
]


def generate_comments(users, items, interactions):
    """生成评论文本数据"""
    # 从交互中筛选出有评分/购买行为的记录
    purchase_interactions = [
        i for i in interactions
        if i["behavior_type"] in ["购买", "评分"]
    ]

    # 构建索引
    user_dict = {u["user_id"]: u for u in users}
    item_dict = {i["item_id"]: i for i in items}

    comments = []
    comment_id = 0

    # 从购买记录中随机采样生成评论
    sample_size = min(NUM_COMMENTS, len(purchase_interactions))
    sampled = random.sample(purchase_interactions, sample_size)

    for inter in sampled:
        user = user_dict.get(inter["user_id"])
        item = item_dict.get(inter["item_id"])
        if not user or not item:
            continue

        rating = inter["rating"]

        # 根据评分决定情感极性
        if rating >= 4:
            sentiment = "积极"
            templates = POSITIVE_TEMPLATES
        elif rating <= 2:
            sentiment = "消极"
            templates = NEGATIVE_TEMPLATES
        else:
            sentiment = "中性"
            templates = None

        # 选择方面词
        aspect = random.choice(["口感", "外观", "新鲜度", "物流", "价格"])

        # 生成评论文本
        if templates:
            template = random.choice(templates[aspect])
            comment_text = template.format(item=item["item_name"])
        else:
            comment_text = random.choice(NEUTRAL_TEMPLATES).format(
                item=item["item_name"])

        comment_id += 1
        comments.append({
            "comment_id": comment_id,
            "user_id": inter["user_id"],
            "item_id": inter["item_id"],
            "comment_text": comment_text,
            "sentiment": sentiment,
            "aspect": aspect,
            "rating": rating,
            "timestamp": inter["timestamp"],
        })

    with open(COMMENTS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=comments[0].keys())
        writer.writeheader()
        writer.writerows(comments)

    print(f"[OK] 评论数据已生成：{COMMENTS_FILE}，共 {len(comments)} 条")
    return comments


if __name__ == "__main__":
    from generate_users import generate_users
    from generate_items import generate_items
    from generate_interactions import generate_interactions

    users = generate_users()
    items = generate_items()
    interactions = generate_interactions(users, items)
    generate_comments(users, items, interactions)