# -*- coding: utf-8 -*-
"""
生成农业知识图谱（JSON格式）
"""
import json
import random
import numpy as np
from config import NUM_KG_TRIPLES, KG_FILE, RANDOM_SEED, PROVINCES

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 知识图谱实体库 ====================

# 作物
CROPS = [
    "水稻", "小麦", "玉米", "大豆", "马铃薯", "红薯", "高粱", "谷子",
    "柑橘", "苹果", "梨", "桃", "葡萄", "猕猴桃", "草莓", "樱桃",
    "芒果", "荔枝", "龙眼", "香蕉", "菠萝", "西瓜", "甜瓜", "石榴",
    "白菜", "番茄", "黄瓜", "辣椒", "茄子", "萝卜", "胡萝卜", "洋葱",
    "大蒜", "生姜", "大葱", "菠菜", "生菜", "芹菜", "韭菜", "豆角",
    "茶树", "咖啡", "花椒", "核桃", "板栗", "油茶", "桑树", "枸杞",
]

# 病虫害
PESTS = [
    "稻飞虱", "稻纵卷叶螟", "二化螟", "三化螟", "稻瘟病", "纹枯病",
    "小麦锈病", "小麦赤霉病", "麦蚜", "玉米螟", "玉米锈病", "草地贪夜蛾",
    "大豆蚜虫", "大豆食心虫", "马铃薯晚疫病", "马铃薯早疫病", "二十八星瓢虫",
    "柑橘黄龙病", "柑橘红蜘蛛", "柑橘潜叶蛾", "柑橘溃疡病", "柑橘炭疽病",
    "苹果腐烂病", "苹果轮纹病", "苹果蚜虫", "苹果红蜘蛛", "桃小食心虫",
    "葡萄霜霉病", "葡萄白粉病", "葡萄炭疽病", "猕猴桃溃疡病", "猕猴桃褐斑病",
    "草莓白粉病", "草莓灰霉病", "草莓红蜘蛛", "樱桃果蝇", "樱桃褐腐病",
    "番茄晚疫病", "番茄灰霉病", "番茄白粉病", "番茄蚜虫", "黄瓜霜霉病",
    "黄瓜白粉病", "黄瓜枯萎病", "辣椒炭疽病", "辣椒疫病", "茄子绵疫病",
    "白菜软腐病", "白菜霜霉病", "白菜蚜虫", "菜青虫", "小菜蛾",
    "茶树茶小绿叶蝉", "茶树炭疽病", "茶树螨类", "枸杞木虱", "枸杞瘿螨",
]

# 农资产品
FERTILIZERS = [
    "尿素", "复合肥", "磷酸二铵", "氯化钾", "硫酸钾", "硝酸铵",
    "碳酸氢铵", "过磷酸钙", "钙镁磷肥", "有机肥", "生物菌肥",
    "叶面肥", "冲施肥", "水溶肥", "缓释肥", "控释肥",
]

PESTICIDES = [
    "吡虫啉", "啶虫脒", "噻虫嗪", "呋虫胺", "烯啶虫胺",
    "阿维菌素", "甲维盐", "氯虫苯甲酰胺", "茚虫威", "虫螨腈",
    "多菌灵", "代森锰锌", "百菌清", "甲基硫菌灵", "戊唑醇",
    "苯醚甲环唑", "丙环唑", "嘧菌酯", "吡唑醚菌酯", "氟环唑",
    "草甘膦", "草铵膦", "乙草胺", "莠去津", "烟嘧磺隆",
    "高效氯氟氰菊酯", "联苯菊酯", "溴氰菊酯", "甲氰菊酯",
]

SEEDS = [
    "水稻种子", "玉米种子", "小麦种子", "大豆种子", "马铃薯种薯",
    "蔬菜种子", "水果苗木", "茶树苗", "中药材种子",
]

# 栽培技术
TECHNIQUES = [
    "育苗", "嫁接", "扦插", "压条", "分株", "修剪", "疏花疏果",
    "套袋", "覆膜", "滴灌", "喷灌", "水肥一体化", "测土配方施肥",
    "绿色防控", "统防统治", "轮作", "间作", "套种", "免耕",
    "秸秆还田", "大棚种植", "温室栽培", "无土栽培", "有机种植",
]

# 气候条件
CLIMATES = [
    "温暖湿润", "温暖干燥", "凉爽湿润", "寒冷干燥", "四季分明",
    "雨热同期", "光照充足", "昼夜温差大", "无霜期长",
]

# 土壤类型
SOILS = [
    "红壤", "黄壤", "紫色土", "水稻土", "潮土", "褐土", "黑土",
    "沙壤土", "黏土", "壤土", "酸性土", "碱性土",
]


def generate_kg():
    """生成农业知识图谱并保存为JSON"""
    triples = []
    triple_set = set()  # 避免重复

    def add_triple(head, relation, tail, head_type, tail_type):
        key = (head, relation, tail)
        if key not in triple_set:
            triple_set.add(key)
            triples.append({
                "head": head,
                "head_type": head_type,
                "relation": relation,
                "tail": tail,
                "tail_type": tail_type,
            })

    # ========== 1. 作物 - 易感病虫害 ==========
    for crop in CROPS:
        n_pests = random.randint(2, 5)
        for pest in random.sample(PESTS, min(n_pests, len(PESTS))):
            add_triple(crop, "易感病虫害", pest, "作物", "病虫害")

    # ========== 2. 病虫害 - 防治方法（农药） ==========
    for pest in PESTS:
        n_pesticides = random.randint(1, 3)
        for pesticide in random.sample(PESTICIDES, min(n_pesticides, len(PESTICIDES))):
            add_triple(pest, "防治方法", pesticide, "病虫害", "农药")

    # ========== 3. 作物 - 适用农资（化肥） ==========
    for crop in CROPS:
        n_fertilizers = random.randint(1, 3)
        for fertilizer in random.sample(FERTILIZERS, min(n_fertilizers, len(FERTILIZERS))):
            add_triple(crop, "适用农资", fertilizer, "作物", "化肥")

    # ========== 4. 作物 - 栽培技术 ==========
    for crop in CROPS:
        n_techniques = random.randint(2, 4)
        for technique in random.sample(TECHNIQUES, min(n_techniques, len(TECHNIQUES))):
            add_triple(crop, "需要技术", technique, "作物", "栽培技术")

    # ========== 5. 作物 - 适宜气候 ==========
    for crop in CROPS:
        n_climates = random.randint(1, 2)
        for climate in random.sample(CLIMATES, min(n_climates, len(CLIMATES))):
            add_triple(crop, "适宜气候", climate, "作物", "气候条件")

    # ========== 6. 作物 - 适宜土壤 ==========
    for crop in CROPS:
        n_soils = random.randint(1, 2)
        for soil in random.sample(SOILS, min(n_soils, len(SOILS))):
            add_triple(crop, "适宜土壤", soil, "作物", "土壤类型")

    # ========== 7. 作物 - 适宜地区（基于省份） ==========
    for province, info in PROVINCES.items():
        for crop in info["crops"]:
            if crop in CROPS:
                add_triple(crop, "适宜地区", province, "作物", "地区")

    # ========== 8. 病虫害 - 易发气候 ==========
    for pest in PESTS:
        n_climates = random.randint(1, 2)
        for climate in random.sample(CLIMATES, min(n_climates, len(CLIMATES))):
            add_triple(pest, "易发气候", climate, "病虫害", "气候条件")

    # ========== 9. 补充三元组至目标数量 ==========
    all_heads = CROPS + PESTS + FERTILIZERS + PESTICIDES + TECHNIQUES
    all_relations = ["关联", "相关", "适用于", "推荐使用"]

    while len(triples) < NUM_KG_TRIPLES:
        head = random.choice(all_heads)
        tail = random.choice(all_heads)
        if head == tail:
            continue
        relation = random.choice(all_relations)
        head_type = "实体"
        tail_type = "实体"
        add_triple(head, relation, tail, head_type, tail_type)

    # 保存为JSON
    kg_data = {
        "metadata": {
            "total_triples": len(triples),
            "entity_types": ["作物", "病虫害", "化肥", "农药", "种子",
                             "栽培技术", "气候条件", "土壤类型", "地区"],
            "relation_types": ["易感病虫害", "防治方法", "适用农资",
                               "需要技术", "适宜气候", "适宜土壤",
                               "适宜地区", "易发气候", "关联", "相关",
                               "适用于", "推荐使用"],
        },
        "triples": triples,
    }

    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 知识图谱已生成：{KG_FILE}，共 {len(triples)} 条三元组")
    return triples


if __name__ == "__main__":
    generate_kg()