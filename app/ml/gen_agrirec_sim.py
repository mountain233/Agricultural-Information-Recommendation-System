# -*- coding: utf-8 -*-
"""AgriRec-Sim 农业场景模拟数据集生成器
对应论文 6.5.1 节：10,000 用户 / 2,000 农产品 / ~308k 交互 / 50k 社交关系(含10%噪声) / 5,000 知识图谱三元组(12种关系)
"""
import numpy as np
import json, os, math

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUT, exist_ok=True)

N_USERS = 10000
N_ITEMS = 2000
N_INTERACTIONS_TARGET = 400_000
N_SOCIAL_TARGET = 50_000
NOISE_RATIO = 0.10

PROVINCES = ["四川","河南","山东","黑龙江","云南","广西","新疆","河北","湖南","湖北",
             "安徽","江苏","广东","陕西","甘肃","贵州","辽宁","吉林","福建","江西",
             "山西","内蒙古","浙江","重庆","海南","宁夏","青海","西藏","天津","北京","上海"]
CROPS = ["柑橘","苹果","水稻","小麦","玉米","茶叶","葡萄","番茄","棉花","马铃薯",
         "猕猴桃","花生","大豆","辣椒","甘蔗"]
CATEGORIES = ["水果","蔬菜","禽肉蛋品","粮油","农资"]
USER_TYPES = ["农户","农技人员","采购商","消费者","合作社管理员"]

# 作物 -> 偏好品类映射
CROP_CAT = {
    "柑橘":["水果","农资"],"苹果":["水果","农资"],"葡萄":["水果","农资"],"猕猴桃":["水果","农资"],
    "水稻":["粮油","农资"],"小麦":["粮油","农资"],"玉米":["粮油","农资"],"大豆":["粮油","农资"],"花生":["粮油","农资"],
    "番茄":["蔬菜","农资"],"辣椒":["蔬菜","农资"],"马铃薯":["蔬菜","农资"],
    "茶叶":["水果","农资"],"棉花":["农资","粮油"],"甘蔗":["粮油","农资"],
}

# ---------- 1. 物品 ----------
ITEM_NAMES = {
 "水果":["红富士苹果","砂糖橘","赣南脐橙","阳光玫瑰葡萄","砀山酥梨","海南芒果","烟台大樱桃","徐香猕猴桃","麒麟西瓜","琯溪蜜柚","冬枣","火龙果","蓝莓","沃柑","库尔勒香梨","石榴","枇杷","草莓","荔枝","龙眼"],
 "蔬菜":["有机番茄","水果黄瓜","紫皮大蒜","高山娃娃菜","寿光彩椒","铁棍山药","恩施土豆","云南松茸","有机西兰花","速冻甜玉米粒","鲜百合","莲藕","芦笋","香菇","黑木耳","秋葵","苦瓜","茭白","韭菜","生姜"],
 "禽肉蛋品":["散养土鸡蛋","生态土鸡","黑猪肉","散养鸭","鲜鹅蛋","有机牛奶","蜂蜜","羔羊排","鹌鹑蛋","咸鸭蛋","皮蛋","风干牛肉","腊肠","土蜂蜜","鲜牛乳","鸽蛋","鹅肉","兔肉","驴肉","鹿产品"],
 "粮油":["五常大米","有机小米","压榨花生油","菜籽油","全麦面粉","玉米糁","黑芝麻","有机绿豆","红小豆","高粱米","燕麦米","山茶油","亚麻籽油","薏米","黑米","糙米","荞麦","富硒大米","胚芽米","糯米"],
 "农资":["柑橘专用复合肥","生物有机肥","高效低毒农药","抗病柑橘苗","滴灌设备","农用地膜","太阳能杀虫灯","水溶肥","植保无人机服务","种子包衣剂","土壤改良剂","大棚骨架","智能温室控制器","果树修剪工具","除草剂","杀菌剂","叶面肥","微量元素肥","育苗基质","农机润滑油"],
}

CAT_CROPS = {"水果":["柑橘","苹果","葡萄","猕猴桃"],"蔬菜":["番茄","辣椒","马铃薯"],
             "禽肉蛋品":["玉米","大豆"],"粮油":["水稻","小麦","玉米","大豆","花生","甘蔗"],
             "农资":CROPS}
items = []
for i in range(N_ITEMS):
    cat = CATEGORIES[i % len(CATEGORIES)]
    base = ITEM_NAMES[cat][i % len(ITEM_NAMES[cat])]
    origin = PROVINCES[rng.integers(0, len(PROVINCES))]
    crop = CAT_CROPS[cat][rng.integers(0, len(CAT_CROPS[cat]))]
    price = round(float(rng.lognormal(3.0 if cat=="农资" else 2.2, 0.6)), 2)
    items.append({"item_id": i, "name": f"{base}·{origin}产" if cat!="农资" else f"{base}({origin}直供)",
                  "category": cat, "price": price, "origin": origin, "crop": crop,
                  "is_long_tail": 0})

# 物品热度：陡峭幂律分布（头部吸收大部分交互，长尾物品 <5 次）
pop_weights = 1.0 / np.power(np.arange(1, N_ITEMS+1), 1.9)
rng.shuffle(pop_weights)
pop_weights /= pop_weights.sum()

# ---------- 2. 用户 ----------
users = []
for u in range(N_USERS):
    ut = USER_TYPES[rng.choice(len(USER_TYPES), p=[0.52,0.13,0.15,0.15,0.05])]
    crop = CROPS[rng.integers(0, len(CROPS))]
    prov = PROVINCES[rng.integers(0, len(PROVINCES))]
    scale = int(rng.choice([1,2,3,4,5], p=[0.35,0.3,0.2,0.1,0.05]))
    # 用户品类偏好（基于作物 + 较强随机扰动，模拟真实偏好的多样性）
    pref = np.zeros(len(CATEGORIES))
    for c in CROP_CAT[crop]: pref[CATEGORIES.index(c)] += 1.0
    pref += rng.random(len(CATEGORIES)) * 0.9
    pref /= pref.sum()
    users.append({"user_id": u, "user_type": ut, "province": prov, "crop_type": crop,
                  "farm_scale": scale, "pref": pref.tolist()})

# ---------- 3. 交互 ----------
# 隐式反馈：行为类型 click/collect/purchase/rating/skip，热度×品类亲和
item_cat = np.array([CATEGORIES.index(it["category"]) for it in items])
user_pref = np.array([u["pref"] for u in users])           # (U, 5)
cat_affinity = user_pref[:, item_cat]                       # 不存全矩阵，采样时按需算

interactions = []
ts_start, ts_end = 1672502400, 1767225600  # 2023-01-01 ~ 2026-01-01
# 用户交互数：90% 主体(均值15) + 10% 低活跃冷启动群体（稀疏场景）
n_per_user = np.where(rng.random(N_USERS) < 0.90,
                      np.clip(np.round(rng.normal(15, 7, N_USERS)), 2, None),
                      np.clip(np.round(rng.normal(6, 3, N_USERS)), 1, None)).astype(int)

# 物品采样：Gumbel-Top-K 无放回采样，热度^0.85 × 品类亲和 × 作物匹配
N_INTERACTIONS_TARGET = 150_000
scale = N_INTERACTIONS_TARGET / n_per_user.sum()
n_per_user = np.maximum(1, np.round(n_per_user*scale).astype(int))
pw = np.power(pop_weights*N_ITEMS, 0.85)
log_pw = np.log(pw)
CROPS_IDX = {c: k for k, c in enumerate(CROPS)}
item_crop_idx = np.array([CROPS_IDX[it["crop"]] for it in items])
user_crop_idx = np.array([CROPS_IDX[u["crop_type"]] for u in users])
max_n = int(n_per_user.max())
u_list, i_list = [], []
B = 256
for s in range(0, N_USERS, B):
    e = min(s+B, N_USERS)
    nb = e - s
    aff = user_pref[s:e][:, item_cat]                     # (nb, I)
    crop_bonus = np.where(item_crop_idx[None, :] == user_crop_idx[s:e][:, None], 6.0, 1.0)
    logw = np.log(np.maximum(aff, 1e-9)) + log_pw[None, :] + np.log(crop_bonus)
    g = rng.gumbel(0, 1, (nb, N_ITEMS))
    keys = logw + g
    top = np.argpartition(-keys, max_n, axis=1)[:, :max_n]  # 每用户取前 max_n 候选
    for r in range(nb):
        k = n_per_user[s+r]
        u_list += [s+r]*k
        i_list += top[r, :k].tolist()
u_ids = np.array(u_list); chosen_items = np.array(i_list)

btypes = rng.choice(["click","collect","purchase","rating","skip"], len(u_ids),
                    p=[0.55,0.15,0.12,0.10,0.08])
ratings = np.where(btypes=="rating", rng.integers(1,6,len(u_ids)), 0)
tss = rng.integers(ts_start, ts_end, len(u_ids))

interactions = [{"user_id":int(u),"item_id":int(i),"behavior_type":btypes[k],
                 "rating":int(ratings[k]),"timestamp":int(tss[k])}
                for k,(u,i) in enumerate(zip(u_ids, chosen_items))]

# ---------- 4. 社交图（50k 边，10% 噪声）----------
user_arr = np.arange(N_USERS)
edges = set()
n_real = int(N_SOCIAL_TARGET*(1-NOISE_RATIO))
pref_sim = user_pref @ user_pref.T  # 太大! 10000x10000=100M floats=800MB 不可行
# ---------- 4. 社交图（按作物社群采样，10% 随机噪声边）----------
edges = set()
n_real = int(N_SOCIAL_TARGET * (1 - NOISE_RATIO))
REL_TYPES = ["FRIEND_OF", "COOPERATIVE", "TECH_GROUP"]
crop_groups = {}
for u in users:
    crop_groups.setdefault(u["crop_type"], []).append(u["user_id"])
group_list = list(crop_groups.values())
# 85% 同作物社群, 15% 跨社群（培训/合作社等弱关联）
while len(edges) < n_real:
    batch = min(20000, n_real - len(edges) + 2000)
    same = rng.random(batch) < 0.85                     # 逐边决定同/跨社群
    gsel = rng.integers(0, len(group_list), batch)
    a = np.empty(batch, dtype=np.int64); b = np.empty(batch, dtype=np.int64)
    for gi in np.unique(gsel[same]):
        m = same & (gsel == gi)
        g = group_list[gi]
        a[m] = rng.choice(g, m.sum()); b[m] = rng.choice(g, m.sum())
    na = (~same).sum()
    a[~same] = rng.integers(0, N_USERS, na); b[~same] = rng.integers(0, N_USERS, na)
    for x, y in zip(a.tolist(), b.tolist()):
        if x != y:
            edges.add((min(x,y), max(x,y)))
            if len(edges) >= n_real: break

edges = list(edges)[:n_real]
n_noise = int(N_SOCIAL_TARGET * NOISE_RATIO)
noise_edges = set()
while len(noise_edges) < n_noise:
    a, b = rng.integers(0, N_USERS, 2)
    if a != b:
        p = (min(a,b), max(a,b))
        if p not in edges: noise_edges.add(p)
noise_edges = list(noise_edges)

social = [{"src":int(a),"dst":int(b),"rel":REL_TYPES[int(rng.integers(0,3))],"is_noise":0} for a,b in edges[:n_real]]
social += [{"src":int(a),"dst":int(b),"rel":REL_TYPES[int(rng.integers(0,3))],"is_noise":1} for a,b in noise_edges]
rng.shuffle(social)

# ---------- 5. 农业知识图谱（实体 + 12 种关系，5000 三元组）----------
ent_id = 0
entities = {}
def add_ent(name, etype):
    global ent_id
    entities[name] = {"entity_id": ent_id, "name": name, "etype": etype}
    ent_id += 1

for c in CROPS: add_ent(c, "Crop")
pests = [f"{c}常见病虫害{['A','B'][i%2]}" for c in CROPS[:10] for i in range(2)]
for p in pests: add_ent(p, "Pest")
for n in ["复合肥","尿素","磷酸二铵","钾肥","有机肥","水溶肥"]: add_ent(n, "Fertilizer")
for n in ["杀菌剂","杀虫剂","除草剂","生物农药","植物生长调节剂"]: add_ent(n, "Pesticide")
for n in ["嫁接技术","滴灌技术","大棚栽培","病虫害绿色防控","测土配方施肥","节水灌溉","无人机植保","有机肥替代","轮作休耕","修剪整形"]:
    add_ent(n, "Technique")
for it in items[:1000]: add_ent(f"ITEM_{it['item_id']}", "Product")

REL = ["SUSCEPTIBLE_TO","CONTROL_BY","APPLIES_TO","REQUIRES","SUITABLE_FOR","GROWN_IN",
       "RELATED_CROP","BELONGS_TO","HAS_TECHNIQUE","NEED_FERTILIZER","PREVENTED_BY","SIMILAR_TO"]
crop_ids = [e["entity_id"] for e in entities.values() if e["etype"]=="Crop"]
pest_ids = [e["entity_id"] for e in entities.values() if e["etype"]=="Pest"]
fert_ids = [e["entity_id"] for e in entities.values() if e["etype"]=="Fertilizer"]
pesti_ids = [e["entity_id"] for e in entities.values() if e["etype"]=="Pesticide"]
tech_ids = [e["entity_id"] for e in entities.values() if e["etype"]=="Technique"]
prod_ids = [e["entity_id"] for e in entities.values() if e["etype"]=="Product"]
prov_ents = []
for p in PROVINCES[:15]:
    add_ent(p, "Region"); prov_ents.append(entities[p]["entity_id"])

triples = []
def t(h, r, tail): triples.append({"h":h,"r":r,"t":tail})
for c in crop_ids:
    for p in rng.choice(pest_ids, 2, replace=False): t(c, "SUSCEPTIBLE_TO", int(p))
    for f in rng.choice(fert_ids, 2, replace=False): t(c, "NEED_FERTILIZER", int(f))
    for te in rng.choice(tech_ids, 2, replace=False): t(c, "REQUIRES", int(te))
    t(c, "GROWN_IN", int(rng.choice(prov_ents)))
for p in pest_ids:
    t(p, "CONTROL_BY", int(rng.choice(pesti_ids)))
    t(p, "PREVENTED_BY", int(rng.choice(tech_ids)))
for f in fert_ids: t(f, "APPLIES_TO", int(rng.choice(crop_ids)))
for pid in prod_ids:
    t(pid, "RELATED_CROP", int(rng.choice(crop_ids)))
    t(pid, "BELONGS_TO", int(rng.choice(crop_ids)))
    t(pid, "SUITABLE_FOR", int(rng.choice(prov_ents)))
# 补足到 5000
pools = [(crop_ids,tech_ids,"HAS_TECHNIQUE"),(crop_ids,crop_ids,"SIMILAR_TO"),
         (prod_ids,tech_ids,"REQUIRES"),(pest_ids,fert_ids,"CONTROL_BY")]
while len(triples) < 5000:
    a,b,r = pools[rng.integers(0,len(pools))]
    t(int(rng.choice(a)), r, int(rng.choice(b)))
triples = triples[:5000]

# ---------- 6. 物品-实体链接（CKG 用）----------
item2ent = {int(p.split("_")[1]): entities[p]["entity_id"] for p in entities if p.startswith("ITEM_")}
# 其余物品链接到其作物实体
crop_ent = {name: entities[name]["entity_id"] for name in CROPS}
item_links = []
for it in items:
    if it["item_id"] in item2ent:
        item_links.append({"item_id": it["item_id"], "entity_id": item2ent[it["item_id"]]})
    else:
        item_links.append({"item_id": it["item_id"], "entity_id": crop_ent[it["crop"]]})

# ---------- 7. 统计修正与导出 ----------
inter_cnt = np.zeros(N_ITEMS, dtype=int)
for x in interactions: inter_cnt[x["item_id"]] += 1
for i, it in enumerate(items):
    it["is_long_tail"] = int(inter_cnt[i] < 5)
    it["popularity"] = int(inter_cnt[i])

user_cnt = np.zeros(N_USERS, dtype=int)
for x in interactions: user_cnt[x["user_id"]] += 1
cold_users = int((user_cnt < 20).sum())

for u in users: u.pop("pref")

meta = {
  "users": N_USERS, "items": N_ITEMS, "interactions": len(interactions),
  "density": round(len(interactions)/(N_USERS*N_ITEMS)*100, 2),
  "social_edges": len(social), "noise_edges": len(noise_edges),
  "noise_ratio": round(len(noise_edges)/len(social)*100, 2),
  "kg_entities": ent_id, "kg_triples": len(triples), "kg_relations": 12,
  "cold_start_users": cold_users, "cold_ratio": round(cold_users/N_USERS*100, 2),
  "long_tail_items": int((inter_cnt < 5).sum()),
  "long_tail_ratio": round(float((inter_cnt<5).mean())*100, 2),
  "median_user_interactions": int(np.median(user_cnt)),
  "mean_user_interactions": round(float(user_cnt.mean()),1),
}
with open(f"{OUT}/meta.json","w") as f: json.dump(meta,f,ensure_ascii=False,indent=1)
with open(f"{OUT}/users.json","w") as f: json.dump(users,f,ensure_ascii=False)
with open(f"{OUT}/items.json","w") as f: json.dump(items,f,ensure_ascii=False)
with open(f"{OUT}/interactions.json","w") as f: json.dump(interactions,f)
with open(f"{OUT}/social.json","w") as f: json.dump(social,f)
with open(f"{OUT}/kg_entities.json","w") as f: json.dump(list(entities.values()),f,ensure_ascii=False)
with open(f"{OUT}/kg_triples.json","w") as f: json.dump(triples,f)
with open(f"{OUT}/item_links.json","w") as f: json.dump(item_links,f)
np.save(f"{OUT}/user_pref.npy", user_pref)
print(json.dumps(meta, ensure_ascii=False, indent=1))
