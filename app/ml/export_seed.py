# -*- coding: utf-8 -*-
"""后处理：修正净化统计为接收者内相对权重口径，并生成种子数据文件"""
import json, os
import numpy as np
from collections import defaultdict

ART = os.path.join(os.path.dirname(__file__), "artifacts")
OUT = os.path.join(os.path.dirname(__file__), "..", "db", "seed-data")
os.makedirs(OUT, exist_ok=True)

social = json.load(open(f"{ART}/social.json"))
ew = json.load(open(f"{ART}/sodra_edges.json"))
metrics = json.load(open(f"{ART}/metrics.json"))
meta = json.load(open(f"{ART}/meta.json"))
items = json.load(open(f"{ART}/items.json"))
users = json.load(open(f"{ART}/users.json"))
inter = json.load(open(f"{ART}/interactions.json"))

# ---- 净化统计：接收者内相对权重（消除度分布干扰）----
noise_flag = {}
for e in social:
    noise_flag[(e["src"], e["dst"])] = e["is_noise"]
    noise_flag[(e["dst"], e["src"])] = e["is_noise"]
src, dst, w, z = ew["src"], ew["dst"], np.array(ew["w"]), np.array(ew["z"])
flags = np.array([noise_flag.get((s, d), 0) for s, d in zip(src, dst)])

by_recv = defaultdict(lambda: {"real": [], "noise": []})
for k in range(len(src)):
    key = "noise" if flags[k] == 1 else "real"
    by_recv[dst[k]][key].append(w[k])

rel_ratios, recv_noise_w, recv_real_w = [], [], []
for u, d in by_recv.items():
    tot = sum(d["real"]) + sum(d["noise"])
    if tot <= 0: continue
    # 噪声边的权重占比 vs 数量占比
    n_w = sum(d["noise"]); n_c = len(d["noise"]); tot_c = len(d["real"]) + n_c
    if n_c > 0:
        recv_noise_w.append(n_w / tot); recv_real_w.append(1 - n_w / tot)
        rel_ratios.append((n_w / tot) / (n_c / tot_c))
rel_ratios = np.array(rel_ratios)
purify = {
    "total_edges": int(len(flags)), "noise_edges": int(flags.sum()),
    "noise_count_ratio": round(float(flags.mean()) * 100, 2),
    "noise_weight_ratio": round(float(np.mean(recv_noise_w)) * 100, 2),
    "relative_compression": round(float(np.mean(rel_ratios)), 3),   # <1 表示噪声边权重被压缩
    "noise_mean_w": float(w[flags == 1].mean()), "real_mean_w": float(w[flags == 0].mean()),
    "noise_mean_z": float(z[flags == 1].mean()), "real_mean_z": float(z[flags == 0].mean()),
    "noise_pruned_ratio": round(float((z[flags == 1] < 0.1).mean()) * 100, 1),
    "real_pruned_ratio": round(float((z[flags == 0] < 0.1).mean()) * 100, 1),
    "weight_bins": None,
}
# 权重分布直方图（真实 vs 噪声），供前端可视化
bins = np.linspace(0, max(w.max(), 1e-6), 21)
purify["weight_bins"] = {
    "edges": bins.round(4).tolist(),
    "noise": np.histogram(w[flags == 1], bins)[0].tolist(),
    "real": np.histogram(w[flags == 0], bins)[0].tolist(),
}
metrics["purify"] = purify

# ---- 场景标签与调度规则（论文 6.3.2）----
user_cnt = np.zeros(len(users), dtype=int)
for x in inter: user_cnt[x["user_id"]] += 1
soc_cnt = np.zeros(len(users), dtype=int); noise_cnt = np.zeros(len(users), dtype=int)
for e in social:
    soc_cnt[e["src"]] += 1; soc_cnt[e["dst"]] += 1
    if e["is_noise"]: noise_cnt[e["src"]] += 1; noise_cnt[e["dst"]] += 1

def scene_of(u):
    c = user_cnt[u]
    if c < 20: return "cold_start"
    if soc_cnt[u] >= 8 and noise_cnt[u] >= 2 and c < 40: return "noisy_social"
    if c >= 40: return "knowledge_rich"
    return "normal"

SCENE_W = {"cold_start": (0.6, 0.25, 0.15), "noisy_social": (0.15, 0.6, 0.25),
           "knowledge_rich": (0.15, 0.25, 0.6), "normal": (1/3, 1/3, 1/3)}
SCENE_ALGO = {"cold_start": "SoCoGNN", "noisy_social": "SoDRA", "knowledge_rich": "PESatNet", "normal": "Fusion"}

for u in users:
    uid = u["user_id"]
    u["interaction_count"] = int(user_cnt[uid])
    u["social_edge_count"] = int(soc_cnt[uid])
    u["noise_edge_count"] = int(noise_cnt[uid])
    u["is_cold_start"] = int(user_cnt[uid] < 20)
    u["scene"] = scene_of(uid)

# ---- 嵌入导出（4 位小数压缩体积）----
emb = {}
for name in ["socognn", "sodra", "pesatnet"]:
    emb[name + "_user"] = np.load(f"{ART}/emb/{name}_user.npy").round(4)
    emb[name + "_item"] = np.load(f"{ART}/emb/{name}_item.npy").round(4)

def quant(v):
    s = float(np.abs(v).max()) / 127.0 or 1e-9
    return {"s": round(s, 8), "q": np.round(v / s).astype(np.int8).tolist()}

emb_rows = []
for i in range(len(users)):
    emb_rows.append({"entity_type": "user", "entity_id": i,
                     "socognn": quant(emb["socognn_user"][i]),
                     "sodra": quant(emb["sodra_user"][i]),
                     "pesatnet": quant(emb["pesatnet_user"][i])})
for i in range(len(items)):
    emb_rows.append({"entity_type": "item", "entity_id": i,
                     "socognn": quant(emb["socognn_item"][i]),
                     "sodra": quant(emb["sodra_item"][i]),
                     "pesatnet": quant(emb["pesatnet_item"][i])})

# ---- 交互抽样（每用户最多 8 条，行为时间线展示用）----
import itertools
by_user = defaultdict(list)
for x in sorted(inter, key=lambda x: -x["timestamp"]):
    if len(by_user[x["user_id"]]) < 8: by_user[x["user_id"]].append(x)
inter_sample = list(itertools.chain.from_iterable(by_user.values()))

# 社交边权重映射回无向边（取双向均值）
w_map = defaultdict(list)
z_map = defaultdict(list)
for k in range(len(src)):
    key = (min(src[k], dst[k]), max(src[k], dst[k]))
    w_map[key].append(float(w[k])); z_map[key].append(float(z[k]))
for e in social:
    key = (min(e["src"], e["dst"]), max(e["src"], e["dst"]))
    e["weight"] = round(float(np.mean(w_map[key])), 6)
    e["gate"] = round(float(np.mean(z_map[key])), 6)


# 修正物品名称：名称中的作物词与 crop 字段对齐（生成器遗留不一致）
ALIAS = {"苹果":"苹果","葡萄":"葡萄","猕猴桃":"猕猴桃","橘":"柑橘","柑":"柑橘","橙":"柑橘","柚":"柑橘",
         "番茄":"番茄","辣椒":"辣椒","彩椒":"辣椒","土豆":"马铃薯","马铃薯":"马铃薯","玉米":"玉米",
         "大豆":"大豆","水稻":"水稻","大米":"水稻","小麦":"小麦","面粉":"小麦","花生":"花生",
         "甘蔗":"甘蔗","茶叶":"茶叶","棉花":"棉花"}
def implied_crops(base):
    return {v for k, v in ALIAS.items() if k in base}
# 每个 (类别,作物) 收集一个可用品名池
pool = defaultdict(set)
for it in items:
    base = it["name"].split("·")[0].split("(")[0]
    imp = implied_crops(base)
    if imp:
        for c in imp: pool[(it["category"], c)].add(base)
ctr = defaultdict(int)
for it in items:
    base = it["name"].split("·")[0].split("(")[0]
    imp = implied_crops(base)
    if imp and it["crop"] not in imp:
        cands = sorted(pool.get((it["category"], it["crop"]), []))
        if cands:
            nb = cands[ctr[(it["category"], it["crop"])] % len(cands)]
            ctr[(it["category"], it["crop"])] += 1
        else:
            nb = f"优质{it['crop']}" if it["category"] != "农资" else f"{it['crop']}专用农资"
        it["name"] = (f"{nb}·{it['origin']}产" if it["category"] != "农资"
                      else f"{nb}({it['origin']}直供)")

kg_ent = json.load(open(f"{ART}/kg_entities.json"))
kg_tri = json.load(open(f"{ART}/kg_triples.json"))
links = json.load(open(f"{ART}/item_links.json"))
item2ent = {l["item_id"]: l["entity_id"] for l in links}
for it in items: it["kg_entity_id"] = item2ent.get(it["item_id"])

def dump(name, obj):
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(name, len(obj) if hasattr(obj, "__len__") else "doc", f"{os.path.getsize(f'{OUT}/{name}.json')/1e6:.1f}MB")

dump("users", users); dump("items", items); dump("interactions", inter_sample)
dump("social_edges", social); dump("kg_entities", kg_ent); dump("kg_triples", kg_tri)
dump("embeddings", emb_rows)
stats = {"metrics": metrics["metrics"], "curves": metrics["curves"], "purify": purify,
         "dataset_meta": meta, "scene_weights": {k: list(v) for k, v in SCENE_W.items()},
         "scene_algo": SCENE_ALGO,
         "cold_coverage": metrics["cold_coverage"], "long_tail_exposure": metrics["long_tail_exposure"],
         "train_seconds": metrics["train_seconds"]}
# 全量用户历史（在线推荐时过滤已交互物品）
uh = defaultdict(list)
for x in inter:
    if x["behavior_type"] != "skip": uh[str(x["user_id"])].append(x["item_id"])
stats["user_history"] = dict(uh)
dump("model_stats", stats)
print("done")
