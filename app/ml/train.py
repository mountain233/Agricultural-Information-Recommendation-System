# -*- coding: utf-8 -*-
"""训练流水线：训练 SoCoGNN / SoDRA / PESatNet，评估并导出嵌入与指标
对应论文 6.3 算法融合策略与 6.5 系统测试
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from common import (ART, DIM, load_all, split_interactions,
                    build_interaction_adj, evaluate)
from models import SoCoGNN, SoDRA, PESatNet

rng = np.random.default_rng(42)
torch.manual_seed(42)
users, items, inter, social, kg_ent, kg_tri, links = load_all()
NU, NI = len(users), len(items)
NE = len(kg_ent)
train, val, test, by_user = split_interactions(inter)
user_cnt = np.array([len(train.get(u, [])) for u in range(NU)])
cold_users = np.where(user_cnt < 20)[0]
noise_dst = set(e["dst"] for e in social if e["is_noise"] == 1) | set(e["src"] for e in social if e["is_noise"] == 1)
noisy_users = np.array(sorted(noise_dst))

adj_ui = build_interaction_adj(NU, NI, train)
# 有向社交边（双向），供 SoCoGNN GAT 与 SoDRA DSC 使用；[0]=接收者，[1]=邻居
soc_src, soc_dst = [], []
for e in social:
    soc_src += [e["src"], e["dst"]]; soc_dst += [e["dst"], e["src"]]
social_indices = torch.tensor([soc_src, soc_dst], dtype=torch.long)

train_sets = {u: set(v.tolist()) for u, v in train.items()}

# 全量正样本三元组（全批训练：每个 epoch 一次全图前向 + 一次反向，标准 LightGCN 高效实现）
POS_U = np.concatenate([np.full(len(v), u, dtype=np.int64) for u, v in train.items()])
POS_I = np.concatenate([v.astype(np.int64) for v in train.values()])
N_POS = len(POS_U)

def sample_negatives():
    """向量化负采样：碰撞率约 1%，可接受"""
    return torch.from_numpy(rng.integers(0, NI, N_POS))

POS_U_T = torch.from_numpy(POS_U); POS_I_T = torch.from_numpy(POS_I)

def bpr_full(ue, ie, neg, l2=1e-4):
    e_u, e_i, e_j = ue[POS_U_T], ie[POS_I_T], ie[neg]
    x = (e_u*e_i).sum(1) - (e_u*e_j).sum(1)
    loss = -F.logsigmoid(x).mean()
    return loss + l2*(e_u.norm(2).pow(2) + e_i.norm(2).pow(2) + e_j.norm(2).pow(2))/len(neg)

# ---------------- SoCoGNN ----------------
def train_socognn(epochs=40, lr=1e-3, cva_w=0.002):
    model = SoCoGNN(NU, NI, social_indices, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, best_state, patience = [], -1, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        uf, itf = model(adj_ui)
        loss = bpr_full(uf, itf, sample_negatives())
        # 官方 CVA：批内同视图对比（unique 后约 1024 节点）
        bu = torch.from_numpy(np.unique(rng.integers(0, NU, 1024)))
        bi = torch.from_numpy(np.unique(rng.integers(0, NI, 1024)))
        loss = loss + cva_w * model.cva(uf, itf, bu, bi)
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append({"epoch": ep+1, "loss": round(loss.item(), 4)})
        if ep % 5 == 4 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                uf, itf = model(adj_ui)
            m = evaluate(uf.numpy(), itf.numpy(), train, val)
            curve[-1]["val_R20"] = round(m["R20"], 4)
            print(f"  [SoCoGNN] ep{ep+1} loss={loss.item():.4f} val R20={m['R20']:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if m["R20"] > best: best, best_state, patience = m["R20"], (uf.clone(), itf.clone()), 0
            else:
                patience += 1
                if patience >= 4: break
    uf, itf = best_state
    return model, uf.numpy(), itf.numpy(), curve

# ---------------- SoDRA ----------------
def train_sodra(epochs=40, lr=1e-3, l0_w=0.05):
    # 行为锚定先验（交互作物直方图余弦相似度）
    src, dst = soc_src, soc_dst
    edge_index = social_indices
    CROPS = ["柑橘","苹果","水稻","小麦","玉米","茶叶","葡萄","番茄","棉花","马铃薯","猕猴桃","花生","大豆","辣椒","甘蔗"]
    crop_of_item = np.array([CROPS.index(it["crop"]) for it in items])
    histc = np.zeros((NU, len(CROPS)))
    for u, its in train.items(): np.add.at(histc[u], crop_of_item[its], 1)
    hcn = histc / (np.linalg.norm(histc, axis=1, keepdims=True) + 1e-9)
    edge_sim = torch.tensor((hcn[np.array(src)] * hcn[np.array(dst)]).sum(1), dtype=torch.float32)
    model = SoDRA(NU, NI, edge_index, edge_sim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, best_state, patience = [], -1, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        uf, itf = model(adj_ui, training=True)
        loss = bpr_full(uf, itf, sample_negatives()) + l0_w * model.dsc.l0()
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append({"epoch": ep+1, "loss": round(loss.item(), 4)})
        if ep % 5 == 4 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                uf, itf = model(adj_ui, training=False)
            m = evaluate(uf.detach().numpy(), itf.detach().numpy(), train, val)
            curve[-1]["val_R20"] = round(m["R20"], 4)
            print(f"  [SoDRA] ep{ep+1} loss={loss.item():.4f} val R20={m['R20']:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if m["R20"] > best: best, best_state, patience = m["R20"], (uf.detach().clone(), itf.detach().clone()), 0
            else:
                patience += 1
                if patience >= 4: break
    uf, itf = best_state
    # 导出净化后的边权重（推理时期望值）
    model.eval()
    with torch.no_grad():
        w, alpha, z = model.edge_weights(model.user_emb, training=False)
    ew = {"src": src, "dst": dst, "w": w.numpy().round(4).tolist(), "z": z.numpy().round(4).tolist()}
    return model, uf.numpy(), itf.numpy(), curve, ew

# ---------------- PESatNet ----------------
def train_pesatnet(epochs=40, lr=1e-3, cl_w=0.001):
    RELS = ["interact"] + sorted(set(t["r"] for t in kg_tri))
    r2i = {r: k for k, r in enumerate(RELS)}
    n_rels = len(RELS)
    # CKG：每个物品是独立实体（KGAT 惯例），节点空间 = 用户 + KG实体 + 物品实体
    NENT = NE + NI
    item2ent = torch.arange(NE, NE + NI, dtype=torch.long)
    item_crop_ent = torch.tensor([links[i]["entity_id"] for i in range(NI)])
    s_, d_, r_ = [], [], []
    for u, its in train.items():
        for i in its:
            e = NU + NE + i          # 物品实体节点 id
            s_ += [u, e]; d_ += [e, u]
            r_ += [r2i["interact"], n_rels + r2i["interact"]]
    for t in kg_tri:
        h, tail, rel = NU + t["h"], NU + t["t"], t["r"]
        s_ += [h, tail]; d_ += [tail, h]
        r_ += [r2i[rel], n_rels + r2i[rel]]
    for i in range(NI):              # 物品实体 → 作物实体（RELATED_CROP 双向）
        e, c = NU + NE + i, NU + int(item_crop_ent[i])
        s_ += [e, c]; d_ += [c, e]
        r_ += [r2i["RELATED_CROP"], n_rels + r2i["RELATED_CROP"]]
    edge_idx = torch.tensor([s_, d_], dtype=torch.long)
    edge_rel = torch.tensor(r_, dtype=torch.long)
    deg = torch.zeros(NU + NENT).index_add_(0, edge_idx[1], torch.ones(len(s_))).clamp(min=1)
    # 对称归一化（双端度乘积开方），保证传播谱半径稳定
    edge_norm = (deg[edge_idx[0]] * deg[edge_idx[1]]).pow(-0.5)
    model = PESatNet(NU, NENT, n_rels, item2ent)
    # 对比损失上下文：交互 R 与 CKG 节点度（官方按 1/sqrt 归一）
    model.set_context(torch.stack([POS_U_T, POS_I_T]), NI, deg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, best_state, patience = [], -1, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        model.temperature = max(0.35, 0.6 - 0.005 * ep)     # 官方温度退火 0.6→0.35
        uf, ef = model(edge_idx, edge_rel, edge_norm, training=True)
        itf = model.item_embeddings(ef)
        loss = bpr_full(uf, itf, sample_negatives())
        # 官方对比损失：批内用户两两 + 正样本×负样本（在线共现，替代全量相似度矩阵）
        bidx = torch.from_numpy(rng.integers(0, N_POS, 1024))
        bu = torch.unique(POS_U_T[bidx])
        bp = POS_I_T[bidx]
        bn = torch.from_numpy(rng.integers(0, NI, len(bidx)))
        u_cl, i_cl = model.contrastive_loss(bu, bp, bn, uf, itf)
        loss = loss + cl_w * (u_cl + i_cl)
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append({"epoch": ep+1, "loss": round(loss.item(), 4)})
        if ep % 5 == 4 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                uf, ef = model(edge_idx, edge_rel, edge_norm, training=False)   # 推理时剥离 REIM/GRAM
                itf = model.item_embeddings(ef)
            m = evaluate(uf.numpy(), itf.numpy(), train, val)
            curve[-1]["val_R20"] = round(m["R20"], 4)
            print(f"  [PESatNet] ep{ep+1} loss={loss.item():.4f} val R20={m['R20']:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if m["R20"] > best: best, best_state, patience = m["R20"], (uf.clone(), itf.clone()), 0
            else:
                patience += 1
                if patience >= 4: break
    uf, itf = best_state
    return model, uf.numpy(), itf.numpy(), curve

if __name__ == "__main__":
    import sys
    EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    os.makedirs(f"{ART}/emb", exist_ok=True)
    t0 = time.time()
    print("== 训练 SoCoGNN ==", flush=True)
    m1, u1, i1, c1 = train_socognn(epochs=EP)
    print("== 训练 SoDRA ==", flush=True)
    m2, u2, i2, c2, ew = train_sodra(epochs=EP)
    print("== 训练 PESatNet ==", flush=True)
    m3, u3, i3, c3 = train_pesatnet(epochs=EP)

    np.save(f"{ART}/emb/socognn_user.npy", u1); np.save(f"{ART}/emb/socognn_item.npy", i1)
    np.save(f"{ART}/emb/sodra_user.npy", u2);   np.save(f"{ART}/emb/sodra_item.npy", i2)
    np.save(f"{ART}/emb/pesatnet_user.npy", u3); np.save(f"{ART}/emb/pesatnet_item.npy", i3)

    def full_eval(ue, ie, subset=None):
        return evaluate(ue, ie, train, test, users_subset=subset)
    metrics = {
        "SoCoGNN":  {"overall": full_eval(u1, i1), "cold": full_eval(u1, i1, cold_users), "noisy": full_eval(u1, i1, noisy_users)},
        "SoDRA":    {"overall": full_eval(u2, i2), "cold": full_eval(u2, i2, cold_users), "noisy": full_eval(u2, i2, noisy_users)},
        "PESatNet": {"overall": full_eval(u3, i3), "cold": full_eval(u3, i3, cold_users), "noisy": full_eval(u3, i3, noisy_users)},
    }

    # ---- 融合策略（论文 6.3.3）：sigmoid 归一后加权 ----
    def fused_scores(u_ids, weights=(1/3, 1/3, 1/3)):
        def sig(x): return 1/(1+np.exp(-x))
        s1 = sig(u1[u_ids] @ i1.T); s2 = sig(u2[u_ids] @ i2.T); s3 = sig(u3[u_ids] @ i3.T)
        return weights[0]*s1 + weights[1]*s2 + weights[2]*s3

    def eval_fused(weights=(1/3,1/3,1/3), subset=None):
        test_u = [u for u in test if len(test[u]) > 0]
        if subset is not None:
            ss = set(subset.tolist()); test_u = [u for u in test_u if u in ss]
        P10, R20, N10 = [], [], []
        B = 512
        for s in range(0, len(test_u), B):
            chunk = np.array(test_u[s:s+B])
            sc = fused_scores(chunk, weights)
            for r, u in enumerate(chunk):
                tr = train.get(u)
                if tr is not None and len(tr): sc[r, tr] = -1e9
            top = np.argpartition(-sc, 20, axis=1)[:, :20]
            for r, u in enumerate(chunk):
                truth = set(test[u].tolist()); rec = top[r]
                P10.append(len(set(rec[:10]) & truth)/10)
                R20.append(len(set(rec[:20]) & truth)/len(truth))
                dcg = sum(1/np.log2(k+2) for k, itx in enumerate(rec[:10]) if itx in truth)
                idcg = sum(1/np.log2(k+2) for k in range(min(len(truth),10)))
                N10.append(dcg/idcg if idcg else 0)
        return {"P10": float(np.mean(P10)), "R20": float(np.mean(R20)), "N10": float(np.mean(N10)), "n": len(test_u)}

    metrics["Fusion"] = {"overall": eval_fused(), "cold": eval_fused((0.6,0.2,0.2), cold_users),
                         "noisy": eval_fused((0.2,0.6,0.2), noisy_users)}

    # ---- 冷启动覆盖率 & 长尾曝光（论文 6.5.5）----
    crop_items = {}
    for it in items: crop_items.setdefault(it["crop"], set()).add(it["item_id"])
    cold_with_test = [u for u in cold_users if u in test and len(test[u]) > 0]
    covered = 0
    B = 512
    for s in range(0, len(cold_with_test), B):
        chunk = np.array(cold_with_test[s:s+B])
        sc = fused_scores(chunk, (0.6,0.2,0.2))
        for r, u in enumerate(chunk):
            tr = train.get(u)
            if tr is not None and len(tr): sc[r, tr] = -1e9
            top10 = np.argpartition(-sc[r], 10)[:10]
            if len(set(top10.tolist()) & crop_items.get(users[u]["crop_type"], set())) > 0:
                covered += 1
    cold_coverage = covered / max(1, len(cold_with_test))

    long_tail_ids = set(it["item_id"] for it in items if it["is_long_tail"] == 1)
    all_test_u = np.array([u for u in test if len(test[u]) > 0])
    lt_hits, tot_rec = 0, 0
    for s in range(0, len(all_test_u), B):
        chunk = all_test_u[s:s+B]
        sc = fused_scores(chunk)
        for r, u in enumerate(chunk):
            tr = train.get(u)
            if tr is not None and len(tr): sc[r, tr] = -1e9
        top10 = np.argpartition(-sc, 10, axis=1)[:, :10]
        lt_hits += sum(len(set(row.tolist()) & long_tail_ids) for row in top10)
        tot_rec += len(chunk)*10
    long_tail_exposure = lt_hits / max(1, tot_rec)

    # ---- SoDRA 图净化统计 ----
    noise_flag = {}
    for e in social:
        noise_flag[(e["src"], e["dst"])] = e["is_noise"]; noise_flag[(e["dst"], e["src"])] = e["is_noise"]
    w_arr = np.array(ew["w"]); z_arr = np.array(ew["z"])
    flags = np.array([noise_flag.get((s_, d_), 0) for s_, d_ in zip(ew["src"], ew["dst"])])
    purify = {
        "total_edges": len(flags), "noise_edges": int(flags.sum()),
        "noise_pruned_ratio": float((z_arr[flags == 1] < 0.1).mean()),
        "real_pruned_ratio": float((z_arr[flags == 0] < 0.1).mean()),
        "noise_mean_w": float(w_arr[flags == 1].mean()), "real_mean_w": float(w_arr[flags == 0].mean()),
        "sample": [{"src": int(ew["src"][k]), "dst": int(ew["dst"][k]),
                    "w": float(w_arr[k]), "z": float(z_arr[k]), "is_noise": int(flags[k])}
                   for k in rng.choice(len(flags), min(400, len(flags)), replace=False)],
    }

    out = {"metrics": metrics, "curves": {"SoCoGNN": c1, "SoDRA": c2, "PESatNet": c3},
           "cold_coverage": cold_coverage, "long_tail_exposure": long_tail_exposure,
           "purify": purify, "train_seconds": time.time()-t0,
           "n_cold": int(len(cold_users)), "n_noisy": int(len(noisy_users))}
    json.dump(out, open(f"{ART}/metrics.json", "w"), ensure_ascii=False, indent=1)
    json.dump(ew, open(f"{ART}/sodra_edges.json", "w"))
    print(json.dumps({k: v["overall"] for k, v in metrics.items()}, indent=1), flush=True)
    print("cold_coverage:", cold_coverage, "long_tail_exposure:", long_tail_exposure, flush=True)
    print("purify:", {k: v for k, v in purify.items() if k != "sample"}, flush=True)
    print(f"全部完成，耗时 {time.time()-t0:.0f}s", flush=True)
