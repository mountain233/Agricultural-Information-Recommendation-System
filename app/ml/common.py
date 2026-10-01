# -*- coding: utf-8 -*-
"""公共模块：数据加载、图构建、评估指标
对应论文 6.5.2：8:1:1 划分，Precision@10 / Recall@20 / NDCG@10
"""
import json, os
import numpy as np
import torch
import scipy.sparse as sp

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DIM = 64
DEV = "cpu"

def load_all():
    users = json.load(open(f"{ART}/users.json"))
    items = json.load(open(f"{ART}/items.json"))
    inter = json.load(open(f"{ART}/interactions.json"))
    social = json.load(open(f"{ART}/social.json"))
    kg_ent = json.load(open(f"{ART}/kg_entities.json"))
    kg_tri = json.load(open(f"{ART}/kg_triples.json"))
    links = json.load(open(f"{ART}/item_links.json"))
    return users, items, inter, social, kg_ent, kg_tri, links

def split_interactions(inter, seed=42):
    """按用户 8:1:1 随机划分；skip 行为不计入正样本"""
    rng = np.random.default_rng(seed)
    pos = [x for x in inter if x["behavior_type"] != "skip"]
    by_user = {}
    for x in pos:
        by_user.setdefault(x["user_id"], []).append(x["item_id"])
    train, val, test = {}, {}, {}
    for u, its in by_user.items():
        its = np.array(its)
        rng.shuffle(its)
        n = len(its)
        n_tr = max(1, int(n*0.8)); n_va = max(1, int(n*0.1))
        train[u] = its[:n_tr]
        val[u] = its[n_tr:n_tr+n_va]
        test[u] = its[n_tr+n_va:]
    return train, val, test, by_user

def to_sparse_tensor(coo):
    i = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    v = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(i, v, coo.shape).coalesce()

def build_interaction_adj(n_users, n_items, train):
    """LightGCN 归一化邻接：D^-1/2 A D^-1/2，节点 = 用户+物品"""
    rows, cols = [], []
    for u, its in train.items():
        for i in its:
            rows += [u, n_users + i]; cols += [n_users + i, u]
    n = n_users + n_items
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(1)).flatten(); deg[deg == 0] = 1
    dinv = np.power(deg, -0.5)
    A = sp.diags(dinv) @ A @ sp.diags(dinv)
    return to_sparse_tensor(A.tocoo())

def build_social_adj(n_users, social):
    rows, cols = [], []
    for e in social:
        a, b = e["src"], e["dst"]
        rows += [a, b]; cols += [b, a]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_users))
    deg = np.asarray(A.sum(1)).flatten(); deg[deg == 0] = 1
    dinv = np.power(deg, -0.5)
    A = sp.diags(dinv) @ A @ sp.diags(dinv)
    return to_sparse_tensor(A.tocoo()), A

@torch.no_grad()
def evaluate(user_emb, item_emb, train, test, users_subset=None, ks=(10, 20)):
    """全排序评估：返回 Precision@10, Recall@20, NDCG@10"""
    ue = torch.from_numpy(user_emb).float(); ie = torch.from_numpy(item_emb).float()
    test_u = [u for u in test if len(test[u]) > 0]
    if users_subset is not None:
        ss = set(users_subset); test_u = [u for u in test_u if u in ss]
    if not test_u: return {"P10": 0, "R20": 0, "N10": 0, "n": 0}
    P10, R20, N10 = [], [], []
    B = 512
    for s in range(0, len(test_u), B):
        chunk = test_u[s:s+B]
        scores = ue[chunk] @ ie.T                            # (nb, I)
        for r, u in enumerate(chunk):
            tr = train.get(u)
            if tr is not None and len(tr): scores[r, torch.from_numpy(tr)] = -1e9
        top = scores.topk(20, dim=1).indices.numpy()
        for r, u in enumerate(chunk):
            truth = set(test[u].tolist())
            rec = top[r]
            p10 = len(set(rec[:10]) & truth) / 10
            r20 = len(set(rec[:20]) & truth) / len(truth)
            dcg = sum(1/np.log2(i+2) for i, it in enumerate(rec[:10]) if it in truth)
            idcg = sum(1/np.log2(i+2) for i in range(min(len(truth), 10)))
            P10.append(p10); R20.append(r20); N10.append(dcg/idcg if idcg else 0)
    return {"P10": float(np.mean(P10)), "R20": float(np.mean(R20)),
            "N10": float(np.mean(N10)), "n": len(test_u)}

def bpr_batch(user_emb_f, item_emb_f, triplets, l2=1e-4):
    """triplets: (u, i_pos, i_neg) tensors"""
    u, i, j = triplets
    ue, ie, je = user_emb_f(u), item_emb_f(i), item_emb_f(j)
    x = (ue*ie).sum(1) - (ue*je).sum(1)
    loss = -torch.log(torch.sigmoid(x) + 1e-10).mean()
    reg = (ue.norm(2).pow(2) + ie.norm(2).pow(2) + je.norm(2).pow(2)) / len(u)
    return loss + l2*reg

def sample_triplets(train, n, n_items, rng):
    us = list(train.keys())
    rows = np.random.default_rng(rng).choice(us, n, replace=True)
    out_u, out_i, out_j = [], [], []
    for u in rows:
        its = train[u]
        i = its[rng.integers(0, len(its))]
        j = rng.integers(0, n_items)
        while j in set(its.tolist()):
            j = rng.integers(0, n_items)
        out_u.append(u); out_i.append(i); out_j.append(j)
    return (torch.tensor(out_u), torch.tensor(out_i), torch.tensor(out_j))
