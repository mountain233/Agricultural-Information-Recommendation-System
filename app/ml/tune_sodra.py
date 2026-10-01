# -*- coding: utf-8 -*-
"""SoDRA 净化效果验证"""
import sys, time
sys.path.insert(0, "/mnt/agents/output/app/ml")
import numpy as np, torch, torch.nn.functional as F
from common import *
import models

users, items, inter, social, kg_ent, kg_tri, links = load_all()
NU, NI = len(users), len(items)
train, val, test, _ = split_interactions(inter)
adj_ui = build_interaction_adj(NU, NI, train)
POS_U = np.concatenate([np.full(len(v), u) for u, v in train.items()])
POS_I = np.concatenate([v for u, v in train.items()])
PUT, PIT = torch.from_numpy(POS_U).long(), torch.from_numpy(POS_I).long()
rng2 = np.random.default_rng(0)
def neg(): return torch.from_numpy(rng2.integers(0, NI, len(PUT)))
def bprl(ue, ie, ng):
    eu, ei, ej = ue[PUT], ie[PIT], ie[ng]
    return -F.logsigmoid((eu*ei).sum(1)-(eu*ej).sum(1)).mean() + 1e-4*(eu.pow(2).sum()+ei.pow(2).sum()+ej.pow(2).sum())/len(ng)

src, dst = [], []
for e in social:
    src += [e["src"], e["dst"]]; dst += [e["dst"], e["src"]]
edge_index = torch.tensor([src, dst], dtype=torch.long)
flags = np.array([e["is_noise"] for e in social for _ in range(2)])  # 双向边同标记

def run(l0_w, init, epochs=30, tag=""):
    torch.manual_seed(0)
    m = models.SoDRA(NU, NI, edge_index)
    with torch.no_grad(): m.hc.log_alpha.fill_(init)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        uf, itf = m(adj_ui, training=True)
        loss = bprl(uf, itf, neg()) + l0_w * m.hc.l0()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            m.eval()
            with torch.no_grad(): uf, itf = m(adj_ui, training=False)
            r = evaluate(uf.numpy(), itf.numpy(), train, val)["R20"]
            w, alpha, z = m.edge_weights(m.user_emb, training=False)
            zn, zr = z.numpy()[flags==1], z.numpy()[flags==0]
            print(f"{tag} ep{ep+1} R20={r:.4f} | z_noise_mean={zn.mean():.3f} z_real_mean={zr.mean():.3f} "
                  f"pruned_noise={(zn<0.5).mean():.3f} pruned_real={(zr<0.5).mean():.3f}", flush=True)

if __name__ == "__main__":
    run(1e-3, 2.0, tag="l0=1e-3")
    run(1e-2, 1.0, tag="l0=1e-2")
