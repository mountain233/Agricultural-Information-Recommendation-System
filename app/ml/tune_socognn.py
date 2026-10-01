# -*- coding: utf-8 -*-
"""SoCoGNN 变体调参实验"""
import sys, time, json
sys.path.insert(0, "/mnt/agents/output/app/ml")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.sparse as sp
from common import *

users, items, inter, social, kg_ent, kg_tri, links = load_all()
NU, NI = len(users), len(items)
train, val, test, _ = split_interactions(inter)
adj_ui = build_interaction_adj(NU, NI, train)
adj_soc, _ = build_social_adj(NU, social)
rows, cols = [], []
for u, its in train.items():
    for i in its: rows.append(i); cols.append(u)
M = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(NI, NU))
deg = np.asarray(M.sum(1)).flatten(); deg[deg == 0] = 1
item_user_coo = to_sparse_tensor((sp.diags(1.0/deg) @ M).tocoo())
POS_U = np.concatenate([np.full(len(v), u) for u, v in train.items()])
POS_I = np.concatenate([v for u, v in train.items()])
PUT, PIT = torch.from_numpy(POS_U).long(), torch.from_numpy(POS_I).long()
rng2 = np.random.default_rng(0)
def neg(): return torch.from_numpy(rng2.integers(0, NI, len(PUT)))
def bprl(ue, ie, ng):
    eu, ei, ej = ue[PUT], ie[PIT], ie[ng]
    return -F.logsigmoid((eu*ei).sum(1)-(eu*ej).sum(1)).mean() + 1e-4*(eu.pow(2).sum()+ei.pow(2).sum()+ej.pow(2).sum())/len(ng)

class SoCoGNN_V(nn.Module):
    """norm_mode: 0 不归一化 1 归一化 2 残差融合 3 残差+归一化"""
    def __init__(self, L=3, tau=0.2, dim=DIM, norm_mode=0):
        super().__init__()
        self.L, self.tau, self.norm_mode = L, tau, norm_mode
        self.user_emb = nn.Parameter(torch.randn(NU, dim)*0.05)
        self.item_emb = nn.Parameter(torch.randn(NI, dim)*0.05)
        self.gate_int = nn.Linear(dim, dim); self.gate_soc = nn.Linear(dim, dim)
        self.att_v = nn.Linear(dim*2, 2)
        self.gate_int_i = nn.Linear(dim, dim); self.att_vi = nn.Linear(dim*2, 2)
    def forward(self):
        Eu, Ei = self.user_emb, self.item_emb
        us, ss, ins, iss = [Eu], [Eu], [Ei], [Ei]
        for l in range(self.L):
            E_new = torch.sparse.mm(adj_ui, torch.cat([Eu, Ei]))
            u_int, i_int = E_new[:NU], E_new[NU:]
            u_soc = torch.sparse.mm(adj_soc, Eu)
            i_soc = torch.sparse.mm(item_user_coo, u_soc)
            u_int_g = u_int*torch.sigmoid(self.gate_int(u_int))
            u_soc_g = u_soc*torch.sigmoid(self.gate_soc(u_soc))
            if self.norm_mode == 1:
                u_int_g = F.normalize(u_int_g, dim=1); u_soc_g = F.normalize(u_soc_g, dim=1)
            a = torch.softmax(self.att_v(torch.cat([u_int_g, u_soc_g], 1)), 1)
            fu = a[:, :1]*u_int_g + a[:, 1:]*u_soc_g
            i_int_g = i_int*torch.sigmoid(self.gate_int_i(i_int))
            ai = torch.softmax(self.att_vi(torch.cat([i_int_g, i_soc], 1)), 1)
            fi = ai[:, :1]*i_int_g + ai[:, 1:]*i_soc
            if self.norm_mode >= 2:            # 残差连接防止逐层收缩
                Eu = Eu + fu; Ei = Ei + fi
                if self.norm_mode == 3:
                    Eu = F.normalize(Eu, dim=1); Ei = F.normalize(Ei, dim=1)
            else:
                Eu, Ei = fu, fi
            us.append(u_int); ss.append(u_soc); ins.append(i_int); iss.append(i_soc)
        st = lambda xs: torch.stack(xs).mean(0)
        return (st(us)+st(ss))/2, st(ins), st(us), st(ss), st(ins), st(iss)
    def cva(self, u_int, u_soc, i_int, i_soc, bu, bi):
        def nce(a, b):
            a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
            sim = a@b.T/self.tau; lb = torch.arange(a.size(0))
            return (F.cross_entropy(sim, lb)+F.cross_entropy(sim.T, lb))/2
        return nce(u_int[bu], u_soc[bu]) + nce(i_int[bi], i_soc[bi])

def run(tag, lr, cva_w, norm_mode, epochs=20, no_social=False, tau=0.2, cva_bs=1024):
    torch.manual_seed(0)
    m = SoCoGNN_V(norm_mode=norm_mode, tau=tau); opt = torch.optim.Adam(m.parameters(), lr=lr)
    best = 0
    for ep in range(epochs):
        if no_social:
            global adj_soc
            saved = adj_soc; adj_soc = torch.sparse_coo_tensor(torch.zeros(2,0).long(), [], (NU,NU)).coalesce()
        uf, itf, u_int, u_soc, i_int, i_soc = m()
        if no_social: adj_soc = saved
        loss = bprl(uf, itf, neg())
        if cva_w > 0:
            loss = loss + cva_w*m.cva(u_int, u_soc, i_int, i_soc,
                   torch.from_numpy(rng2.integers(0, NU, cva_bs)), torch.from_numpy(rng2.integers(0, NI, cva_bs)))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4:
            with torch.no_grad(): uf, itf, *_ = m()
            r = evaluate(uf.numpy(), itf.numpy(), train, val)["R20"]
            best = max(best, r)
            print(f"{tag} ep{ep+1} loss={loss.item():.3f} R20={r:.4f}", flush=True)
    print(f"{tag} BEST={best:.4f}", flush=True)

if __name__ == "__main__":
    run("cva.005-tau1 ", 1e-3, 0.005, 2, tau=1.0)
    run("cva.002-tau1 ", 1e-3, 0.002, 2, tau=1.0)
