# -*- coding: utf-8 -*-
"""SoDRA 净化诊断：独立重训若干 epoch，观察 beta / 门值 / 接收者内 alpha 对比"""
import numpy as np, torch, torch.nn.functional as F, time, sys
from common import load_all, split_interactions, build_interaction_adj, evaluate
from models import SoDRA

torch.manual_seed(42); rng = np.random.default_rng(42)
users, items, inter, social, kg_ent, kg_tri, links = load_all()
NU, NI = len(users), len(items)
train, val, test, by_user = split_interactions(inter)
adj_ui = build_interaction_adj(NU, NI, train)

src, dst = [], []
for e in social:
    src += [e["src"], e["dst"]]; dst += [e["dst"], e["src"]]
edge_index = torch.tensor([src, dst], dtype=torch.long)
CROPS = ["柑橘","苹果","水稻","小麦","玉米","茶叶","葡萄","番茄","棉花","马铃薯","猕猴桃","花生","大豆","辣椒","甘蔗"]
crop_of_item = np.array([CROPS.index(it["crop"]) for it in items])
histc = np.zeros((NU, len(CROPS)))
for u, its in train.items(): np.add.at(histc[u], crop_of_item[its], 1)
hcn = histc / (np.linalg.norm(histc, axis=1, keepdims=True) + 1e-9)
edge_sim = torch.tensor((hcn[np.array(src)] * hcn[np.array(dst)]).sum(1), dtype=torch.float32)
flags = np.array([e["is_noise"] for e in social for _ in (0, 1)])
print("edge_sim: 噪声 %.3f vs 真实 %.3f" % (edge_sim.numpy()[flags==1].mean(), edge_sim.numpy()[flags==0].mean()))

POS_U = np.concatenate([np.full(len(v), u, dtype=np.int64) for u, v in train.items()])
POS_I = np.concatenate([v.astype(np.int64) for v in train.values()])
POS_U_T = torch.from_numpy(POS_U); POS_I_T = torch.from_numpy(POS_I)
def bpr_full(ue, ie, neg, l2=1e-4):
    e_u, e_i, e_j = ue[POS_U_T], ie[POS_I_T], ie[neg]
    x = (e_u*e_i).sum(1) - (e_u*e_j).sum(1)
    return -F.logsigmoid(x).mean() + l2*(e_u.norm(2).pow(2)+e_i.norm(2).pow(2)+e_j.norm(2).pow(2))/len(neg)

l0_w = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
beta_init = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
model = SoDRA(NU, NI, edge_index, edge_sim)
with torch.no_grad(): model.beta.fill_(beta_init)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"l0_w={l0_w} beta_init={beta_init} epochs={epochs}")
for ep in range(epochs):
    model.train()
    uf, itf = model(adj_ui, training=True)
    loss = bpr_full(uf, itf, torch.from_numpy(rng.integers(0, NI, len(POS_U)))) + l0_w*model.hc.l0()
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            uf, itf = model(adj_ui, training=False)
            m = evaluate(uf.numpy(), itf.numpy(), train, val)
            w, alpha, z = model.edge_weights(model.user_emb, training=False)
            a = alpha.numpy(); zz = z.numpy()
        # 接收者内对比（拥有≥1噪声边且≥1真实边的接收者）
        rel = []
        recv = np.array(src)
        for r in np.unique(recv):
            mask = recv == r
            f = flags[mask]
            if f.sum() >= 1 and (1-f).sum() >= 1:
                rel.append(a[mask][f==1].mean() / (a[mask][f==0].mean() + 1e-9))
        print(f"ep{ep+1} loss={loss.item():.4f} R20={m['R20']:.4f} beta={model.beta.item():.2f} "
              f"z(noise)={zz[flags==1].mean():.3f} z(real)={zz[flags==0].mean():.3f} "
              f"alpha(noise)={a[flags==1].mean():.4f} alpha(real)={a[flags==0].mean():.4f} "
              f"接收者内噪声/真实={np.mean(rel):.3f}", flush=True)
