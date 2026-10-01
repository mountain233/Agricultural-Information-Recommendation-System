# -*- coding: utf-8 -*-
"""三个算法模型实现（依据论文官方代码移植，适配独立训练管线）：
- SoCoGNN（第三章官方代码 model.py）：社交图 GAT 注意力传播 + 交互图 LightGCN 传播
  + Graph_Comb 融合（tanh 变换 + 自门控 + 拼接线性 + L2 归一）+ CVA 对比正则
- SoDRA（第四章官方代码 model(1).py）：DSC 去噪（社交图 GCN 特征 + 双路注意力 MLP
  + Hard Concrete 边门）+ Graph_Comb 双向 Transformer 融合
  ※ 官方代码中 L0 正则作用在未接入前向的 log_alpha 参数上（无梯度路径），此处按论文
    公式 4-4~4-6 的意图改为直接作用于边门 logit；并按论文 4.3「行为锚定」加入可学习
    行为一致性先验 beta·edge_sim。
- PESatNet（第五章官方代码 PESatNet.py）：LightKG 式标量关系编码 CKG 传播
  + REIM(factor=32)/GRAM 逐层实体增强 + 度加权相似度对比损失（温度退火 0.6→0.35）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 64

# ============================ 公共 ============================
class SelfGatingLayer(nn.Module):
    """官方 SelfGatingLayer：x * sigmoid(Linear(x))"""
    def __init__(self, embed_dim):
        super().__init__()
        self.gating_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.gating_linear.weight)
    def forward(self, x):
        return x * torch.sigmoid(self.gating_linear(x))


# ============================ SoCoGNN ============================
class GraphAttentionLayer(nn.Module):
    """官方 GraphAttentionLayer：社交图上的 GAT（对非零边做逐行 softmax）"""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj_indices, n):
        Wh = h @ self.W
        row, col = adj_indices[0], adj_indices[1]
        e = self.leakyrelu((torch.cat([Wh[row], Wh[col]], 1) @ self.a).squeeze(1))
        # 逐行 softmax（官方代码缺 exp，此处为标准 softmax 口径）
        exp_e = torch.exp(e - e.max())
        denom = torch.zeros(n, device=h.device).scatter_add(0, row, exp_e)
        att = exp_e / (denom[row] + 1e-12)
        att = F.dropout(att, self.dropout, training=self.training)
        idx = torch.stack([row, col])
        att_mat = torch.sparse_coo_tensor(idx, att, (n, n)).coalesce()
        return F.elu(torch.sparse.mm(att_mat, Wh))


class GraphComb(nn.Module):
    """官方 Graph_Comb：tanh 变换 + 自门控 + 拼接线性 + L2 归一"""
    def __init__(self, embed_dim):
        super().__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)
        self.self_gating_x = SelfGatingLayer(embed_dim)
        self.self_gating_y = SelfGatingLayer(embed_dim)
    def forward(self, x, y):
        h1 = self.self_gating_x(torch.tanh(self.att_x(x)))
        h2 = self.self_gating_y(torch.tanh(self.att_y(y)))
        return F.normalize(self.comb(torch.cat([h1, h2], 1)), p=2, dim=1)


class CVA(nn.Module):
    """官方 CVA 对比正则（式3-10~3-14 的代码实现：批内同视图 InfoNCE 均匀性项）
    temp=0.1, ureg=ireg=0.5 为官方默认值。"""
    def __init__(self, temp=0.1, ureg=0.5, ireg=0.5):
        super().__init__()
        self.temp, self.ureg, self.ireg = temp, ureg, ireg
    def ssl_loss(self, data, index):
        emb = F.normalize(data[index], p=2, dim=1)
        pos = torch.exp(torch.ones(emb.size(0), device=emb.device) / self.temp)
        all_s = torch.exp((emb @ emb.T) / self.temp).sum(1)
        return (-torch.log(pos / all_s)).mean()
    def forward(self, users_emb, items_emb, user_idx, item_idx):
        return self.ureg * self.ssl_loss(users_emb, user_idx) + \
               self.ireg * self.ssl_loss(items_emb, item_idx)


class SoCoGNN(nn.Module):
    """官方 SoCoGNN（LightGCN 子类）：社交 GAT + 交互 LightGCN + Graph_Comb 逐层融合"""
    def __init__(self, n_users, n_items, social_indices, n_layers=3, dim=DIM):
        super().__init__()
        self.n_users, self.n_items, self.L = n_users, n_items, n_layers
        self.user_emb = nn.Parameter(torch.randn(n_users, dim) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_items, dim) * 0.1)
        self.register_buffer("social_idx", social_indices)  # [2, E] 有向社交边
        self.gat_layer = GraphAttentionLayer(dim, dim, dropout=0.6, alpha=0.2)
        self.graph_comb = GraphComb(dim)
        self.cva = CVA()

    def forward(self, adj_ui):
        all_emb = torch.cat([self.user_emb, self.item_emb], 0)
        embs = [all_emb]
        for _ in range(self.L):
            users_emb, _ = torch.split(all_emb, [self.n_users, self.n_items])
            u_soc = self.gat_layer(users_emb, self.social_idx, self.n_users)
            all_inter = torch.sparse.mm(adj_ui, all_emb)
            u_int, i_next = torch.split(all_inter, [self.n_users, self.n_items])
            u_next = self.graph_comb(u_soc, u_int)
            all_emb = torch.cat([u_next, i_next], 0)
            embs.append(all_emb)
        final = torch.stack(embs, 1).mean(1)
        return torch.split(final, [self.n_users, self.n_items])


# ============================ SoDRA ============================
class DSC(nn.Module):
    """官方 DSC 去噪模块：社交图 GCN 特征 → 双路注意力 logit → Hard Concrete 边门
    （按论文 4.3 加入行为锚定先验 beta·edge_sim）
    edge_index: [2, E]，[0]=接收者，[1]=邻居（传播源），与官方 spmm(denoised_S, users_emb) 一致"""
    def __init__(self, dim, edge_index, edge_sim, n_layers=2):
        super().__init__()
        self.register_buffer("edge_rcv", edge_index[0])   # 接收者
        self.register_buffer("edge_nbr", edge_index[1])   # 邻居（传播源）
        self.register_buffer("edge_sim", edge_sim)
        self.beta_prior = nn.Parameter(torch.tensor(10.0))  # 行为先验强度（可学习）
        self.tau = nn.Parameter(torch.tensor(0.5))         # 信任阈值（可学习）：低于阈值的行为一致性视为可疑
        # GCN 层（Â = D̂^-1/2 (A+I) D̂^-1/2，手工实现以替代 PyG GCNConv）
        self.gcn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        # 注意力 MLP
        self.att_nbs = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.att_self = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        # Hard Concrete 超参（官方取值，gamma 为负下界）
        self.hc_gamma, self.hc_zeta, self.hc_beta = -0.01, 1.0, 1.0
        self.last_logits = None

    def hard_concrete(self, logits, training):
        if training:
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / self.hc_beta)
        else:
            s = torch.sigmoid(logits)
        return torch.clamp(s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma, 0.0, 1.0)

    def l0(self):
        """L0 正则（官方作用于未接入的参数，此处按论文意图作用于实际边门 logit）"""
        if self.last_logits is None:
            return torch.tensor(0.0)
        return torch.sigmoid(
            self.last_logits - self.hc_beta * np.log(-self.hc_gamma / self.hc_zeta)).mean()

    def forward(self, x, adj_hat, training=True):
        h = x
        for lin in self.gcn:
            h = F.relu(torch.sparse.mm(adj_hat, lin(h)))
        f_nbr, f_rcv = h[self.edge_nbr], h[self.edge_rcv]
        logits = (self.att_nbs(f_nbr) + self.att_self(f_rcv)).squeeze(1) \
                 + self.beta_prior * (self.edge_sim - self.tau)   # 行为锚定先验（中心化）
        self.last_logits = logits
        return self.hard_concrete(logits, training)


class TransformerBlock(nn.Module):
    """官方 TransformerBlock：全维 MHA + LayerNorm + FFN(×4, GELU) + Dropout"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
                                 nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value):
        attn_out, _ = self.attn(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
        x = self.norm1(query + self.dropout(attn_out.squeeze(0)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class GraphCombBi(nn.Module):
    """官方 Graph_Comb（SoDRA 版）：自门控 + 双向 Transformer + 融合 + L2 归一"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.self_gating_x = SelfGatingLayer(embed_dim)
        self.self_gating_y = SelfGatingLayer(embed_dim)
        self.x_to_y = TransformerBlock(embed_dim, num_heads, dropout)
        self.y_to_x = TransformerBlock(embed_dim, num_heads, dropout)
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
    def forward(self, x, y):
        x = self.self_gating_x(x)
        y = self.self_gating_y(y)
        x_ = self.x_to_y(x, y, y)   # y 作 K/V，x 作 Q
        y_ = self.y_to_x(y, x, x)
        return F.normalize(self.fusion_layer(torch.cat([x_, y_], -1)), p=2, dim=1)


class SoDRA(nn.Module):
    """官方 SoDRA（LightGCN 子类）：每层用 DSC 净化社交图再传播，Graph_Comb 融合"""
    def __init__(self, n_users, n_items, edge_index, edge_sim=None, n_layers=2, dim=DIM):
        super().__init__()
        self.n_users, self.n_items, self.L = n_users, n_items, n_layers
        self.user_emb = nn.Parameter(torch.randn(n_users, dim) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_items, dim) * 0.1)
        self.register_buffer("edge_rcv", edge_index[0])   # 接收者
        self.register_buffer("edge_nbr", edge_index[1])   # 邻居（传播源）
        n_edges = edge_index.size(1)
        if edge_sim is None:
            edge_sim = torch.zeros(n_edges)
        self.dsc = DSC(dim, edge_index, edge_sim, n_layers=2)
        self.graph_comb = GraphCombBi(dim)
        # GCN 归一化邻接 Â = D̂^-1/2(A+I)D̂^-1/2（静态，供 DSC 特征提取）
        n = n_users
        idx = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]])], 1)  # 对称化
        self_idx = torch.arange(n)
        idx = torch.cat([idx, torch.stack([self_idx, self_idx])], 1)                   # 自环
        deg = torch.zeros(n).index_add_(0, idx[0], torch.ones(idx.size(1)))
        vals = (deg[idx[0]] * deg[idx[1]]).pow(-0.5)
        self.register_buffer("ah_idx", idx)
        self.register_buffer("ah_val", vals)

    def denoise_weights(self, users_emb, training):
        adj_hat = torch.sparse_coo_tensor(self.ah_idx, self.ah_val,
                                          (self.n_users, self.n_users)).coalesce()
        return self.dsc(users_emb, adj_hat, training)

    def forward(self, adj_ui, training=True):
        all_emb = torch.cat([self.user_emb, self.item_emb], 0)
        embs = [all_emb]
        for _ in range(self.L):
            users_emb, _ = torch.split(all_emb, [self.n_users, self.n_items])
            w = self.denoise_weights(users_emb, training)
            S = torch.sparse_coo_tensor(
                torch.stack([self.edge_rcv, self.edge_nbr]), w,
                (self.n_users, self.n_users)).coalesce()   # [接收者, 邻居]
            u_soc = torch.sparse.mm(S, users_emb)
            all_inter = torch.sparse.mm(adj_ui, all_emb)
            u_int, i_next = torch.split(all_inter, [self.n_users, self.n_items])
            u_next = self.graph_comb(u_soc, u_int)
            all_emb = torch.cat([u_next, i_next], 0)
            embs.append(all_emb)
        final = torch.stack(embs, 1).mean(1)
        return torch.split(final, [self.n_users, self.n_items])

    def edge_weights(self, users_emb, training=False):
        """导出推理期边权重：w=拉伸后的确定门值，z=门概率 sigmoid(logits)，alpha=logits"""
        w = self.denoise_weights(users_emb, training)
        logits = self.dsc.last_logits
        z = torch.sigmoid(logits)
        return w, logits, z


# ============================ PESatNet ============================
class REIM(nn.Module):
    """官方 REIM（EMA 坐标注意力移植）：factor=32 分组，(N,dim)→(N,dim,1,1) 特征图"""
    def __init__(self, channels=DIM, factor=32):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        y1 = torch.matmul(x11, x12)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        y2 = torch.matmul(x21, x22)
        weights = (y1 + y2).reshape(b * self.groups, 1, h, w).sigmoid()
        return (group_x * weights).reshape(b, c, h, w)


class GRAM(nn.Module):
    """官方 GRAM：通道注意力 + 空间注意力 + 残差"""
    def __init__(self, channel=DIM, reduction=8, spatial_kernel=5):
        super().__init__()
        reduction = min(reduction, max(1, channel // 2))
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False), nn.Sigmoid())
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        ch = x * self.channel_att(x)
        max_out, _ = torch.max(ch, dim=1, keepdim=True)
        avg_out = torch.mean(ch, dim=1, keepdim=True)
        sp = self.spatial_att(torch.cat([max_out, avg_out], dim=1))
        return x * sp + x


class PESatNet(nn.Module):
    """官方 PESatNet：标量关系编码 CKG 传播 + 逐层 REIM(0.02 残差)/GRAM 实体增强"""
    def __init__(self, n_users, n_entities, n_rels, item2ent, n_layers=3, dim=DIM,
                 mess_dropout=0.1):
        super().__init__()
        self.n_users, self.n_entities, self.L = n_users, n_entities, n_layers
        self.user_emb = nn.Parameter(torch.randn(n_users, dim) * 0.1)
        self.ent_emb = nn.Parameter(torch.randn(n_entities, dim) * 0.1)
        self.rel_w = nn.Parameter(torch.ones(n_rels * 2))   # 标量关系编码（双向）
        self.register_buffer("item2ent", item2ent)
        self.reim = REIM(dim, factor=32)
        self.gram = GRAM(dim, reduction=8, spatial_kernel=5)
        self.channel_dropout = nn.Dropout1d(p=0.2)
        self.mess_dropout = nn.Dropout(p=mess_dropout)
        self.temperature = 0.6

    def reim_refine(self, emb):
        return self.reim(emb.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def gram_refine(self, emb, training):
        m = emb.permute(1, 0).unsqueeze(0).unsqueeze(-1)     # (1,C,N,1)
        m = self.gram(m)
        b, c, h, w = m.shape
        m3 = m.view(b, c, -1)
        if training:
            m3 = self.channel_dropout(m3)                    # 通道 Dropout（训练时）
        m = m3.view(b, c, h, w)
        if training:
            m = m + torch.randn_like(m) * 0.01               # 轻噪声（训练时）
        return m.squeeze(-1).squeeze(0).permute(1, 0)

    def forward(self, edge_idx, edge_rel, edge_norm, training=True):
        Eu, Ee = self.user_emb, self.ent_emb
        if training:
            Eu, Ee = self.mess_dropout(Eu), self.mess_dropout(Ee)
        embs = torch.cat([Eu, Ee], 0)
        running = embs.clone()
        for _ in range(self.L):
            msg = embs[edge_idx[0]] * (self.rel_w[edge_rel] * edge_norm).unsqueeze(1)
            agg = torch.zeros_like(embs).index_add(0, edge_idx[1], msg)
            u, e = agg[:self.n_users], agg[self.n_users:]
            e = e + 0.02 * self.reim_refine(e)               # REIM 残差（α=0.02）
            e = self.gram_refine(e, training)                # GRAM
            embs = torch.cat([u, e], 0)
            running = running + embs
        agg = running / (self.L + 1)
        return agg[:self.n_users], agg[self.n_users:]

    def item_embeddings(self, ent_final):
        return ent_final[self.item2ent]

    def set_context(self, R_idx, n_items, deg):
        """注册对比损失所需的图上下文（训练脚本调用一次）
        R_idx: [2, I] 交互边（row=user, col=item 局部 id）；deg: CKG 节点原始度"""
        self.register_buffer("R_rows", R_idx[0])
        self.register_buffer("R_cols", R_idx[1])
        d = 1.0 / torch.sqrt(deg.float().clamp(min=1))       # 官方：1/sqrt(degree)
        self.register_buffer("deg_norm", d)
        self._n_items = n_items

    def _co_count(self, nodes_a, nodes_b, a_is_user=True):
        """在线批内共同邻居计数（替代官方全量 Similarity_matrix，避免 OOM）
        a_is_user=True: 用户-用户（共同物品数）；False: 物品-物品（共同用户数）"""
        dev = self.deg_norm.device
        if a_is_user:
            keys, nbrs, n_key = self.R_rows, self.R_cols, self.n_users
            width = self._n_items
        else:
            keys, nbrs, n_key = self.R_cols, self.R_rows, self._n_items
            width = self.n_users
        def mat(nodes):
            m = torch.full((n_key,), -1, dtype=torch.long, device=dev)
            m[nodes] = torch.arange(nodes.size(0), device=dev)
            r = m[keys]
            sel = r >= 0
            A = torch.zeros(nodes.size(0), width, device=dev)
            A[r[sel], nbrs[sel]] = 1.0
            return A
        return mat(nodes_a) @ mat(nodes_b).T

    def _cl(self, nodes_a, nodes_b, emb_a, emb_b, deg_a, deg_b, a_is_user):
        """官方对比损失：Σ (1-d_i d_j)·exp(cos_ij·(1+sim_ij)/T)，批内取均值
        sim_ij = clamp(-|co_ij|·d_i·d_j, -50, 50)，d 为 1/sqrt 归一度"""
        co = self._co_count(nodes_a, nodes_b, a_is_user)
        da, db = deg_a.unsqueeze(1), deg_b.unsqueeze(0)
        sim = 1 + (-(co.abs()) * (da * db)).clamp(-50.0, 50.0)
        mat = 1 - da * db
        cos = F.normalize(emb_a, dim=1) @ F.normalize(emb_b, dim=1).T
        expo = (cos * sim / self.temperature).clamp(-30.0, 30.0)   # 数值安全
        return (mat * torch.exp(expo)).mean()

    def contrastive_loss(self, user_idx, pos_idx, neg_idx, uf, itf):
        """user_loss: 批内用户两两；item_loss: 正样本×负样本。节点度取 CKG 用户/物品实体度"""
        du = self.deg_norm[user_idx]
        di_pos = self.deg_norm[self.item2ent[pos_idx]]
        di_neg = self.deg_norm[self.item2ent[neg_idx]]
        u_loss = self._cl(user_idx, user_idx, uf[user_idx], uf[user_idx], du, du, True)
        i_loss = self._cl(pos_idx, neg_idx, itf[pos_idx], itf[neg_idx], di_pos, di_neg, False)
        return u_loss, i_loss
