"""
Learned predictors of residual link lifetime (PyTorch).

All models receive, for a batch of snapshots t and an observation window of W slots:
  node : (B, W, N, Fn)   node features  (position, velocity, degree)
  edge : (B, W, M, Fe)   pair features  (distance, radial/relative speed, SNR, dSNR,
                                          capacity, LET, link-up flag), zero if link down
  up   : (B, W, M)       link state
and output the predicted residual lifetime of every pair at slot t: (B, M) in [0, H].

Models
  MLP        per-link, current slot only
  LSTM       per-link, sequence over the window (JPE / LPMD-GPSR style, no graph)
  StaticGNN  edge-aware GAT on the current snapshot only (no temporal model)
  STGNN      edge-aware GAT on every snapshot + node GRU + link GRU

Note: MLP and LSTM do not aggregate neighbour features, but their node
features include the local node degree.

v4 (Hello information model): node has W+1 slots (t-W..t); hel (B, W, M) marks
active links whose Hello was received. For an active link with a missed Hello
(up & ~hel), the pair-level inputs of the previous slot are used (held), and
GNN messages are exchanged only over links with a received Hello.
StaticGNN with agg=False is the same network without neighbour aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, depth=2, drop=0.1):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def pair_node_feats(node, iu, ju):
    """Symmetric combination of the two endpoint feature vectors."""
    a, b = node[..., iu, :], node[..., ju, :]
    return torch.cat([a + b, (a - b).abs()], -1)


def held_pair_feats(node, up, hel, iu, ju):
    """node (B, W+1, N, Fn) -> (B, W, M, 2Fn); held pairs use the previous slot."""
    cur, prev = pair_node_feats(node[:, 1:], iu, ju), pair_node_feats(node[:, :-1], iu, ju)
    miss = (up & ~hel).unsqueeze(-1)
    return torch.where(miss, prev, cur)


class EdgeGAT(nn.Module):
    """Dense edge-feature-aware graph attention layer (multi-head)."""

    def __init__(self, d, fe, heads=4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(fe, d, bias=False)
        self.a_dst = nn.Parameter(0.1 * torch.randn(heads, self.dk))
        self.a_src = nn.Parameter(0.1 * torch.randn(heads, self.dk))
        self.a_e = nn.Parameter(0.1 * torch.randn(heads, self.dk))
        self.self_lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, E, A):
        # h (B, N, d) | E (B, N, N, fe) | A (B, N, N) bool
        B, N, d = h.shape
        Wh = self.W(h).view(B, N, self.h, self.dk)
        Ue = self.U(E).view(B, N, N, self.h, self.dk)
        sc = ((Wh * self.a_dst).sum(-1)[:, :, None, :]          # receiver i
              + (Wh * self.a_src).sum(-1)[:, None, :, :]        # sender j
              + (Ue * self.a_e).sum(-1))                        # link (i, j)
        sc = F.leaky_relu(sc, 0.2)
        mask = A.unsqueeze(-1)
        sc = sc.masked_fill(~mask, -1e4)
        att = torch.softmax(sc, dim=2) * mask                   # isolated node -> zero message
        msg = torch.einsum("bijh,bjhd->bihd", att, Wh) + (att.unsqueeze(-1) * Ue).sum(2)
        out = F.elu(self.self_lin(h) + msg.reshape(B, N, d))
        return self.norm(out + h)


class MLPModel(nn.Module):
    def __init__(self, fn, fe, H, d=128):
        super().__init__()
        self.H, self.net = H, mlp(fe + 2 * fn, d, 1, depth=3)

    def forward(self, node, edge, up, hel, iu, ju):
        pn = held_pair_feats(node[:, -2:], up[:, -1:], hel[:, -1:], iu, ju)[:, -1]
        x = torch.cat([edge[:, -1], pn], -1)
        return self.H * torch.sigmoid(self.net(x).squeeze(-1))


class LSTMModel(nn.Module):
    def __init__(self, fn, fe, H, d=64):
        super().__init__()
        self.H = H
        self.lstm = nn.LSTM(fe + 2 * fn, d, num_layers=2, batch_first=True, dropout=0.1)
        self.head = mlp(d, d, 1)

    def forward(self, node, edge, up, hel, iu, ju):
        B, W, M, _ = edge.shape
        x = torch.cat([edge, held_pair_feats(node, up, hel, iu, ju)], -1)   # (B, W, M, F)
        x = x.permute(0, 2, 1, 3).reshape(B * M, W, -1)
        _, (h, _) = self.lstm(x)
        return self.H * torch.sigmoid(self.head(h[-1]).view(B, M))


class STGNN(nn.Module):
    """Proposed model. temporal=False gives the StaticGNN ablation."""

    def __init__(self, fn, fe, H, n_nodes, iu, ju, d=64, layers=2, heads=4, dg=32, temporal=True, agg=True):
        super().__init__()
        self.H, self.N, self.temporal, self.agg = H, n_nodes, temporal, agg
        self.register_buffer("iu", iu)
        self.register_buffer("ju", ju)
        self.enc = nn.Linear(fn, d)
        self.gat = nn.ModuleList([EdgeGAT(d, fe, heads) for _ in range(layers)])
        if temporal:
            self.node_gru = nn.GRU(d, d, batch_first=True)
            self.edge_gru = nn.GRU(fe, dg, batch_first=True)
        else:
            self.edge_mlp = mlp(fe, dg, dg, depth=1)
        self.head = mlp(2 * d + dg + fe, 128, 1, depth=2)

    def dense(self, edge, up):
        B, W, M, fe = edge.shape
        E = edge.new_zeros(B, W, self.N, self.N, fe)
        A = torch.zeros(B, W, self.N, self.N, dtype=torch.bool, device=edge.device)
        E[:, :, self.iu, self.ju] = edge
        E[:, :, self.ju, self.iu] = edge
        A[:, :, self.iu, self.ju] = up
        A[:, :, self.ju, self.iu] = up
        return E, A

    def set_nodes(self, n_nodes, iu, ju):
        """Re-use the trained weights for a swarm of another size."""
        self.N, self.iu, self.ju = n_nodes, iu, ju

    def forward(self, node, edge, up, hel, iu=None, ju=None):
        miss_last = (up[:, -1] & ~hel[:, -1]).unsqueeze(-1)            # (B, M, 1)
        if not self.temporal:                       # current and previous snapshot only
            node, edge, up, hel = node[:, -2:], edge[:, -2:], up[:, -2:], hel[:, -2:]
        else:
            node = node[:, 1:]
        B, W, N, _ = node.shape
        A_src = hel if self.agg else torch.zeros_like(hel)              # messages only with a received Hello
        E, A = self.dense(edge, A_src)
        h = self.enc(node).reshape(B * W, N, -1)
        E, A = E.reshape(B * W, N, N, -1), A.reshape(B * W, N, N)
        for layer in self.gat:
            h = layer(h, E, A)
        h = h.view(B, W, N, -1)
        if self.temporal:
            zs, _ = self.node_gru(h.permute(0, 2, 1, 3).reshape(B * N, W, -1))   # (B*N, W, d)
            zs = zs.view(B, N, W, -1)
            z, zp = zs[:, :, -1], zs[:, :, -2]
            M = edge.shape[2]
            _, g = self.edge_gru(edge.permute(0, 2, 1, 3).reshape(B * M, W, -1))
            g = g[-1].view(B, M, -1)
        else:
            z, zp = h[:, -1], h[:, -2]
            g = self.edge_mlp(edge[:, -1])
        zi = torch.where(miss_last, zp[:, self.iu], z[:, self.iu])
        zj = torch.where(miss_last, zp[:, self.ju], z[:, self.ju])
        x = torch.cat([zi + zj, (zi - zj).abs(), g, edge[:, -1]], -1)
        return self.H * torch.sigmoid(self.head(x).squeeze(-1))


def censored_huber(pred, y, cens, mask, H, delta=2.0, lam=1.0):
    """Huber loss on uncensored links; one-sided penalty (pred < H) on censored links.
    NOTE: because every model outputs H*sigmoid(.) <= H and censored targets are H,
    this loss is mathematically identical to plain_huber. Kept for reference only."""
    unc = F.huber_loss(pred, y, reduction="none", delta=delta)
    cen = F.huber_loss(torch.clamp(H - pred, min=0), torch.zeros_like(pred), reduction="none", delta=delta)
    loss = torch.where(cens, lam * cen, unc)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def plain_huber(pred, y, cens, mask, H, delta=2.0, lam=1.0):
    """Huber loss over all active links; censored samples have target H (= lower bound)."""
    loss = F.huber_loss(pred, y, reduction="none", delta=delta)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)
