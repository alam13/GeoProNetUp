import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


def get_soft_label(rmsd, mode='exp', temperature=1.0):
    if mode == 'exp':
        return torch.exp(-rmsd / max(temperature, 1e-6))
    if mode == 'linear':
        return torch.clamp(1.0 - rmsd / max(temperature, 1e-6), min=0.0)
    return rmsd


def loss_fn_kd(pred, target, reduction='mean'):
    return F.mse_loss(pred, target, reduction=reduction)


class loss_fn_dir(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class loss_fn_cos(nn.Module):
    def __init__(self, device=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        cos = (pred_n * target_n).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        ang = torch.acos(cos)
        if self.reduction == 'sum':
            return ang.sum()
        if self.reduction == 'none':
            return ang
        return ang.mean()


class _GeoBase(nn.Module):
    def __init__(self, in_channels, args, out_channels=3):
        super().__init__()
        hidden = args.d_graph_layer
        layers = max(1, args.n_graph_layer)
        heads = max(1, args.heads)
        edge_dim = args.edge_dim
        dropout = args.dropout_rate
        self.residual = getattr(args, 'residue', False)

        self.input_proj = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    beta=True,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden))

        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )

    def encode(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, edge_index, edge_attr)
            h = norm(h + update) if self.residual else norm(update)
            h = F.relu(h)
        return h

    def forward(self, x, edge_index, edge_attr, batch=None, flexible_idx=None):
        h = self.encode(x, edge_index, edge_attr)
        return self.out(h)


class Net_coor(_GeoBase):
    def __init__(self, in_channels, args):
        super().__init__(in_channels, args, out_channels=3)


class Net_coor_res(_GeoBase):
    def __init__(self, in_channels, args):
        super().__init__(in_channels, args, out_channels=3)


class Net_coor_dir(_GeoBase):
    def __init__(self, in_channels, args):
        super().__init__(in_channels, args, out_channels=8)


class Net_coor_len(_GeoBase):
    def __init__(self, in_channels, args):
        super().__init__(in_channels, args, out_channels=1)


class Net_coor_cent(_GeoBase):
    def __init__(self, in_channels, args):
        super().__init__(in_channels, args, out_channels=3)

    def forward(self, x, edge_index, edge_attr, batch=None, flexible_idx=None):
        h = self.encode(x, edge_index, edge_attr)
        if batch is None:
            return self.out(h).mean(dim=0, keepdim=True)
        if flexible_idx is not None:
            pooled = global_mean_pool(h[flexible_idx], batch[flexible_idx])
        else:
            pooled = global_mean_pool(h, batch)
        return self.out(pooled)


class Net_coor_torsion(_GeoBase):
    """Joint head: coordinate displacement + node torsion logits."""
    def __init__(self, in_channels, args):
        super().__init__(in_channels, args, out_channels=3)
        hidden = args.d_graph_layer
        self.torsion_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, flexible_idx=None):
        h = self.encode(x, edge_index, edge_attr)
        coor_delta = self.out(h)
        torsion_node = self.torsion_head(h).squeeze(-1)
        return coor_delta, torsion_node
