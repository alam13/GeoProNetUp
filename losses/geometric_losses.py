import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGeoLoss(nn.Module):
    """Uncertainty-weighted multi-objective loss."""

    def __init__(self, initial_weights):
        super().__init__()
        self.log_vars = nn.ParameterDict(
            {
                key: nn.Parameter(torch.log(torch.tensor(1.0 / max(weight, 1e-8), dtype=torch.float32)))
                for key, weight in initial_weights.items()
            }
        )

    def forward(self, losses):
        total = 0.0
        for key, loss_val in losses.items():
            log_var = self.log_vars[key]
            precision = torch.exp(-log_var)
            total = total + precision * loss_val + log_var
        return total


def kabsch_align_torch(predicted, target):
    in_dtype = predicted.dtype
    predicted = predicted.float()
    target = target.float()
    finite_rows = torch.isfinite(predicted).all(dim=1) & torch.isfinite(target).all(dim=1)
    if finite_rows.sum() < 3:
        return torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0).to(in_dtype)

    predicted_f = predicted[finite_rows]
    target_f = target[finite_rows]

    pred_centroid = predicted_f.mean(dim=0, keepdim=True)
    target_centroid = target_f.mean(dim=0, keepdim=True)

    pred_centered = predicted_f - pred_centroid
    target_centered = target_f - target_centroid

    h = pred_centered.transpose(0, 1) @ target_centered
    u, _, v_t = torch.linalg.svd(h)
    r = v_t.transpose(0, 1) @ u.transpose(0, 1)

    if torch.det(r) < 0:
        v_t = v_t.clone()
        v_t[-1, :] *= -1
        r = v_t.transpose(0, 1) @ u.transpose(0, 1)

    t = target_centroid.transpose(0, 1) - r @ pred_centroid.transpose(0, 1)
    aligned = torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    aligned_f = (r @ predicted_f.transpose(0, 1) + t).transpose(0, 1)
    aligned[finite_rows] = aligned_f
    return aligned.to(in_dtype)


def steric_clash_penalty(data, pred, flexible_mask, steric_cutoff):
    if data.dist.size(0) == 0:
        return pred.new_tensor(0.0)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    ligand_to_protein = flexible_mask[src] & (~flexible_mask[dst])
    if ligand_to_protein.sum() == 0:
        return pred.new_tensor(0.0)

    src_idx = src[ligand_to_protein]
    dst_idx = dst[ligand_to_protein]

    pred_coor = data.x[:, -3:].clone()
    pred_coor[flexible_mask] = data.x[flexible_mask, -3:] + pred
    d = (pred_coor[src_idx] - pred_coor[dst_idx]).square().sum(dim=1).sqrt()
    return torch.relu(steric_cutoff - d).square().mean()


def torsion_smoothness_penalty(data, pred, flexible_mask):
    src = data.edge_index[0]
    dst = data.edge_index[1]
    ligand_edge = flexible_mask[src] & flexible_mask[dst]
    if ligand_edge.sum() == 0:
        return pred.new_tensor(0.0)
    src = src[ligand_edge]
    dst = dst[ligand_edge]
    delta = pred[src] - pred[dst]
    return delta.norm(dim=1).mean()


def geopronet_loss(data, pred, target, flexible_mask, weights, steric_cutoff, adaptive_loss=None):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    align_loss = F.l1_loss(kabsch_align_torch(pred, target), target)
    coord_loss = F.mse_loss(pred, target)
    steric_loss = steric_clash_penalty(data, pred, flexible_mask, steric_cutoff)
    torsion_loss = torsion_smoothness_penalty(data, pred, flexible_mask)

    losses = {
        "align": align_loss,
        "coord": coord_loss,
        "steric": steric_loss,
        "torsion": torsion_loss,
        "dihedral": pred.new_tensor(0.0),
    }
    if adaptive_loss is not None:
        return adaptive_loss(losses), losses

    total = (
        weights["align"] * align_loss
        + weights["coord"] * coord_loss
        + weights["steric"] * steric_loss
        + weights["torsion"] * torsion_loss
        + weights["dihedral"] * losses["dihedral"]
    )
    return total, losses
