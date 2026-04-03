import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sk_metrics

from losses.geometric_losses import geopronet_loss, kabsch_align_torch
from train_loop import build_graph_inputs, ranking_targets


SPACE = 100.0


def calibration_metrics(labels, probs, bins=10):
    labels = np.array(labels, dtype=np.float32)
    probs = np.array(probs, dtype=np.float32)
    if labels.size == 0:
        return None, None
    brier = float(np.mean((probs - labels) ** 2))

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / labels.size)
    return float(ece), brier


@torch.no_grad()
def evaluate(model, loader, device, cfg, adaptive_loss=None):
    model.eval()
    rank_labels = []
    rank_probs = []
    total_loss = 0.0
    all_rmsds = []

    rank_loss_op = torch.nn.BCEWithLogitsLoss()

    for data in loader:
        data = data.to(device)
        edge_index, edge_attr = build_graph_inputs(data, use_alpha_channel=cfg.use_alpha_channel)
        flexible_mask = data.flexible_idx.bool()
        target = data.y.float()
        rank_logit = None

        if cfg.model_type == "Net_coor_torsion":
            out_all, _ = model(data.x.float(), edge_index, edge_attr)
            pred = out_all[flexible_mask]
        elif cfg.model_type == "Net_coor_two_stage":
            out_all, rank_logit = model(data.x.float(), edge_index, edge_attr, data.batch if hasattr(data, "batch") else None)
            pred = out_all[flexible_mask]
        else:
            pred = model(data.x.float(), edge_index, edge_attr)[flexible_mask]

        loss, _ = geopronet_loss(
            data,
            pred,
            target,
            flexible_mask,
            weights={
                "align": cfg.lambda_align,
                "coord": cfg.lambda_coord,
                "steric": cfg.lambda_steric,
                "torsion": cfg.lambda_torsion,
                "dihedral": cfg.lambda_dihedral,
            },
            steric_cutoff=cfg.steric_cutoff,
            adaptive_loss=adaptive_loss,
        )

        if cfg.model_type == "Net_coor_two_stage" and rank_logit is not None:
            batch_full = data.batch if hasattr(data, "batch") else torch.zeros((data.x.size(0),), dtype=torch.long, device=device)
            batch_flex = batch_full[flexible_mask]
            num_graphs = int(batch_full.max().item()) + 1 if batch_full.numel() > 0 else 1
            rank_target = ranking_targets(data, target, batch_flex, num_graphs, cfg.rank_good_th)
            loss = loss + cfg.lambda_rank * rank_loss_op(rank_logit.view(-1), rank_target.view(-1))
            rank_labels.extend(rank_target.detach().cpu().tolist())
            rank_probs.extend(torch.sigmoid(rank_logit).detach().cpu().tolist())

        total_loss += float(loss.item())

        aligned = kabsch_align_torch(pred.float(), target.float())
        rmsd = math.sqrt(F.mse_loss(target.float(), aligned, reduction="sum").item() / max(pred.size(0), 1))
        all_rmsds.append(rmsd * SPACE)

    rmsd_arr = np.array(all_rmsds, dtype=np.float32) if all_rmsds else np.array([0.0], dtype=np.float32)
    metrics = {
        "val_loss": float(total_loss / max(len(loader), 1)),
        "rmsd_mean_A": float(rmsd_arr.mean()),
        "rmsd_median_A": float(np.median(rmsd_arr)),
        "sr_2a": float((rmsd_arr <= 2.0).mean()),
        "sr_5a": float((rmsd_arr <= 5.0).mean()),
    }
    if len(rank_labels) > 1 and len(set(rank_labels)) > 1:
        metrics["rank_auc"] = float(sk_metrics.roc_auc_score(rank_labels, rank_probs))
        ece, brier = calibration_metrics(rank_labels, rank_probs)
        metrics["rank_ece"] = ece
        metrics["rank_brier"] = brier
    else:
        metrics["rank_auc"] = None
        metrics["rank_ece"] = None
        metrics["rank_brier"] = None
    return metrics
