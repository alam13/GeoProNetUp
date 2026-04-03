import torch

from losses.geometric_losses import geopronet_loss


SPACE = 100.0


def build_graph_inputs(data, use_alpha_channel=False):
    edge_index = data.edge_index
    edge_attr = data.dist.float()
    if use_alpha_channel and hasattr(data, "alpha"):
        edge_attr = torch.cat([edge_attr, data.alpha.float()], dim=1)
    return edge_index, edge_attr


def ranking_targets(data, target, batch_flex, num_graphs, rank_good_th):
    if hasattr(data, "pose_rmsd"):
        rmsd = data.pose_rmsd.float().view(-1)
        if rmsd.numel() == num_graphs:
            return (rmsd <= rank_good_th).float()
    labels = []
    for g in range(num_graphs):
        idx = batch_flex == g
        if idx.sum() == 0:
            labels.append(target.new_tensor(0.0))
            continue
        rmsd = torch.sqrt(target[idx].square().sum(dim=1).mean()) * SPACE
        labels.append((rmsd <= rank_good_th).float())
    return torch.stack(labels)


def train_one_epoch(model, loader, optimizer, device, cfg, adaptive_loss=None):
    model.train()
    rank_loss_op = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        edge_index, edge_attr = build_graph_inputs(data, use_alpha_channel=cfg.use_alpha_channel)
        flexible_mask = data.flexible_idx.bool()
        target = data.y.float()

        optimizer.zero_grad()
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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)
