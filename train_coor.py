# Almost out of storage … If you run out, you can't create or edit files, send or receive email on Gmail, or back up to Google Photos.
import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from scipy.spatial import distance
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics as sk_metrics

import sys
import os
import argparse
import math
import json
import numpy as np
from time import time 
from tqdm import tqdm

from dataset import PDBBindCoor
from model import loss_fn_kd, get_soft_label, loss_fn_dir, loss_fn_cos
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", help="which model we use", type=str, default='Net_coor_res')
parser.add_argument("--loss", help="which loss function we use", type=str, default='L1Loss')
parser.add_argument("--loss_reduction", help="reduction approach for loss function", type=str, default='mean')
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--start_epoch", help="epoch", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--atomwise", help="if we train the model atomwisely", type=int, default = 0)
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 3)
parser.add_argument("--data_path", help="train keys", type=str, default='pdbbind/pdbbind_rmsd_srand200/')
parser.add_argument("--heads", help="number of heads for multi-attention", type=int, default = 1)
parser.add_argument("--edge_dim", help="dimension of edge feature", type=int, default = 3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 1)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 256)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 0)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 512)
parser.add_argument("--output", help="train result", type=str, default='none')
parser.add_argument("--model_dir", help="save best model", type=str, default='best_model.pt')
parser.add_argument("--pre_model", help="pre trained model", type=str, default='None')
parser.add_argument("--th", help="threshold for positive pose", type=float, default=3.00)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--weight_bias", help="weight bias", type=float, default=1.0)
parser.add_argument("--last", help="activation of last layer", type=str, default='log')
parser.add_argument("--KD", help="if we apply knowledge distillation (Yes / No)", type=str, default='No')
parser.add_argument("--KD_soft", help="function convert rmsd to softlabel", type=str, default='exp')
parser.add_argument("--edge", help="if we use edge attr", type=bool, default=False)
parser.add_argument("--plt_dir", help="path to the plot figure", type=str, default='best_model_plt')
parser.add_argument("--flexible", help="if we only calculate flexible nodes", default=False, action='store_true')
parser.add_argument("--residue", help="if we apply residue connection to CONV layers", default=False, action='store_true')
parser.add_argument("--iterative", help="if we iteratively calculate the pose", type=int, default = 0)
parser.add_argument("--pose_limit", help="maximum poses to be evaluated", type=int, default = 0)
parser.add_argument("--step_len", help="length of the moving vector", type=float, default = 0.03)
parser.add_argument("--class_dir", help="classify the direction on each axis", default=False, action='store_true')
parser.add_argument("--hinge", help="rate of hinge loss", type=float, default = 0)
parser.add_argument("--tot_seed", help="num of seeds in the dataset", type=int, default = 8)
parser.add_argument("--hidden_dim", help="dimension of the hidden layer", type=int, default=256)
parser.add_argument("--output_dim", help="dimension of the final output", type=int, default=3)
parser.add_argument("--lambda_align", help="weight for transformation-normalized alignment loss", type=float, default=0.5)
parser.add_argument("--lambda_coord", help="weight for direct coordinate regression loss", type=float, default=0.5)
parser.add_argument("--lambda_steric", help="weight for steric clash penalty", type=float, default=0.05)
parser.add_argument("--steric_cutoff", help="minimum allowed inter-atomic distance (normalized by SPACE)", type=float, default=0.02)
parser.add_argument("--lambda_torsion", help="weight for torsion smoothness regularization", type=float, default=0.02)
parser.add_argument("--use_novel_features", help="expect 13D edge features from novel data path", default=False, action='store_true')
parser.add_argument("--lambda_dihedral", help="weight for direct dihedral supervision", type=float, default=0.1)
parser.add_argument("--torsion_iters", help="number of iterative torsion updates", type=int, default=1)
parser.add_argument("--use_alpha_channel", help="concatenate alpha_ijk edge side tensor into message passing edge_attr", default=False, action='store_true')
parser.add_argument("--metrics_file", help="path to epoch-level metrics jsonl", type=str, default='none')
parser.add_argument("--metrics_per_complex_file", help="path to per-complex metrics jsonl", type=str, default='none')
parser.add_argument("--skip_nonfinite_batches", help="skip optimizer step for NaN/Inf loss batches", default=False, action='store_true')
parser.add_argument("--use_lr_scheduler", help="enable ReduceLROnPlateau on validation loss", default=False, action='store_true')
parser.add_argument("--lr_patience", help="epochs to wait before reducing LR", type=int, default=5)
parser.add_argument("--lr_factor", help="multiplicative factor for LR reduction", type=float, default=0.5)
parser.add_argument("--lr_min", help="minimum LR allowed by scheduler", type=float, default=1e-6)
parser.add_argument("--adaptive_loss_weights", help="learn loss-term weights during training", default=False, action='store_true')
parser.add_argument("--radius_start", help="initial edge radius schedule cutoff in normalized coordinate units", type=float, default=0.0)
parser.add_argument("--radius_end", help="final edge radius schedule cutoff in normalized coordinate units", type=float, default=0.0)
parser.add_argument("--lambda_rank", help="weight for pose ranking auxiliary loss", type=float, default=0.1)
parser.add_argument("--rank_good_th", help="RMSD(A) threshold to tag a pose as good for ranking", type=float, default=2.0)
parser.add_argument("--seed", help="global random seed for reproducibility", type=int, default=42)

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if args.use_novel_features and args.edge_dim != 13 and not args.use_alpha_channel:
    raise ValueError("GeoProNet novel mode expects --edge_dim=13. If you enable --use_alpha_channel then use --edge_dim=14.")
if args.use_novel_features and args.use_alpha_channel and args.edge_dim != 14:
    raise ValueError("GeoProNet novel+alpha mode expects --edge_dim=14.")

if args.atomwise:
    args.batch_size = 1


SPACE = 100.0
bond_th = 6

path = args.data_path

train_datasets = []
test_datasets = []
train_loaders = []
test_loaders = []

train_dataset=PDBBindCoor(root=path, split='train')
test_dataset=PDBBindCoor(root=path, split='test')
train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=1)


train_loader_size = len(train_loader.dataset)
test_dataset_size = len(test_dataset)
test_loader_size = len(test_loader.dataset)


weight = 4.100135326385498 + args.weight_bias
print(f"weight: 1:{weight}")


def Kabsch_3D(A, B):
    #print("A_Shape: ", A.shape)
    #print("B_Shape: ", B.shape)
    # assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    #t = -R @ centroid_B + centroid_A
    return R, t


def _dir_2_coor2(out, length):
    out = out.exp()
    x = out[:, 4:8].sum(1) - out[:, :4].sum(1)
    y = out[:,[2,3,6,7]].sum(1) - out[:,[0,1,4,5]].sum(1)
    z = out[:,[1,3,5,7]].sum(1) - out[:,[0,2,4,6]].sum(1)
    ans = torch.stack([x, y, z], 1)

    return ans*length


def _dir_2_coor(out, length):
    """Backward-compatible alias used by legacy atomwise branch."""
    return _dir_2_coor2(out, length)





gpu_id = str(args.gpu_id)
device_str = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)

print('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(train_datasets[0].num_features, train_datasets[0].num_classes, args).to(device)

from model import Net_coor, Net_coor_res, Net_coor_dir, Net_coor_len, Net_coor_cent, Net_coor_torsion, Net_coor_two_stage

if args.model_type == 'Net_coor_res':
    model = Net_coor_res(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor':
    #model = Net_coor(train_dataset.num_features, args).to(device)
    #GraphSAGE = Net_coor(train_dataset.num_features, args.hidden_dim, args.output_dim).to(device)
    model = Net_coor(train_dataset.num_features, args).to(device)


elif args.model_type == 'Net_coor_dir':
    model = Net_coor_dir(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_len':
    model = Net_coor_len(train_dataset.num_features, args).to(device)
    # assert args.class_dir
elif args.model_type == 'Net_coor_cent':
    model = Net_coor_cent(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_torsion':
    model = Net_coor_torsion(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_two_stage':
    model = Net_coor_two_stage(train_dataset.num_features, args).to(device)

if args.pre_model != 'None':
    model = torch.load(args.pre_model, map_location=device_str, weights_only=False).to(device)
# loss_op = torch.nn.MSELoss()

#model.double()
#torch.set_default_dtype(torch.double)

if args.loss == 'L1Loss':
    loss_op = torch.nn.L1Loss(reduction=args.loss_reduction)
elif args.loss == 'MSELoss':
    loss_op = torch.nn.MSELoss(reduction=args.loss_reduction)
elif args.loss == 'CosineEmbeddingLoss':
    loss_op = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
    cos_target = torch.tensor([1]).to(device)
elif args.loss == 'CosineAngle':
    loss_op = loss_fn_cos(device, reduction=args.loss_reduction)
    loss_op2 = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
    # loss_op = torch.nn.CosineEmbeddingLoss(reduction='none')
    cos_target = torch.tensor([1]).to(device)
if args.class_dir:
    # loss_op = loss_fn_dir(device)
    loss_op = torch.nn.CrossEntropyLoss()
    assert args.model_type == 'Net_coor_dir'
# loss_op_kld = torch.nn.KLDivLoss()
hinge = torch.tensor([args.hinge]).to(device)
rank_loss_op = torch.nn.BCEWithLogitsLoss()


class AdaptiveGeoLoss(torch.nn.Module):
    """
    Uncertainty-weighted multi-objective loss.
    Uses learned log variances to rebalance terms online.
    """
    def __init__(self, initial_weights):
        super().__init__()
        self.log_vars = torch.nn.ParameterDict({
            key: torch.nn.Parameter(torch.log(torch.tensor(1.0 / max(weight, 1e-8), dtype=torch.float32)))
            for key, weight in initial_weights.items()
        })

    def forward(self, losses):
        total = 0.0
        for key, loss_val in losses.items():
            log_var = self.log_vars[key]
            precision = torch.exp(-log_var)
            total = total + precision * loss_val + log_var
        return total


adaptive_loss = None
if args.adaptive_loss_weights:
    adaptive_loss = AdaptiveGeoLoss({
        "align": args.lambda_align,
        "coord": args.lambda_coord,
        "steric": args.lambda_steric,
        "torsion": args.lambda_torsion,
        "dihedral": args.lambda_dihedral,
    }).to(device)

opt_params = list(model.parameters())
if adaptive_loss is not None:
    opt_params += list(adaptive_loss.parameters())
optimizer = torch.optim.Adam(opt_params, lr=args.lr)


scheduler = None
if args.use_lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

def bond_dist(data, pred, fix_idx):
    # print(data.x.size(), data.edge_index.size(), pred.size())
    x = data.edge_index[0, fix_idx]
    y = data.edge_index[1, fix_idx]
    node_x = data.x[x, -3:] + pred[x]
    node_y = data.x[y, -3:] + pred[y]
    # dist = (node_x - node_y).square()
    dist = torch.nn.MSELoss(reduction='none')(node_x, node_y)

    return dist.sum(-1).sqrt()

def kabsch_align_torch(predicted, target):
    """Align predicted coordinates to target with Kabsch (differentiable torch path)."""
    # SVD kernels used in Kabsch are not implemented for fp16 on some CUDA paths.
    # Disable autocast and force fp32 math for this block.
    in_dtype = predicted.dtype
    with torch.cuda.amp.autocast(enabled=False):
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
        try:
            u, _, v_t = torch.linalg.svd(h)
        except RuntimeError:
            return torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0).to(in_dtype)
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


def steric_clash_penalty(data, pred, flexible_mask):
    """Soft physical constraint: penalize short ligand-protein contacts."""
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
    penalty = torch.relu(args.steric_cutoff - d).square().mean()
    return penalty


def clash_stats(data, pred, flexible_mask):
    """Return (clash_rate, clash_penalty) for ligand-protein contacts."""
    if data.dist.size(0) == 0:
        z = pred.new_tensor(0.0)
        return z, z

    src = data.edge_index[0]
    dst = data.edge_index[1]
    ligand_to_protein = flexible_mask[src] & (~flexible_mask[dst])
    if ligand_to_protein.sum() == 0:
        z = pred.new_tensor(0.0)
        return z, z

    src_idx = src[ligand_to_protein]
    dst_idx = dst[ligand_to_protein]
    pred_coor = data.x[:, -3:].clone()
    pred_coor[flexible_mask] = data.x[flexible_mask, -3:] + pred
    d = (pred_coor[src_idx] - pred_coor[dst_idx]).square().sum(dim=1).sqrt()
    clash_rate = (d < args.steric_cutoff).float().mean()
    clash_pen = torch.relu(args.steric_cutoff - d).square().mean()
    return clash_rate, clash_pen


def torsion_smoothness_penalty(data, pred, flexible_mask):
    """Approximate torsion regularization using ligand-ligand edge displacement smoothness."""
    src = data.edge_index[0]
    dst = data.edge_index[1]
    ligand_edge = flexible_mask[src] & flexible_mask[dst]
    if ligand_edge.sum() == 0:
        return pred.new_tensor(0.0)

    src = src[ligand_edge]
    dst = dst[ligand_edge]
    delta = pred[src] - pred[dst]
    return delta.norm(dim=1).mean()


def _bond_graph_from_data(data, flexible_mask):
    if not hasattr(data, 'bonds'):
        return {}
    bonds = data.bonds.long()
    graph = {}
    for row in bonds:
        if row.numel() < 2:
            continue
        i, j = int(row[0].item()), int(row[1].item())
        if i >= flexible_mask.size(0) or j >= flexible_mask.size(0):
            continue
        if not (bool(flexible_mask[i]) and bool(flexible_mask[j])):
            continue
        graph.setdefault(i, set()).add(j)
        graph.setdefault(j, set()).add(i)
    return graph


def _rotatable_bonds(graph):
    ans = []
    for i in graph:
        for j in graph[i]:
            if i < j and len(graph[i]) > 1 and len(graph[j]) > 1:
                ans.append((i, j))
    return ans


def _dihedral_torch(a, b, c, d, eps=1e-8):
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)
    n1 = n1 / (torch.norm(n1) + eps)
    n2 = n2 / (torch.norm(n2) + eps)
    m1 = torch.cross(n1, b2 / (torch.norm(b2) + eps))
    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)
    return torch.atan2(y, x)


def _rotate_points(points, origin, axis, theta):
    axis = axis / (torch.norm(axis) + 1e-8)
    p = points - origin
    c = torch.cos(theta)
    s = torch.sin(theta)
    cross = torch.cross(axis.expand_as(p), p, dim=1)
    dot = (p * axis).sum(dim=1, keepdim=True)
    rot = p * c + cross * s + axis * dot * (1 - c)
    return rot + origin


def _downstream_subtree(graph, root, blocked):
    stack = [root]
    seen = {blocked}
    out = []
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)
        for nei in graph.get(cur, []):
            if nei not in seen:
                stack.append(nei)
    return out


def apply_torsion_updates(coords, graph, delta_theta, iters=1):
    out = coords.clone()
    bonds = _rotatable_bonds(graph)
    if len(bonds) == 0:
        return out
    for _ in range(max(1, iters)):
        for i, j in bonds:
            axis = out[j] - out[i]
            downstream = _downstream_subtree(graph, j, i)
            if not downstream:
                continue
            idx = torch.tensor(downstream, dtype=torch.long, device=out.device)
            theta = 0.5 * (delta_theta[i] + delta_theta[j])
            out[idx] = _rotate_points(out[idx], out[j], axis, theta)
    return out


def dihedral_supervision_loss(data, pred_abs, target_abs, flexible_mask, torsion_node):
    graph = _bond_graph_from_data(data, flexible_mask)
    bonds = _rotatable_bonds(graph)
    if len(bonds) == 0:
        return pred_abs.new_tensor(0.0)

    losses = []
    for i, j in bonds:
        left = [n for n in graph[i] if n != j]
        right = [n for n in graph[j] if n != i]
        if not left or not right:
            continue
        u = left[0]
        v = right[0]
        p_ang = _dihedral_torch(pred_abs[u], pred_abs[i], pred_abs[j], pred_abs[v])
        t_ang = _dihedral_torch(target_abs[u], target_abs[i], target_abs[j], target_abs[v])
        diff = torch.atan2(torch.sin(p_ang - t_ang), torch.cos(p_ang - t_ang))
        losses.append(diff.square())
    if not losses:
        return pred_abs.new_tensor(0.0)
    return torch.stack(losses).mean()


def torsion_error_metrics(data, pred_abs, target_abs, flexible_mask):
    """Return (MAE, angular_RMSE) on ligand rotatable bond dihedrals."""
    graph = _bond_graph_from_data(data, flexible_mask)
    bonds = _rotatable_bonds(graph)
    if len(bonds) == 0:
        z = pred_abs.new_tensor(0.0)
        return z, z

    diffs = []
    for i, j in bonds:
        left = [n for n in graph[i] if n != j]
        right = [n for n in graph[j] if n != i]
        if not left or not right:
            continue
        u = left[0]
        v = right[0]
        p_ang = _dihedral_torch(pred_abs[u], pred_abs[i], pred_abs[j], pred_abs[v])
        t_ang = _dihedral_torch(target_abs[u], target_abs[i], target_abs[j], target_abs[v])
        diff = torch.atan2(torch.sin(p_ang - t_ang), torch.cos(p_ang - t_ang))
        diffs.append(diff)
    if not diffs:
        z = pred_abs.new_tensor(0.0)
        return z, z
    diffs = torch.stack(diffs)
    return diffs.abs().mean(), diffs.square().mean().sqrt()


def geopronet_loss(data, pred, target, flexible_mask, torsion_node=None):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    aligned = kabsch_align_torch(pred, target)
    align_loss = F.l1_loss(aligned, target)
    coord_loss = F.mse_loss(pred, target)
    steric_loss = steric_clash_penalty(data, pred, flexible_mask)
    torsion_loss = torsion_smoothness_penalty(data, pred, flexible_mask)
    dihedral_loss = pred.new_tensor(0.0)

    if torsion_node is not None:
        start = data.x[flexible_mask, -3:]
        pred_abs = start + pred
        target_abs = start + target
        graph = _bond_graph_from_data(data, flexible_mask)
        pred_abs = apply_torsion_updates(pred_abs, graph, torsion_node[flexible_mask], args.torsion_iters)
        dihedral_loss = dihedral_supervision_loss(data, pred_abs, target_abs, flexible_mask, torsion_node)

    total = (
        args.lambda_align * align_loss
        + args.lambda_coord * coord_loss
        + args.lambda_steric * steric_loss
        + args.lambda_torsion * torsion_loss
        + args.lambda_dihedral * dihedral_loss
    )
    if adaptive_loss is not None:
        total = adaptive_loss({
            "align": align_loss,
            "coord": coord_loss,
            "steric": steric_loss,
            "torsion": torsion_loss,
            "dihedral": dihedral_loss,
        })
    return total
def ranking_targets_from_displacement(data, target, flexible_mask):
    if hasattr(data, "pose_rmsd"):
        rmsd = data.pose_rmsd.float().view(-1)
        if rmsd.numel() > 0:
            return (rmsd <= args.rank_good_th).float()
    if hasattr(data, 'batch'):
        batch_full = data.batch
    else:
        batch_full = torch.zeros((data.x.size(0),), dtype=torch.long, device=target.device)
    batch_flex = batch_full[flexible_mask]
    num_graphs = int(batch_full.max().item()) + 1 if batch_full.numel() > 0 else 1
    labels = []
    for g in range(num_graphs):
        idx = (batch_flex == g)
        if idx.sum() == 0:
            labels.append(target.new_tensor(0.0))
            continue
        rmsd = torch.sqrt(target[idx].square().sum(dim=1).mean()) * SPACE
        labels.append((rmsd <= args.rank_good_th).float())
    return torch.stack(labels)


def rank_calibration_metrics(labels, probs, bins=10):
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
    
from torch_geometric.utils import add_self_loops
def current_radius(epoch):
    if args.radius_start <= 0 or args.radius_end <= 0:
        return None
    if args.epoch <= 1:
        return args.radius_end
    progress = (epoch - args.start_epoch) / max(args.epoch - 1, 1)
    progress = max(0.0, min(1.0, progress))
    return args.radius_start + (args.radius_end - args.radius_start) * progress


def build_graph_inputs(data, epoch=None):
    edge_index = data.edge_index
    alpha = data.alpha.float() if hasattr(data, 'alpha') else None

    radius = current_radius(epoch) if epoch is not None else None
    if radius is not None:
        src = edge_index[0]
        dst = edge_index[1]
        pos = data.x[:, -3:].float()
        d = (pos[src] - pos[dst]).square().sum(dim=1).sqrt()
        mask = d <= radius
        if mask.sum() > 0:
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
            if alpha is not None:
                alpha = alpha[mask]

    if args.use_alpha_channel and alpha is not None:
        edge_attr = torch.cat([edge_attr, alpha], dim=1)
    return edge_index, edge_attr



def update_input_data(data, pred):
    # Assuming pred is a tensor containing the predicted x, y, z coordinates for flexible atoms
    # Update data.x for flexible atoms
    flexible_idx = data.flexible_idx.bool()
    pred = pred.to(data.x.dtype)
    pred = pred.to(data.x.device)
    data.x[flexible_idx, -3:] = pred

    # Recalculate edge_index and edge_attr based on the updated coordinates
    pos = data.x[:, -3:]
    edge_index = []
    edge_attr = []

    for i in range(pos.size(0)):
        for j in range(i + 1, pos.size(0)):
            edge_index.append((i, j))
            edge_attr.append(pos[j] - pos[i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr, dim=0)

    # Add self-loops to edge_index
    edge_index, _ = add_self_loops(edge_index)

    # Update data.edge_index and data.edge_attr
    data.edge_index = edge_index
    data.dist = edge_attr

    return data




def train(epoch):
    model.train()
    total_reward = 0.0
    total_loss1 = 0
    total_loss2 = 0
    total_loss = 0
    tot = 0
    skipped_nonfinite = 0
    t = time()
    pbar = tqdm(total=train_loader_size)
    pbar.set_description('Training poses...')
    
    for data in train_loader:
        num_atoms = data.x.size()[0]
        print("num_atoms",num_atoms)
        num_flexible_atoms = data.x[data.flexible_idx.bool()].size()[0]
        print("num_flexible_atoms",num_flexible_atoms)
        #print(data)
        with torch.cuda.amp.autocast():
            data = data.to(device)
            if args.atomwise:
                flexible_len = data.flexible_len.cpu().item()
                all_atom_idx = torch.randperm(flexible_len)
                avg_loss = 0.0
                for idx in range(args.atomwise):
                    st = (idx * flexible_len) // args.atomwise
                    ed = ((idx + 1) * flexible_len) // args.atomwise
                    atom_idx = all_atom_idx[st:ed]
                    optimizer.zero_grad()
                    edge_index, edge_attr = build_graph_inputs(data, epoch)
                    pred = model(data.x.to(device).float(), edge_index.to(device), edge_attr)[atom_idx]
                    pred = _dir_2_coor(pred, args.step_len)
                    loss = loss_op(pred.to(device).float(), data.y[atom_idx])
                    avg_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                total_loss += avg_loss / args.atomwise
                tot += 1
                pbar.update(1)
                continue

            optimizer.zero_grad()
            if args.flexible:
                torsion_node = None
                rank_logit = None
                if args.model_type != 'Net_coor_cent':
                    if args.model_type == 'Net_coor_torsion':
                        edge_index, edge_attr = build_graph_inputs(data, epoch)
                        pred_all, torsion_node = model(data.x.to(device).float(), edge_index, edge_attr)
                        pred = pred_all[data.flexible_idx.bool()]
                    elif args.model_type == 'Net_coor_two_stage':
                        edge_index, edge_attr = build_graph_inputs(data, epoch)
                        pred_all, rank_logit = model(data.x.to(device).float(), edge_index, edge_attr, data.batch if hasattr(data, 'batch') else None)
                        pred = pred_all[data.flexible_idx.bool()]
                    else:
                        edge_index, edge_attr = build_graph_inputs(data, epoch)
                        pred = model(data.x.to(device).float(), edge_index, edge_attr)[data.flexible_idx.bool()]
                    
                
                    
                    

                    #if epoch ==150 or epoch == 250:
                        #updated_data = update_input_data(data.clone(), pred.cpu().detach())
                        #data.x = updated_data.x
                        #data.edge_index = updated_data.edge_index
                        #data.dist = updated_data.dist

            
                    data = data.to(device)
                    
                
                   
                if args.class_dir:
                    y = data.y.gt(0).long()
                    y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                    loss = loss_op(pred, y)
                elif args.loss == 'CosineEmbeddingLoss':
                    loss = loss_op(pred, data.y, cos_target)
                elif args.loss == 'CosineAngle':
                    loss = loss_op(pred, data.y)
                elif args.model_type == 'Net_coor_len':
                    length = data.y.square().sum(1).sqrt().reshape(pred.size()[0],1)
                    loss = loss_op(pred, length)
                elif args.model_type == 'Net_coor_cent':
                    edge_index, edge_attr = build_graph_inputs(data, epoch)
                    pred = model(data.x, edge_index, edge_attr, data.batch, data.flexible_idx.bool())
                    y = global_mean_pool(data.y, data.batch[data.flexible_idx.bool()])
                    loss = loss_op(pred, y)
                elif args.hinge != 0:
                    fix_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                    bond_diff = bond_dist(data, pred, fix_idx) - bond_dist(data, data.y, fix_idx)
                    l = fix_idx.size()[0]
                    loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(bond_diff, torch.LongTensor([-1] * l).to(device))
                    loss = loss_op(pred, data.y) + loss1 * hinge
                else:
                    
                    target = data.y.to(device).float()
                    torsion_node = torsion_node if args.model_type == 'Net_coor_torsion' else None
                    loss = geopronet_loss(data, pred.to(device).float(), target, data.flexible_idx.bool(), torsion_node=torsion_node)
                    if args.model_type == 'Net_coor_two_stage':
                        rank_target = ranking_targets_from_displacement(data, target, data.flexible_idx.bool())
                        rank_loss = rank_loss_op(rank_logit.view(-1), rank_target.view(-1))
                        loss = loss + args.lambda_rank * rank_loss


            else:
                edge_index, edge_attr = build_graph_inputs(data, epoch)
                pred = model(data.x, edge_index, edge_attr)
                loss = loss_op(pred, data.y)
            if args.loss == 'CosineEmbeddingLoss':
                total_loss += loss.item() / pred.size()[0] * args.batch_size
            if args.loss == 'CosineAngle':
                total_loss += loss.item() / pred.size()[0] * args.batch_size
            else:
                total_loss1 += loss.item() * args.batch_size
                #total_loss2 += k_loss.item() * args.batch_size
                #total_loss3 += m_loss.item() * args.batch_size
        if args.skip_nonfinite_batches and (not torch.isfinite(loss).all().item()):
            skipped_nonfinite += 1
            pbar.update(1)
            continue
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #print(data.x[data.flexible_idx.bool()])
        #print(pred)
        
        tot += 1
        pbar.update(1)
        # break
    pbar.close()
    
    print(f"trained {tot} batches, skipped_nonfinite {skipped_nonfinite}, take {time() - t}s")
    return total_loss1 / train_loader_size


    



@torch.no_grad()
def test(loader, epoch):
    model.eval()
    t = time()

    total_loss = 0
    total_rmsd = 0.0
    total_rmsd_in = 0.0
    all_rmsds = []
    all_rmsds_in = []
    total_atoms = 0

    pose_idx = 0
    gstd = 0
    total_rmsds = [0.0 for i in range(args.iterative)]
    avg_rmsd = 0.0
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    fpl = [0 for i in range(8)]


    all_atoms = 0
    ligand_atoms = 0
    diff_complex = 0
    rmsd_per_pdb = []
    num_pose_per_pdb = []
    rmsd_per_pdb_in = []
    pdb = ''
    pdbs = []
    torsion_abs_err_sum = 0.0
    torsion_sq_err_sum = 0.0
    torsion_count = 0
    clash_rate_sum = 0.0
    clash_penalty_sum = 0.0
    rank_labels = []
    rank_probs = []

    pbar = tqdm(total=test_loader_size)
    pbar.set_description('Testing poses...')
    for data in loader:
        pbar.update(1)
        # with torch.cuda.amp.autocast():
        num_atoms = data.x.size()[0]
        print("num_atoms",num_atoms)
        num_flexible_atoms = data.x[data.flexible_idx.bool()].size()[0]
        print("num_flexible_atoms",num_flexible_atoms)

        if data.pdb != pdb:
            diff_complex += 1
            all_atoms = num_atoms
            ligand_atoms = num_flexible_atoms
            rmsd_per_pdb.append(0.0)
            rmsd_per_pdb_in.append(0.0)
            num_pose_per_pdb.append(0)
            pdb = data.pdb
        
            pdbs.append(pdb[0])

        if data.x.size()[0] != num_atoms:
            print(f"num_flexible_atoms: {num_flexible_atoms}, data.x.size: {data.x.size()[0]}, data.y.size: {num_atoms}")
        if args.flexible:
            if args.model_type != 'Net_coor_cent':
                if args.model_type == 'Net_coor_torsion':
                    edge_index, edge_attr = build_graph_inputs(data.to(device), epoch)
                    out_all, torsion_node = model(data.x.to(device).float(), edge_index.to(device), edge_attr)
                    out = out_all[data.flexible_idx.bool()]
                elif args.model_type == 'Net_coor_two_stage':
                    torsion_node = None
                    edge_index, edge_attr = build_graph_inputs(data.to(device), epoch)
                    out_all, rank_logit = model(data.x.to(device).float(), edge_index.to(device), edge_attr, data.batch.to(device) if hasattr(data, 'batch') else None)
                    out = out_all[data.flexible_idx.bool()]
                else:
                    torsion_node = None
                    edge_index, edge_attr = build_graph_inputs(data.to(device), epoch)
                    out = model(data.x.to(device).float(), edge_index.to(device), edge_attr)[data.flexible_idx.bool()]
                                
                
                #out = _dir_2_coor(out, args.step_len)
                #out = _dir_2_coor2(out, args.step_len)
            if args.class_dir:
                #y = data.y[data.flexible_idx.bool()].gt(0).long().to(device)
                edge_index, edge_attr = build_graph_inputs(data.to(device), epoch)
                y =  model(data.x.to(device), edge_index.to(device), edge_attr)[data.flexible_idx.bool()]
                y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                loss = loss_op(out, y)
                for i in range(8):
                    fpl[i] += y.eq(i).sum().cpu().item()
                tn += y.size()[0]
                #out = _dir_2_coor(out, args.step_len)
            elif args.loss == 'CosineEmbeddingLoss':
                loss = loss_op(out, data.y.to(device), cos_target)
            elif args.loss == 'CosineAngle':
                loss = (1 - loss_op2(out, data.y.to(device), cos_target)).acos().sum()
            elif args.model_type == 'Net_coor_len':
                length = data.y.to(device).square().sum(1).sqrt().reshape(out.size()[0],1)
                loss = loss_op(out, length)
                out = data.y.to(device)
            elif args.model_type == 'Net_coor_cent':
                edge_index, edge_attr = build_graph_inputs(data.to(device), epoch)
                pred = model(data.x.to(device), edge_index.to(device), edge_attr, data.batch.to(device), data.flexible_idx.bool().to(device)).cpu()
                y = global_mean_pool(data.y, data.batch[data.flexible_idx.bool()])
                loss = loss_op(pred, y)
                out = pred.repeat(num_flexible_atoms, 1)
            elif args.hinge != 0:
                fix_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                bond_diff = bond_dist(data.to(device), out, fix_idx) - bond_dist(data.to(device), data.y.to(device), fix_idx)
                loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(bond_diff, torch.LongTensor([-1 for _ in fix_idx]).to(device))
                loss = loss_op(out, data.y.to(device)) + loss1 * args.hinge

            else:
                target = data.y.to(device).float()
                loss = geopronet_loss(
                    data.to(device),
                    out.to(device).float(),
                    target,
                    data.flexible_idx.bool().to(device),
                    torsion_node=torsion_node,
                )
                if args.model_type == 'Net_coor_two_stage':
                    rank_target = ranking_targets_from_displacement(data.to(device), target, data.flexible_idx.bool().to(device))
                    r_loss = rank_loss_op(rank_logit.view(-1), rank_target.view(-1))
                    loss = loss + args.lambda_rank * r_loss
                    rank_labels.extend(rank_target.detach().cpu().tolist())
                    rank_probs.extend(torch.sigmoid(rank_logit).detach().cpu().tolist())
                pred_clean = torch.nan_to_num(out.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
                target_clean = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                flex_mask = data.flexible_idx.bool().to(device)
                c_rate, c_pen = clash_stats(data.to(device), pred_clean, flex_mask)
                clash_rate_sum += float(c_rate.item())
                clash_penalty_sum += float(c_pen.item())

                if args.model_type == 'Net_coor_torsion' and torsion_node is not None:
                    start = data.x.to(device)[flex_mask, -3:]
                    pred_abs = start + pred_clean
                    target_abs = start + target_clean
                    t_mae, t_rmse = torsion_error_metrics(data.to(device), pred_abs, target_abs, flex_mask)
                    torsion_abs_err_sum += float(t_mae.item())
                    torsion_sq_err_sum += float(t_rmse.item() ** 2)
                    torsion_count += 1
            out_eval = torch.nan_to_num(out.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            y_eval = torch.nan_to_num(data.y.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            aligned_out = kabsch_align_torch(out_eval, y_eval)
            rmsds = math.sqrt(F.mse_loss(y_eval, aligned_out, reduction='sum').cpu().item() / max(num_flexible_atoms, 1))
            
                        
            total_rmsd += rmsds
            all_rmsds.append(rmsds)
            
                
                

            num_pose_per_pdb[-1] += 1
            rmsd_per_pdb[-1] += rmsds


        else: # not flexible
            edge_index, edge_attr = build_graph_inputs(data.to(device), epoch)
            out = model(data.x.to(device), edge_index.to(device), edge_attr)
            loss = loss_op(out, data.y.to(device)) 
            rmsds = F.mse_loss(data.y, out.cpu()[data.flexible_idx.bool()], reduction='sum').item()
            total_rmsd += math.sqrt(rmsds / num_flexible_atoms)
            all_rmsds.append(math.sqrt(rmsds / num_flexible_atoms))


        if (epoch <= 1):
            if args.flexible:
                rmsds = math.sqrt(torch.sum(torch.square(data.y)).item() / num_flexible_atoms)
                total_rmsd_in += rmsds
                all_rmsds_in.append(rmsds)
                rmsd_per_pdb_in[-1] += rmsds
            else:
                rmsds = F.mse_loss(data.y, data.x[:, -3:], reduction='sum').item()
                total_rmsd_in += math.sqrt(rmsds / num_atoms)
                all_rmsds_in.append(math.sqrt(rmsds / num_atoms))
        # all_rmsds_in = all_rmsds_in + rmsds

        if args.loss == 'CosineEmbeddingLoss':
            total_loss += loss.item() / num_flexible_atoms
        elif args.loss == 'CosineAngle':
            total_loss += loss.item() / num_flexible_atoms
        else:
            total_loss += loss.item()

        pose_idx += 1
        if args.pose_limit > 0 and pose_idx >= args.pose_limit:
            break
    
    pbar.close()
    tt = time() - t
    print(f"Spend {tt}s")
    if args.class_dir:
        print(f"class_dir histogram x: {fpl}, all: {tn}")
    else:
        print("class_dir disabled: histogram counters are not used.")
    print([i / pose_idx for i in total_rmsds])
    print(avg_rmsd / pose_idx)

    print(f'diff_complex {diff_complex}')
    assert diff_complex % args.tot_seed == 0
    diff_complex = diff_complex // args.tot_seed
    print(f'diff_complex {diff_complex}')
    for ii in range(1, args.tot_seed):
        for jj in range(diff_complex):
            num_pose_per_pdb[jj] += num_pose_per_pdb[ii * diff_complex + jj]
            rmsd_per_pdb[jj] += rmsd_per_pdb[ii * diff_complex + jj]
            rmsd_per_pdb_in[jj] += rmsd_per_pdb_in[ii * diff_complex + jj]

    avg_rmsd_per_pdb = sum([r / d for r, d in zip(rmsd_per_pdb[:diff_complex], num_pose_per_pdb[:diff_complex])]) / diff_complex
    avg_rmsd_per_pdb_in = sum([r / d for r, d in zip(rmsd_per_pdb_in[:diff_complex], num_pose_per_pdb[:diff_complex])]) / diff_complex
    #return total_loss / pose_idx, avg_rmsd_per_pdb, avg_rmsd_per_pdb_in
    rmsd_arr = np.array(all_rmsds, dtype=np.float32) if len(all_rmsds) else np.array([0.0], dtype=np.float32)
    rmsd_arr_A = rmsd_arr * SPACE
    if epoch <= 1:
        avg_rmsd_per_pdb_in = avg_rmsd_per_pdb_in * SPACE
    else:
        avg_rmsd_per_pdb_in = None

    elapsed = max(time() - t, 1e-8)
    metrics = {
        "val_loss": float(total_loss / pose_idx),
        "rmsd_mean_A": float(rmsd_arr_A.mean()),
        "rmsd_median_A": float(np.median(rmsd_arr_A)),
        "rmsd_std_A": float(rmsd_arr_A.std()),
        "rmsd_p90_A": float(np.percentile(rmsd_arr_A, 90)),
        "sr_2a": float((rmsd_arr_A <= 2.0).mean()),
        "sr_5a": float((rmsd_arr_A <= 5.0).mean()),
        "avg_rmsd_per_complex_A": float(avg_rmsd_per_pdb * SPACE),
        "avg_input_rmsd_per_complex_A": (float(avg_rmsd_per_pdb_in) if avg_rmsd_per_pdb_in is not None else None),
        "num_poses": int(pose_idx),
        "num_complexes": int(diff_complex),
        "avg_poses_per_complex": float(pose_idx / max(diff_complex, 1)),
        "clash_rate": float(clash_rate_sum / max(pose_idx, 1)),
        "clash_penalty": float(clash_penalty_sum / max(pose_idx, 1)),
        "efficiency_pose_per_sec": float(pose_idx / elapsed),
        "torsion_mae_rad": (float(torsion_abs_err_sum / torsion_count) if torsion_count > 0 else None),
        "torsion_angular_rmse_rad": (float(math.sqrt(torsion_sq_err_sum / torsion_count)) if torsion_count > 0 else None),
    }

    if len(rank_labels) > 1 and len(set(rank_labels)) > 1:
        metrics["rank_auc"] = float(sk_metrics.roc_auc_score(rank_labels, rank_probs))
        ece, brier = rank_calibration_metrics(rank_labels, rank_probs)
        metrics["rank_ece"] = ece
        metrics["rank_brier"] = brier
    else:
        metrics["rank_auc"] = None
        metrics["rank_ece"] = None
        metrics["rank_brier"] = None
    per_complex = []
    for jj in range(diff_complex):
        denom = max(num_pose_per_pdb[jj], 1)
        per_complex.append({
            "pdb": str(pdbs[jj]) if jj < len(pdbs) else str(jj),
            "num_poses": int(num_pose_per_pdb[jj]),
            "avg_rmsd_A": float((rmsd_per_pdb[jj] / denom) * SPACE),
            "avg_input_rmsd_A": (float((rmsd_per_pdb_in[jj] / denom) * SPACE) if epoch <= 1 else None),
        })

    return metrics, per_complex




if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.plt_dir):
    os.makedirs(args.plt_dir)
min_rmsd = 10.0
best_epoch = 0

for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
    loss1 = train(epoch)
    print(f"Train Loss: {loss1}")
    #loss2, rmsd, rmsd_in = test(test_loader, epoch)
    #print(f"Epoch: {epoch} Train Loss: {loss1} Validation Loss: {loss2}  Avg RMSD: {rmsd}")
    eval_metrics, per_complex_metrics = test(test_loader, epoch)
    loss2 = eval_metrics["val_loss"]
    if scheduler is not None:
        scheduler.step(loss2)
    current_lr = optimizer.param_groups[0]["lr"]
    if adaptive_loss is not None:
        for key, param in adaptive_loss.log_vars.items():
            eval_metrics[f"adaptive_weight_{key}"] = float(torch.exp(-param).detach().cpu().item())
    rmsd = eval_metrics["avg_rmsd_per_complex_A"]
    rmsd_in = eval_metrics["avg_input_rmsd_per_complex_A"]
    print(f"Epoch: {epoch} Train Loss: {loss1} Validation Loss: {loss2}  Avg RMSD: {rmsd} SR@2A: {eval_metrics['sr_2a']} SR@5A: {eval_metrics['sr_5a']} LR: {current_lr}")
    #print(f"Epoch: {epoch} Proposed Loss: {ploss} Kabsch Loss: {kloss}  Model Loss: {mloss}")
    torch.cuda.empty_cache()
    if epoch <= 1:
        print(f"Avg RMSD of inputs: {rmsd_in}")
        if args.output != 'none':
            with open(args.output, 'a') as f:
                f.write(f"Epoch: {epoch} Train Loss: {loss1} Validation Loss: {loss2}  Avg RMSD: {rmsd} SR@2A: {eval_metrics['sr_2a']} SR@5A: {eval_metrics['sr_5a']} LR: {current_lr}\n")
            #f.write(f"Epoch: {epoch} Proposed Loss: {ploss} Kabsch Loss: {kloss}  Model Loss: {mloss}\n")
    if args.metrics_file != 'none':
        os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True) if os.path.dirname(args.metrics_file) else None
        with open(args.metrics_file, 'a') as mf:
            payload = {"epoch": epoch, "train_loss": float(loss1), "lr": float(current_lr), **eval_metrics}
            mf.write(json.dumps(payload) + "\n")
    if args.metrics_per_complex_file != 'none':
        os.makedirs(os.path.dirname(args.metrics_per_complex_file), exist_ok=True) if os.path.dirname(args.metrics_per_complex_file) else None
        with open(args.metrics_per_complex_file, 'a') as pf:
            for row in per_complex_metrics:
                row_payload = {"epoch": epoch, **row}
                pf.write(json.dumps(row_payload) + "\n")
    if epoch > 3 and (min_rmsd > rmsd) :
        saved_model_dir = os.path.join(args.model_dir, f'model_{epoch}.pt')
        torch.save(model.state_dict(), saved_model_dir)
        os.system(f'chmod 777 {saved_model_dir}')
        print(f"save model at epoch {epoch}, rmsd of {rmsd} !!!!!!!!")
        if args.output != 'none':
            with open(args.output, 'a') as f:
                f.write(f"save model at epoch {epoch}, rmsd of {rmsd} !!!!!!!!\n")
        if min_rmsd > rmsd:
            min_rmsd = rmsd
            best_epoch = epoch
    print("")
    os.system(f'chmod 777 {args.output}')

print(f"\nBest model at epoch {best_epoch}, rmsd is {min_rmsd}")



