"""Geometry utilities for KTransPose novel feature variants.

These functions are optional and backward-compatible with the original
3-channel edge features (ligand-ligand, ligand-protein, protein-protein).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def unit_vector(delta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(delta)
    if norm < eps:
        return np.zeros_like(delta)
    return delta / norm


def rbf_expand(distance_value: float, centers: Sequence[float], gamma: float = 16.0) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float32)
    d = np.float32(distance_value)
    return np.exp(-gamma * (d - c) ** 2)


def torsion_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-8) -> float:
    """Return dihedral angle in radians for points a-b-c-d."""
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)

    if n1_norm < eps or n2_norm < eps or b2_norm < eps:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / b2_norm)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(math.atan2(y, x))


def build_ligand_adjacency(edge_gt: Iterable[Tuple[int, int]], ligand_nodes: int) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {i: [] for i in range(ligand_nodes)}
    for i, j in edge_gt:
        if 0 <= i < ligand_nodes and 0 <= j < ligand_nodes:
            adj[i].append(j)
    return adj


def local_torsion_stats(i: int, j: int, coords: np.ndarray, ligand_adj: Dict[int, List[int]]) -> Tuple[float, float]:
    """Compute mean sin/cos torsion around ligand bond (i,j)."""
    if i not in ligand_adj or j not in ligand_adj:
        return 0.0, 1.0

    left = [n for n in ligand_adj[i] if n != j]
    right = [n for n in ligand_adj[j] if n != i]
    if not left or not right:
        return 0.0, 1.0

    vals = []
    for u in left:
        for v in right:
            angle = torsion_angle(coords[u], coords[i], coords[j], coords[v])
            vals.append(angle)

    if not vals:
        return 0.0, 1.0
    vals = np.asarray(vals, dtype=np.float32)
    return float(np.sin(vals).mean()), float(np.cos(vals).mean())
