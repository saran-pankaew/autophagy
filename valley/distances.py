"""Attractor distance definitions. Every function returns a square matrix."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .geometry import GraphGeometry, shortest_path_cost


def _from_gram(G: np.ndarray) -> np.ndarray:
    """D[i, j] = sqrt(G_ii + G_jj - 2 G_ij) for a PSD Gram matrix G."""
    diag = np.diag(G)
    D2 = diag[:, None] + diag[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def quadratic_form_distance(S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Distance induced by a PSD matrix K: d(s, s')^2 = (s - s') K (s - s').

    Covers Hamming (K = I), node-weighted Hamming (K = diag(w)) and the
    heat-kernel distance (K = exp(-t L)) with a single vectorised expression.
    """
    return _from_gram(S @ K @ S.T)


def hamming(S: np.ndarray, squared: bool = False) -> np.ndarray:
    """Plain Hamming distance. With binary states this equals the L1 distance."""
    D = np.abs(S[:, None, :] - S[None, :, :]).sum(axis=2)
    return D**2 if squared else D


def weighted_hamming(S: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Node-weighted Hamming: sum_k w_k * |s_k - s'_k|."""
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != S.shape[1]:
        raise ValueError("weights length must match the number of nodes")
    return np.abs(S[:, None, :] - S[None, :, :]) @ w


def heat_kernel(S: np.ndarray, geometry: GraphGeometry, t: float) -> np.ndarray:
    """Heat-kernel (Laplacian) state distance at diffusion scale t."""
    return quadratic_form_distance(S, geometry.kernel(t))


def diffusion_features(
    S: np.ndarray, geometry: GraphGeometry, t: float
) -> np.ndarray:
    """Euclidean feature map Z = S K_t^(1/2); ||z_i - z_j|| = heat-kernel d."""
    return S @ geometry.kernel_sqrt(t)


def module_coarse_grained(
    S: np.ndarray, modules: Sequence[Sequence[int]]
) -> np.ndarray:
    """Euclidean distance between module-average activity vectors."""
    Phi = np.column_stack([S[:, list(m)].mean(axis=1) for m in modules])
    return _from_gram(Phi @ Phi.T)


def reference_axis(
    S: np.ndarray, ref_low: np.ndarray, ref_high: np.ndarray
) -> np.ndarray:
    """1D projection onto a chosen attractor pair, then absolute difference."""
    axis = np.asarray(ref_high, dtype=float) - np.asarray(ref_low, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("reference attractors are identical")
    z = (S - np.asarray(ref_low, dtype=float)) @ (axis / norm)
    return np.abs(z[:, None] - z[None, :])


def wasserstein(
    S: np.ndarray, W: np.ndarray, reg: Optional[float] = 0.05
) -> np.ndarray:
    """Optimal-transport distance between activity profiles over the network.

    Ground cost = shortest-path length on the influence graph. Each state is
    normalised into a distribution over active nodes; all-zero states are
    replaced by the uniform distribution. Requires the `pot` package.
    """
    import ot  # optional dependency

    C = shortest_path_cost(W)
    C = C / C.max()
    mass = S.sum(axis=1, keepdims=True)
    uniform = np.full(S.shape[1], 1.0 / S.shape[1])
    Pdist = np.where(mass > 0, S / np.where(mass > 0, mass, 1.0), uniform)
    m = S.shape[0]
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            if reg:
                value = ot.sinkhorn2(Pdist[i], Pdist[j], C, reg)
            else:
                value = ot.emd2(Pdist[i], Pdist[j], C)
            D[i, j] = D[j, i] = float(np.squeeze(value))
    return D


def set_distance(
    groups: Sequence[np.ndarray],
    K: np.ndarray,
    linkage: str = "average",
) -> np.ndarray:
    """Set-to-set distance for cyclic attractors given as arrays of states.

    linkage: "average", "single", "complete" or "hausdorff".
    """
    m = len(groups)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            block = quadratic_form_distance(
                np.vstack([groups[i], groups[j]]), K
            )
            ni = groups[i].shape[0]
            cross = block[:ni, ni:]
            if linkage == "average":
                value = cross.mean()
            elif linkage == "single":
                value = cross.min()
            elif linkage == "complete":
                value = cross.max()
            elif linkage == "hausdorff":
                value = max(cross.min(axis=1).max(), cross.min(axis=0).max())
            else:
                raise ValueError(f"unknown linkage {linkage!r}")
            D[i, j] = D[j, i] = float(value)
    return D