"""Node-level geometry: weight matrices, Laplacians and heat kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import networkx as nx
import numpy as np

Symmetrisation = Literal["absolute", "signed", "presence"]
LaplacianKind = Literal["combinatorial", "normalised", "random_walk_sym"]


def weight_matrix(
    A: np.ndarray,
    P: Optional[np.ndarray] = None,
    mode: Symmetrisation = "absolute",
) -> np.ndarray:
    """Symmetric non-negative weight matrix from the signed adjacency.

    absolute : W = (|A| + |A|.T) / 2            <- default, ignores sign
    presence : W = (P + P.T) / 2                <- keeps dual edges too
    signed   : W = (A + A.T) / 2                <- may break PSD downstream
    """
    if mode == "absolute":
        M = np.abs(A)
    elif mode == "presence":
        if P is None:
            raise ValueError("presence mode requires the presence mask P")
        M = np.abs(P)
    elif mode == "signed":
        M = A
    else:
        raise ValueError(f"unknown symmetrisation {mode!r}")
    W = 0.5 * (M + M.T)
    np.fill_diagonal(W, 0.0)
    return W


def laplacian(W: np.ndarray, kind: LaplacianKind = "combinatorial") -> np.ndarray:
    d = W.sum(axis=1)
    if kind == "combinatorial":
        return np.diag(d) - W
    safe = np.where(d > 0, d, 1.0)
    if kind == "normalised":
        inv_sqrt = 1.0 / np.sqrt(safe)
        return np.eye(len(d)) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
    if kind == "random_walk_sym":
        # symmetrised random-walk Laplacian: keeps degree heterogeneity milder
        inv = 1.0 / safe
        M = 0.5 * (inv[:, None] * W + W * inv[None, :])
        return np.eye(len(d)) - M
    raise ValueError(f"unknown Laplacian kind {kind!r}")


@dataclass
class GraphGeometry:
    """One eigendecomposition of L, reused across all diffusion scales t."""

    W: np.ndarray
    kind: LaplacianKind = "combinatorial"

    def __post_init__(self) -> None:
        self.L = laplacian(self.W, self.kind)
        L_sym = 0.5 * (self.L + self.L.T)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L_sym)
        self.eigenvalues = np.clip(self.eigenvalues, 0.0, None)

    def kernel(self, t: float) -> np.ndarray:
        """K_t = exp(-t L) from the cached spectrum."""
        scale = np.exp(-t * self.eigenvalues)
        return (self.eigenvectors * scale) @ self.eigenvectors.T

    def kernel_sqrt(self, t: float) -> np.ndarray:
        """K_t ** 0.5 = exp(-t L / 2); the feature map for the state distance."""
        scale = np.exp(-0.5 * t * self.eigenvalues)
        return (self.eigenvectors * scale) @ self.eigenvectors.T

    def spectral_gap(self) -> float:
        """First non-trivial eigenvalue; 1 / gap is a natural scale for t."""
        nonzero = self.eigenvalues[self.eigenvalues > 1e-10]
        return float(nonzero[0]) if nonzero.size else float("nan")

    def suggest_t_grid(self, n_points: int = 9) -> np.ndarray:
        """Log-spaced t values bracketing the graph's diffusion time scales."""
        gap = self.spectral_gap()
        largest = float(self.eigenvalues[-1]) or 1.0
        lo = 0.01 / largest
        hi = 10.0 / gap if np.isfinite(gap) and gap > 0 else 10.0
        return np.logspace(np.log10(lo), np.log10(hi), n_points)


def graph_from_weights(W: np.ndarray, nodes) -> nx.Graph:
    G = nx.from_numpy_array(W)
    return nx.relabel_nodes(G, {i: name for i, name in enumerate(nodes)})


def shortest_path_cost(W: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path cost matrix, used as ground cost for Wasserstein."""
    G = nx.from_numpy_array(W)
    n = W.shape[0]
    C = np.full((n, n), np.inf)
    for source, lengths in nx.all_pairs_shortest_path_length(G):
        for target, value in lengths.items():
            C[source, target] = float(value)
    finite = C[np.isfinite(C)]
    fallback = (finite.max() + 1.0) if finite.size else 1.0
    return np.where(np.isfinite(C), C, fallback)