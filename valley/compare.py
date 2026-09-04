"""Build every candidate distance matrix and compare them."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import distances as dist
from .bnet import BooleanModel, signed_adjacency, state_matrix
from .geometry import GraphGeometry, graph_from_weights, weight_matrix


def upper(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def mantel(D1: np.ndarray, D2: np.ndarray, method: str = "spearman") -> float:
    """Correlation between two distance matrices on off-diagonal entries."""
    a, b = upper(D1), upper(D2)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    fn = spearmanr if method == "spearman" else pearsonr
    return float(fn(a, b)[0])


def centrality_weights(W: np.ndarray, nodes, kind: str = "degree") -> np.ndarray:
    G = graph_from_weights(W, nodes)
    if kind == "degree":
        scores = dict(G.degree(weight="weight"))
    elif kind == "betweenness":
        scores = nx.betweenness_centrality(G, weight=None)
    elif kind == "eigenvector":
        scores = nx.eigenvector_centrality_numpy(G, weight="weight")
    elif kind == "pagerank":
        scores = nx.pagerank(G, weight="weight")
    else:
        raise ValueError(f"unknown centrality {kind!r}")
    w = np.array([float(scores.get(name, 0.0)) for name in nodes])
    total = w.sum()
    return w * (len(w) / total) if total > 0 else np.ones_like(w)


def scc_modules(A: np.ndarray) -> list:
    """Strongly connected components of the directed influence graph."""
    G = nx.from_numpy_array(np.abs(A), create_using=nx.DiGraph)
    return [sorted(component) for component in nx.strongly_connected_components(G)]


def build_distance_matrices(
    attractors: pd.DataFrame,
    model: BooleanModel,
    t_values: Optional[Sequence[float]] = None,
    symmetrisation: str = "absolute",
    laplacian_kind: str = "combinatorial",
    include_wasserstein: bool = False,
) -> Dict[str, np.ndarray]:
    """Return {approach_name: distance_matrix} for all approaches under test."""
    S, _ = state_matrix(attractors, model)
    A, P = signed_adjacency(model)
    W = weight_matrix(A, P, mode=symmetrisation)
    geometry = GraphGeometry(W, kind=laplacian_kind)

    if t_values is None:
        t_values = geometry.suggest_t_grid()

    out: Dict[str, np.ndarray] = {"hamming": dist.hamming(S)}

    for kind in ("degree", "betweenness", "pagerank"):
        try:
            w = centrality_weights(W, model.nodes, kind=kind)
        except Exception as exc:  # keep the sweep going
            print(f"[skip] weighted_hamming:{kind} -> {exc}")
            continue
        out[f"weighted_hamming:{kind}"] = dist.weighted_hamming(S, w)

    for t in t_values:
        out[f"heat_kernel:t={t:.4g}"] = dist.heat_kernel(S, geometry, float(t))

    modules = scc_modules(A)
    if 1 < len(modules) < model.n:
        out["module_scc"] = dist.module_coarse_grained(S, modules)

    if S.shape[0] >= 2:
        out["reference_axis"] = dist.reference_axis(S, S[0], S[-1])

    if include_wasserstein:
        try:
            out["wasserstein"] = dist.wasserstein(S, W)
        except Exception as exc:
            print(f"[skip] wasserstein -> {exc}")

    return out


def compare_matrices(
    matrices: Dict[str, np.ndarray], reference: str = "hamming"
) -> pd.DataFrame:
    """Per-approach summary: spread, and agreement with the reference distance."""
    ref = matrices[reference]
    rows = []
    for name, D in matrices.items():
        values = upper(D)
        scale = values.max() or 1.0
        rows.append(
            {
                "approach": name,
                "mean": float(values.mean()),
                "cv": float(values.std() / (values.mean() or np.nan)),
                "n_distinct": int(np.unique(np.round(values / scale, 6)).size),
                "spearman_vs_ref": mantel(D, ref, "spearman"),
                "pearson_vs_ref": mantel(D, ref, "pearson"),
            }
        )
    return pd.DataFrame(rows).set_index("approach").sort_values("spearman_vs_ref")


def cross_correlation(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Spearman correlation between every pair of approaches."""
    names = list(matrices)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for b in names[i:]:
            value = 1.0 if a == b else mantel(matrices[a], matrices[b])
            M.loc[a, b] = M.loc[b, a] = value
    return M


def separation_scores(
    matrices: Dict[str, np.ndarray], labels: Sequence[str]
) -> pd.DataFrame:
    """Silhouette score per approach, for known attractor / phenotype labels."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        raise ValueError("need at least two distinct labels")
    rows = []
    for name, D in matrices.items():
        score = silhouette_score(D, labels, metric="precomputed")
        rows.append({"approach": name, "silhouette": float(score)})
    return pd.DataFrame(rows).set_index("approach").sort_values(
        "silhouette", ascending=False
    )


def sweep_t(
    attractors: pd.DataFrame,
    model: BooleanModel,
    t_values: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """How much the heat kernel departs from Hamming as t grows.

    A Spearman correlation near 1 across the whole grid means the structural
    term adds nothing on this model.
    """
    S, _ = state_matrix(attractors, model)
    A, P = signed_adjacency(model)
    W = weight_matrix(A, P)
    geometry = GraphGeometry(W)
    if t_values is None:
        t_values = geometry.suggest_t_grid(15)
    D_hamming = dist.hamming(S)
    rows = []
    for t in t_values:
        D = dist.heat_kernel(S, geometry, float(t))
        values = upper(D)
        rows.append(
            {
                "t": float(t),
                "spearman_vs_hamming": mantel(D, D_hamming),
                "mean_distance": float(values.mean()),
                "relative_spread": float(
                    values.std() / (values.mean() or np.nan)
                ),
                "n_distinct": int(np.unique(np.round(values, 8)).size),
            }
        )
    return pd.DataFrame(rows)