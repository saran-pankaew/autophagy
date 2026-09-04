"""Project attractor distances into 2 or 3 dimensions.

The distance matrix stays the object of study; an embedding is only a lossy
view of it. So every method here returns an `Embedding` and is meant to be
scored with `score_embeddings`, never trusted on visual inspection alone.

Methods follow the candidate list of the design notes: classical MDS as the
deterministic reference, non-metric MDS for ordinal-only distances, heat-kernel
feature PCA as the projection coherent with the selected distance, UMAP and
PHATE for basin structure, t-SNE for exploration only, and a force-directed
layout when the landscape is better read as a network of basins.

Typical use
-----------
>>> D = matrices["heat_kernel:t=1"]
>>> embeddings = build_embeddings(D, n_components=2)
>>> score_embeddings(embeddings, D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from .compare import upper
from .distances import diffusion_features
from .geometry import GraphGeometry

_DIM_NAMES = ("dim1", "dim2", "dim3")
_ALL_METHODS = (
    "classical_mds",
    "nonmetric_mds",
    "heat_kernel_pca",
    "umap",
    "phate",
    "tsne",
    "force_directed",
)
_STOCHASTIC = {"nonmetric_mds", "umap", "phate", "tsne", "force_directed"}


# --- container --------------------------------------------------------------


@dataclass
class Embedding:
    """Coordinates produced by one embedding method, plus its diagnostics."""

    name: str
    coords: np.ndarray
    params: Dict[str, object] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return int(self.coords.shape[1])

    def to_frame(
        self,
        labels: Optional[Sequence[str]] = None,
        height: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Tidy coordinate table, optionally with the landscape elevation U."""
        frame = pd.DataFrame(
            self.coords, columns=list(_DIM_NAMES[: self.n_components])
        )
        if labels is not None:
            frame.index = pd.Index(list(labels), name="attractor")
        if height is not None:
            frame["U"] = np.asarray(height, dtype=float)
        return frame


# --- helpers ----------------------------------------------------------------


def _as_distance(D) -> np.ndarray:
    """Validate and symmetrise a pairwise distance matrix."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def embedded_distances(coords: np.ndarray) -> np.ndarray:
    """Euclidean distances between embedded points."""
    X = np.asarray(coords, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def _safe_neighbours(n: int, requested: int) -> int:
    """Neighbourhood size that stays valid for small attractor sets."""
    return int(max(2, min(requested, n - 1)))


# --- deterministic embeddings ----------------------------------------------


def classical_mds(D, n_components: int = 2) -> Embedding:
    """Classical (Torgerson) MDS: eigendecomposition of the centred Gram matrix.

    Global distances are preserved, so between-cluster gaps are readable.
    Deterministic; used as the reference embedding. `negative_eigenvalue_mass`
    reports how far the distance is from being Euclidean.
    """
    D = _as_distance(D)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    B = 0.5 * (B + B.T)
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    kept = np.clip(values[:n_components], 0.0, None)
    coords = vectors[:, :n_components] * np.sqrt(kept)
    positive_total = np.clip(values, 0.0, None).sum() or np.nan
    absolute_total = np.abs(values).sum() or np.nan
    return Embedding(
        name="classical_mds",
        coords=coords,
        params={"n_components": n_components},
        info={
            "eigenvalues": values[:n_components],
            "variance_explained": float(kept.sum() / positive_total),
            "negative_eigenvalue_mass": float(
                -values[values < 0].sum() / absolute_total
            ),
        },
    )


def heat_kernel_pca(
    S: np.ndarray,
    geometry: GraphGeometry,
    t: float,
    n_components: int = 2,
) -> Embedding:
    """PCA on Z = S K_t^(1/2), the exact feature map of the heat-kernel distance.

    Equivalent to classical MDS on d_t, but deterministic and computed without
    ever forming the distance matrix. The principled default for this project.
    """
    Z = diffusion_features(np.asarray(S, dtype=float), geometry, float(t))
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, singular, _ = np.linalg.svd(Zc, full_matrices=False)
    coords = U[:, :n_components] * singular[:n_components]
    variance = singular**2
    return Embedding(
        name="heat_kernel_pca",
        coords=coords,
        params={"t": float(t), "n_components": n_components},
        info={
            "singular_values": singular[:n_components],
            "variance_explained": float(
                variance[:n_components].sum() / (variance.sum() or np.nan)
            ),
            "features": Z,
        },
    )


# --- stochastic / iterative embeddings -------------------------------------


def nonmetric_mds(
    D,
    n_components: int = 2,
    random_state: int = 0,
    n_init: int = 8,
    max_iter: int = 500,
) -> Embedding:
    """Non-metric MDS: preserves the rank order of distances only.

    Use when the distance matrix is trusted ordinally but not metrically.
    """
    from sklearn.manifold import MDS

    D = _as_distance(D)
    model = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        metric=False,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        normalized_stress="auto",
    )
    coords = model.fit_transform(D)
    return Embedding(
        name="nonmetric_mds",
        coords=np.asarray(coords, dtype=float),
        params={"n_components": n_components, "random_state": random_state},
        info={"sklearn_stress": float(model.stress_)},
    )


def umap_embedding(
    D,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> Embedding:
    """UMAP on the precomputed distance matrix: local neighbourhood structure.

    Main figure candidate when basin separation is the goal. Between-cluster
    distances in the output are not quantitatively interpretable.
    """
    import umap  # optional dependency

    D = _as_distance(D)
    k = _safe_neighbours(D.shape[0], n_neighbors)
    reducer = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=k,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(D)
    return Embedding(
        name="umap",
        coords=np.asarray(coords, dtype=float),
        params={
            "n_components": n_components,
            "n_neighbors": k,
            "min_dist": min_dist,
            "random_state": random_state,
        },
    )


def phate_embedding(
    D,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    random_state: int = 0,
) -> Embedding:
    """PHATE on the precomputed distance matrix: diffusion / potential geometry.

    Gives the smoothest landscape-like view of gradual transitions between
    attractor regions.
    """
    import phate  # optional dependency

    D = _as_distance(D)
    k = _safe_neighbours(D.shape[0], knn)
    operator = phate.PHATE(
        n_components=n_components,
        knn_dist="precomputed_distance",
        knn=k,
        decay=decay,
        random_state=random_state,
        verbose=0,
    )
    coords = operator.fit_transform(D)
    return Embedding(
        name="phate",
        coords=np.asarray(coords, dtype=float),
        params={
            "n_components": n_components,
            "knn": k,
            "decay": decay,
            "random_state": random_state,
        },
    )


def tsne_embedding(
    D,
    n_components: int = 2,
    perplexity: Optional[float] = None,
    random_state: int = 0,
) -> Embedding:
    """t-SNE on the precomputed distance matrix. Exploratory only.

    Global layout is not interpretable, so this is a check on whether basins
    separate at all, never a source of landscape geometry.
    """
    from sklearn.manifold import TSNE

    D = _as_distance(D)
    n = D.shape[0]
    if perplexity is None:
        perplexity = max(2.0, min(30.0, (n - 1) / 3.0))
    model = TSNE(
        n_components=n_components,
        metric="precomputed",
        init="random",
        perplexity=float(perplexity),
        random_state=random_state,
    )
    coords = model.fit_transform(D)
    return Embedding(
        name="tsne",
        coords=np.asarray(coords, dtype=float),
        params={
            "n_components": n_components,
            "perplexity": float(perplexity),
            "random_state": random_state,
        },
    )


def force_directed(
    D,
    n_components: int = 2,
    k_neighbours: int = 5,
    random_state: int = 0,
) -> Embedding:
    """Spring layout of the k-nearest-neighbour graph over attractors.

    Reads the landscape as a network of basins; edge weights decay with the
    attractor distance.
    """
    import networkx as nx

    D = _as_distance(D)
    n = D.shape[0]
    k = _safe_neighbours(n, k_neighbours)
    scale = float(D[D > 0].mean()) if np.any(D > 0) else 1.0
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in np.argsort(D[i])[1 : k + 1]:
            G.add_edge(i, int(j), weight=float(np.exp(-D[i, int(j)] / scale)))
    positions = nx.spring_layout(
        G, dim=n_components, weight="weight", seed=random_state
    )
    coords = np.vstack([positions[i] for i in range(n)])
    return Embedding(
        name="force_directed",
        coords=coords,
        params={
            "n_components": n_components,
            "k_neighbours": k,
            "random_state": random_state,
        },
        info={"n_edges": int(G.number_of_edges())},
    )


# --- elevation --------------------------------------------------------------


def elevation(probabilities, eps: float = 1e-12) -> np.ndarray:
    """Landscape height U(s) = -log(P(s) + eps), with P from MaBoSS.

    Deep wide valleys are robust attractors; shallow ones are marginal.
    """
    p = np.asarray(probabilities, dtype=float)
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    return -np.log(p + eps)


# --- validation -------------------------------------------------------------


def stress(D, coords, normalise: bool = True) -> float:
    """Kruskal stress-1: global distance distortion, 0 is perfect.

    With `normalise`, the embedded distances are first rescaled optimally, so
    methods that only preserve distances up to a scale are not penalised.
    """
    target = upper(_as_distance(D))
    fitted = upper(embedded_distances(coords))
    if normalise:
        denominator = float((fitted**2).sum())
        if denominator > 0:
            fitted = fitted * float((target * fitted).sum() / denominator)
    total = float((target**2).sum()) or np.nan
    return float(np.sqrt(float(((target - fitted) ** 2).sum()) / total))


def spearman_fidelity(D, coords) -> float:
    """Rank correlation between original and embedded pairwise distances."""
    target = upper(_as_distance(D))
    fitted = upper(embedded_distances(coords))
    if np.allclose(target, target[0]) or np.allclose(fitted, fitted[0]):
        return float("nan")
    return float(spearmanr(target, fitted)[0])


def trustworthiness(D, coords, n_neighbors: int = 5) -> float:
    """Share of k-nearest neighbours preserved by the projection, 1 is perfect."""
    from sklearn.manifold import trustworthiness as _trustworthiness

    D = _as_distance(D)
    k = int(max(1, min(n_neighbors, (D.shape[0] - 1) // 2)))
    return float(
        _trustworthiness(
            D, np.asarray(coords, dtype=float), n_neighbors=k, metric="precomputed"
        )
    )


def silhouette(coords, labels) -> float:
    """Separation of known phenotype labels in the embedded space."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        raise ValueError("need at least two distinct labels")
    return float(silhouette_score(np.asarray(coords, dtype=float), labels))


def procrustes_stability(coordinate_sets: Sequence[np.ndarray]) -> float:
    """Mean pairwise Procrustes disparity; 0 means identical maps."""
    sets = [np.asarray(c, dtype=float) for c in coordinate_sets]
    if len(sets) < 2:
        return float("nan")
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            values.append(float(procrustes(sets[i], sets[j])[2]))
    return float(np.mean(values))


def stability_over_seeds(
    embedder: Callable[[int], Embedding],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> float:
    """Procrustes stability of a stochastic embedder across random seeds.

    >>> stability_over_seeds(lambda seed: umap_embedding(D, random_state=seed))
    """
    return procrustes_stability([embedder(int(seed)).coords for seed in seeds])


# --- orchestration ----------------------------------------------------------


def build_embeddings(
    D,
    n_components: int = 2,
    S: Optional[np.ndarray] = None,
    geometry: Optional[GraphGeometry] = None,
    t: Optional[float] = None,
    include: Optional[Iterable[str]] = None,
    random_state: int = 0,
) -> Dict[str, Embedding]:
    """Run every available embedding on one attractor distance matrix.

    `heat_kernel_pca` is only built when S, geometry and t are all given, since
    it works from the state matrix rather than from D. Missing optional
    dependencies are reported and skipped instead of raising.
    """
    D = _as_distance(D)
    wanted = set(include) if include is not None else set(_ALL_METHODS)
    builders: Dict[str, Callable[[], Embedding]] = {
        "classical_mds": lambda: classical_mds(D, n_components),
        "nonmetric_mds": lambda: nonmetric_mds(
            D, n_components, random_state=random_state
        ),
        "umap": lambda: umap_embedding(
            D, n_components, random_state=random_state
        ),
        "phate": lambda: phate_embedding(
            D, n_components, random_state=random_state
        ),
        "tsne": lambda: tsne_embedding(
            D, n_components, random_state=random_state
        ),
        "force_directed": lambda: force_directed(
            D, n_components, random_state=random_state
        ),
    }
    if S is not None and geometry is not None and t is not None:
        builders["heat_kernel_pca"] = lambda: heat_kernel_pca(
            S, geometry, float(t), n_components
        )

    out: Dict[str, Embedding] = {}
    for name in _ALL_METHODS:
        if name not in wanted or name not in builders:
            continue
        try:
            out[name] = builders[name]()
        except Exception as exc:  # keep the comparison going
            print(f"[skip] {name} -> {exc}")
    return out


def score_embeddings(
    embeddings: Dict[str, Embedding],
    D,
    labels: Optional[Sequence[str]] = None,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """One row per embedding: distortion, neighbourhood fidelity, separation.

    Low `stress` and high `spearman_vs_distance` mean the global geometry of D
    survived; high `trustworthiness` means neighbourhoods did. `silhouette`
    requires phenotype labels. Report the chosen method with these numbers.
    """
    D = _as_distance(D)
    rows = []
    for name, item in embeddings.items():
        row: Dict[str, object] = {
            "embedding": name,
            "n_components": item.n_components,
            "deterministic": name not in _STOCHASTIC,
            "stress": stress(D, item.coords),
            "spearman_vs_distance": spearman_fidelity(D, item.coords),
        }
        for key, fn in (
            ("trustworthiness", lambda: trustworthiness(D, item.coords, n_neighbors)),
            ("silhouette", lambda: silhouette(item.coords, labels)),
        ):
            if key == "silhouette" and labels is None:
                continue
            try:
                row[key] = fn()
            except Exception as exc:
                print(f"[skip] {key}:{name} -> {exc}")
                row[key] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("embedding").sort_values("stress")


def compare_embeddings(
    D,
    n_components: int = 2,
    S: Optional[np.ndarray] = None,
    geometry: Optional[GraphGeometry] = None,
    t: Optional[float] = None,
    labels: Optional[Sequence[str]] = None,
    include: Optional[Iterable[str]] = None,
    random_state: int = 0,
) -> Dict[str, object]:
    """Build every embedding of one distance matrix and score them together.

    Returns {"embeddings": {name: Embedding}, "scores": DataFrame}.
    """
    embeddings = build_embeddings(
        D,
        n_components=n_components,
        S=S,
        geometry=geometry,
        t=t,
        include=include,
        random_state=random_state,
    )
    return {
        "embeddings": embeddings,
        "scores": score_embeddings(embeddings, D, labels=labels),
    }