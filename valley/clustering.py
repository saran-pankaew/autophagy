"""Partition the attractors, taking a distance matrix as the only input.

Clustering happens on the distance matrix, never on the 2D embedding: the
embedding's distortion is measured in `embedding.py`, so clustering its
coordinates would inherit that distortion silently. Labels are computed here
and only *displayed*, by `plotting.plot_embedding(clusters=...)`.

Two regimes:

- The heat-kernel distance is induced by a PSD matrix, hence exactly Euclidean
  through Z = S K_t^(1/2). Ward and k-means are then legitimate; pass Z as
  `features`.
- Hamming, weighted Hamming and Wasserstein carry no such guarantee, so only
  the precomputed-distance methods apply to them.

Typical use
-----------
>>> D = matrices["heat_kernel:t=1"]
>>> print(choose_k(D, method="average"))
>>> result = cluster_attractors(D, method="average", k=3)
>>> score_clusterings(build_clusterings(D, k=3), D)

Resolution-aware variant, using the whole t grid instead of one chosen scale:

>>> consensus = consensus_clustering(attractors, model, k=3)
>>> consensus["stability"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import distances as dist
from .bnet import BooleanModel, signed_adjacency, state_matrix
from .geometry import GraphGeometry, weight_matrix

NOISE = -1
_LINKAGES = ("single", "average", "complete", "weighted")
_DISTANCE_METHODS = (
    "single",
    "average",
    "complete",
    "kmedoids",
    "spectral",
    "hdbscan",
    "affinity",
)
_FEATURE_METHODS = ("ward", "kmeans")


# --- container --------------------------------------------------------------


@dataclass
class Clustering:
    """One partition of the attractors, plus how it was obtained.

    `medoids` holds one real attractor index per cluster, so a partition always
    comes with named representatives rather than abstract centroids.
    """

    name: str
    labels: np.ndarray
    medoids: Optional[np.ndarray] = None
    params: Dict[str, object] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def k(self) -> int:
        assigned = self.labels[self.labels != NOISE]
        return int(np.unique(assigned).size)

    @property
    def n_noise(self) -> int:
        return int((self.labels == NOISE).sum())

    def to_series(self, labels: Optional[Sequence[str]] = None) -> pd.Series:
        index = (
            pd.Index(list(labels), name="attractor") if labels is not None else None
        )
        return pd.Series(self.labels, index=index, name="cluster")

    def groups(self, labels: Optional[Sequence[str]] = None) -> Dict[int, List]:
        """cluster -> attractor labels, or positional indices when unnamed."""
        names = (
            list(labels) if labels is not None else list(range(len(self.labels)))
        )
        out: Dict[int, List] = {}
        for position, cluster in enumerate(self.labels):
            out.setdefault(int(cluster), []).append(names[position])
        return dict(sorted(out.items()))


# --- helpers ----------------------------------------------------------------


def _as_distance(D) -> np.ndarray:
    """Validate and symmetrise a pairwise distance matrix."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def linkage_matrix(D, method: str = "average") -> np.ndarray:
    """SciPy linkage on the condensed distances; also the dendrogram source."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    if method not in _LINKAGES:
        raise ValueError(f"{method!r} is not a distance-based linkage")
    return linkage(squareform(_as_distance(D), checks=False), method=method)


def _cut(
    Z: np.ndarray,
    k: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> np.ndarray:
    """Flat labels from a linkage, by cluster count or by height."""
    from scipy.cluster.hierarchy import fcluster

    if k is not None:
        return fcluster(Z, t=int(k), criterion="maxclust").astype(int) - 1
    if distance_threshold is None:
        raise ValueError("give either k or distance_threshold")
    return (
        fcluster(Z, t=float(distance_threshold), criterion="distance").astype(int) - 1
    )


def medoids_of(D, labels: Sequence[int]) -> np.ndarray:
    """Index of the most central real attractor in each cluster."""
    D = _as_distance(D)
    labels = np.asarray(labels, dtype=int)
    out = []
    for level in sorted({int(v) for v in labels if int(v) != NOISE}):
        members = np.flatnonzero(labels == level)
        costs = D[np.ix_(members, members)].sum(axis=1)
        out.append(int(members[int(np.argmin(costs))]))
    return np.array(out, dtype=int)


# --- k-medoids --------------------------------------------------------------


def _kmedoids_seed(D: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ style seeding, on the distance matrix rather than on points."""
    n = D.shape[0]
    chosen = [int(rng.integers(n))]
    while len(chosen) < k:
        nearest = D[:, chosen].min(axis=1) ** 2
        nearest[chosen] = 0.0
        total = float(nearest.sum())
        if total <= 0:
            remaining = [i for i in range(n) if i not in chosen]
            chosen.append(int(rng.choice(remaining)))
            continue
        chosen.append(int(rng.choice(n, p=nearest / total)))
    return np.array(chosen, dtype=int)


def kmedoids(
    D,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    weights: Optional[Sequence[float]] = None,
    random_state: int = 0,
):
    """k-medoids (Voronoi iteration) on a precomputed distance matrix.

    Unlike k-means this never averages states, so each cluster is represented
    by an actual attractor. `weights` (e.g. MaBoSS attractor probabilities)
    let probable attractors decide where the medoid sits.

    Returns (labels, medoid_indices, total_cost).
    """
    D = _as_distance(D)
    n = D.shape[0]
    k = int(min(max(1, k), n))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(random_state)

    best = None
    for _ in range(max(1, n_init)):
        medoids = _kmedoids_seed(D, k, rng)
        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            updated = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                if members.size == 0:
                    continue
                block = D[np.ix_(members, members)] * w[members][None, :]
                updated[cluster] = members[int(np.argmin(block.sum(axis=1)))]
            if np.array_equal(np.sort(updated), np.sort(medoids)):
                medoids = updated
                break
            medoids = updated
        labels = np.argmin(D[:, medoids], axis=1)
        cost = float((w * D[np.arange(n), medoids[labels]]).sum())
        if best is None or cost < best[0]:
            best = (cost, labels.astype(int), medoids.copy())

    cost, labels, medoids = best
    return labels, medoids, cost


# --- one partition ----------------------------------------------------------


def cluster_attractors(
    D=None,
    method: str = "average",
    k: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    features: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    min_cluster_size: int = 3,
    random_state: int = 0,
    **kwargs,
) -> Clustering:
    """Cluster the attractors of one distance matrix.

    Pass `D` for the distance-based methods, and `features` (typically
    Z = S K_t^(1/2)) for the Euclidean-only ones. The linkages accept either
    `k` or `distance_threshold`; kmedoids, kmeans, ward and spectral need `k`.
    Missing optional dependencies raise, so `build_clusterings` can skip them.
    """
    if method in _LINKAGES:
        Z = linkage_matrix(D, method=method)
        labels = _cut(Z, k=k, distance_threshold=distance_threshold)
        info: Dict[str, object] = {"linkage": Z, "euclidean_required": False}

    elif method == "kmedoids":
        if k is None:
            raise ValueError("kmedoids requires k")
        labels, seeds, cost = kmedoids(
            D, k, weights=weights, random_state=random_state, **kwargs
        )
        info = {"cost": cost, "seed_medoids": seeds, "euclidean_required": False}

    elif method in _FEATURE_METHODS:
        if features is None:
            raise ValueError(
                f"{method!r} is Euclidean-only; pass features, e.g. "
                "Z = diffusion_features(S, geometry, t)"
            )
        X = np.asarray(features, dtype=float)
        if method == "ward":
            from sklearn.cluster import AgglomerativeClustering

            estimator = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
                distance_threshold=distance_threshold,
            )
        else:
            from sklearn.cluster import KMeans

            if k is None:
                raise ValueError("kmeans requires k")
            estimator = KMeans(n_clusters=int(k), n_init=10, random_state=random_state)
        labels = estimator.fit_predict(X).astype(int)
        info = {"euclidean_required": True}

    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        Dm = _as_distance(D)
        offdiag = Dm[Dm > 0]
        scale = float(sigma) if sigma else (float(np.median(offdiag)) or 1.0)
        affinity = np.exp(-(Dm**2) / (2.0 * scale**2))
        labels = (
            SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=random_state,
                **kwargs,
            )
            .fit_predict(affinity)
            .astype(int)
        )
        info = {"sigma": scale, "euclidean_required": False}

    elif method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        labels = (
            HDBSCAN(
                metric="precomputed",
                min_cluster_size=int(min_cluster_size),
                **kwargs,
            )
            .fit_predict(_as_distance(D))
            .astype(int)
        )
        info = {
            "min_cluster_size": int(min_cluster_size),
            "euclidean_required": False,
        }

    elif method == "affinity":
        from sklearn.cluster import AffinityPropagation

        estimator = AffinityPropagation(
            affinity="precomputed", random_state=random_state, **kwargs
        )
        labels = estimator.fit_predict(-_as_distance(D)).astype(int)
        info = {
            "exemplars": np.asarray(estimator.cluster_centers_indices_, dtype=int),
            "euclidean_required": False,
        }

    else:
        known = sorted(set(_DISTANCE_METHODS + _FEATURE_METHODS + _LINKAGES))
        raise ValueError(f"unknown method {method!r}; choose from {known}")

    labels = np.asarray(labels, dtype=int)
    centres = medoids_of(D, labels) if D is not None else None
    return Clustering(
        name=method,
        labels=labels,
        medoids=centres,
        params={
            "k": k,
            "distance_threshold": distance_threshold,
            "random_state": random_state,
        },
        info=info,
    )


def build_clusterings(
    D,
    k: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
    include: Optional[Iterable[str]] = None,
    random_state: int = 0,
) -> Dict[str, Clustering]:
    """Run every available method on one distance matrix.

    Euclidean-only methods are attempted only when `features` is given.
    Missing optional dependencies are reported and skipped, never raised, so
    the comparison always returns something.
    """
    candidates = _DISTANCE_METHODS + _FEATURE_METHODS
    wanted = set(include) if include is not None else set(candidates)
    out: Dict[str, Clustering] = {}
    for method in candidates:
        if method not in wanted:
            continue
        if method in _FEATURE_METHODS and features is None:
            continue
        try:
            out[method] = cluster_attractors(
                D,
                method=method,
                k=k,
                features=features,
                weights=weights,
                random_state=random_state,
            )
        except Exception as exc:  # keep the comparison going
            print(f"[skip] {method} -> {exc}")
    return out


# --- scoring ----------------------------------------------------------------


def _within_over_between(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-cluster distance over mean between-cluster distance."""
    same = labels[:, None] == labels[None, :]
    offdiag = ~np.eye(labels.size, dtype=bool)
    within = D[same & offdiag]
    between = D[(~same) & offdiag]
    if within.size == 0 or between.size == 0:
        return float("nan")
    return float(within.mean() / (between.mean() or np.nan))


def score_clusterings(
    clusterings: Dict[str, Clustering],
    D,
    reference_labels: Optional[Sequence] = None,
) -> pd.DataFrame:
    """One row per partition: size, separation, and external agreement.

    `reference_labels` is any external partition worth comparing against:
    curated phenotypes, or the trap-space / succession-diagram grouping from
    biobalm. Agreement validates the metric; disagreement is the result that
    needs an explanation.
    """
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    Dm = _as_distance(D)
    rows = []
    for name, item in clusterings.items():
        row: Dict[str, object] = {
            "clustering": name,
            "k": item.k,
            "n_noise": item.n_noise,
        }
        assigned = item.labels != NOISE
        try:
            if np.unique(item.labels[assigned]).size < 2:
                raise ValueError("fewer than two clusters")
            row["silhouette"] = float(
                silhouette_score(
                    Dm[np.ix_(assigned, assigned)],
                    item.labels[assigned],
                    metric="precomputed",
                )
            )
        except Exception as exc:
            print(f"[skip] silhouette:{name} -> {exc}")
            row["silhouette"] = float("nan")
        row["within_over_between"] = _within_over_between(Dm, item.labels)
        if reference_labels is not None:
            row["ari_vs_reference"] = float(
                adjusted_rand_score(np.asarray(reference_labels), item.labels)
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .set_index("clustering")
        .sort_values("silhouette", ascending=False)
    )


def choose_k(
    D,
    method: str = "average",
    k_values: Sequence[int] = tuple(range(2, 8)),
    **kwargs,
) -> pd.DataFrame:
    """Silhouette as a function of k. Report the curve, not only its argmax."""
    Dm = _as_distance(D)
    limit = Dm.shape[0] - 1
    rows = []
    for k in k_values:
        if not 2 <= int(k) <= limit:
            continue
        try:
            item = cluster_attractors(Dm, method=method, k=int(k), **kwargs)
        except Exception as exc:
            print(f"[skip] k={k} -> {exc}")
            continue
        scores = score_clusterings({f"k={k}": item}, Dm).iloc[0]
        rows.append(
            {
                "k": int(k),
                "silhouette": float(scores["silhouette"]),
                "within_over_between": float(scores["within_over_between"]),
            }
        )
    return pd.DataFrame(rows)


# --- consensus across the diffusion scale -----------------------------------


def consensus_matrix(clusterings) -> np.ndarray:
    """Co-association matrix C: how often two attractors share a cluster.

    Unassigned (-1) attractors never count as sharing a cluster, so a point
    that stays noise at every scale ends up far from everything in 1 - C.
    """
    items = list(clusterings.values()) if isinstance(clusterings, dict) else list(
        clusterings
    )
    if not items:
        raise ValueError("no clusterings to combine")
    first = items[0].labels if isinstance(items[0], Clustering) else items[0]
    n = len(np.asarray(first))
    C = np.zeros((n, n), dtype=float)
    for item in items:
        labels = np.asarray(
            item.labels if isinstance(item, Clustering) else item, dtype=int
        )
        shared = (labels[:, None] == labels[None, :]) & (labels[:, None] != NOISE)
        C += shared.astype(float)
    C /= float(len(items))
    np.fill_diagonal(C, 1.0)
    return C


def consensus_clustering(
    attractors: pd.DataFrame,
    model: BooleanModel,
    t_values: Optional[Sequence[float]] = None,
    k: Optional[int] = None,
    per_scale_k: Optional[int] = None,
    per_scale_method: str = "average",
    method: str = "average",
    symmetrisation: str = "absolute",
    laplacian_kind: str = "combinatorial",
) -> Dict[str, object]:
    """Cluster at every diffusion scale, then cluster the agreement.

    t becomes a resolution axis instead of a nuisance parameter: pairs that
    co-cluster at every scale are robust groups, pairs that co-cluster only at
    large t are joined by pathway-level rather than node-level similarity.

    Returns {consensus, per_scale, clustering, stability, geometry}.
    """
    from sklearn.metrics import adjusted_rand_score

    S, _ = state_matrix(attractors, model)
    A, P = signed_adjacency(model)
    W = weight_matrix(A, P, mode=symmetrisation)
    geometry = GraphGeometry(W, kind=laplacian_kind)
    if t_values is None:
        t_values = geometry.suggest_t_grid()

    target_k = per_scale_k or k
    if target_k is None:
        raise ValueError("give k (or per_scale_k) so partitions are comparable across t")

    per_scale: Dict[float, Clustering] = {}
    for t in t_values:
        D = dist.heat_kernel(S, geometry, float(t))
        try:
            per_scale[float(t)] = cluster_attractors(
                D, method=per_scale_method, k=int(target_k)
            )
        except Exception as exc:
            print(f"[skip] t={float(t):.4g} -> {exc}")
    if not per_scale:
        raise ValueError("no diffusion scale produced a usable partition")

    C = consensus_matrix(per_scale)
    final = cluster_attractors(1.0 - C, method=method, k=int(k or target_k))

    scales = list(per_scale)
    stability = pd.DataFrame(
        [
            {
                "t": scale,
                "ari_vs_consensus": float(
                    adjusted_rand_score(final.labels, per_scale[scale].labels)
                ),
                "ari_vs_smallest_t": float(
                    adjusted_rand_score(
                        per_scale[scales[0]].labels, per_scale[scale].labels
                    )
                ),
            }
            for scale in scales
        ]
    )
    return {
        "consensus": C,
        "per_scale": per_scale,
        "clustering": final,
        "stability": stability,
        "geometry": geometry,
    }


def bootstrap_stability(
    attractors: pd.DataFrame,
    model: BooleanModel,
    t: float,
    k: int,
    method: str = "average",
    n_replicates: int = 25,
    dropout: float = 0.1,
    random_state: int = 0,
) -> Dict[str, object]:
    """Mean ARI between the full partition and node-dropout replicates.

    Dropping a fraction of the nodes perturbs the geometry, not the attractor
    set, so labels stay comparable. A low mean ARI means the clusters hang on
    a few nodes and should not be reported as phenotypes.
    """
    from sklearn.metrics import adjusted_rand_score

    S, _ = state_matrix(attractors, model)
    A, P = signed_adjacency(model)
    geometry = GraphGeometry(weight_matrix(A, P))
    reference = cluster_attractors(
        dist.heat_kernel(S, geometry, float(t)), method=method, k=int(k)
    )

    rng = np.random.default_rng(random_state)
    n_nodes = S.shape[1]
    keep_count = max(2, int(round(n_nodes * (1.0 - float(dropout)))))
    scores = []
    for _ in range(int(n_replicates)):
        keep = np.sort(rng.choice(n_nodes, size=keep_count, replace=False))
        try:
            Wk = weight_matrix(A[np.ix_(keep, keep)], P[np.ix_(keep, keep)])
            Dk = dist.heat_kernel(S[:, keep], GraphGeometry(Wk), float(t))
            item = cluster_attractors(Dk, method=method, k=int(k))
        except Exception as exc:
            print(f"[skip] replicate -> {exc}")
            continue
        scores.append(float(adjusted_rand_score(reference.labels, item.labels)))
    return {
        "reference": reference,
        "ari": np.array(scores, dtype=float),
        "mean_ari": float(np.mean(scores)) if scores else float("nan"),
        "keep_count": keep_count,
    }


# --- interpretation ---------------------------------------------------------


def cluster_activity(
    attractors: pd.DataFrame,
    labels: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    drop_noise: bool = True,
) -> pd.DataFrame:
    """Mean node activity per cluster: one row per cluster, one column per node.

    With `weights` (e.g. MaBoSS attractor probabilities) the mean is weighted,
    so a cluster profile is dominated by its probable attractors. The relative
    mass of each cluster is attached as `frame.attrs["mass"]`.
    """
    frame = pd.DataFrame(attractors).astype(float)
    labels = np.asarray(labels, dtype=int)
    if labels.shape[0] != frame.shape[0]:
        raise ValueError("one label per attractor is required")
    w = None if weights is None else np.asarray(weights, dtype=float)

    rows: Dict[int, pd.Series] = {}
    mass: Dict[int, float] = {}
    for level in sorted({int(v) for v in labels}):
        if drop_noise and level == NOISE:
            continue
        members = np.flatnonzero(labels == level)
        block = frame.iloc[members]
        if w is None:
            rows[level] = block.mean(axis=0)
            mass[level] = float(members.size) / float(frame.shape[0])
        else:
            wm = w[members]
            total = float(wm.sum())
            rows[level] = pd.Series(
                (block.to_numpy(dtype=float) * wm[:, None]).sum(axis=0)
                / (total or np.nan),
                index=frame.columns,
            )
            mass[level] = total / (float(w.sum()) or np.nan)

    out = pd.DataFrame(rows).T
    out.index.name = "cluster"
    out.attrs["mass"] = mass
    return out


def cluster_signatures(
    attractors: pd.DataFrame,
    labels: Sequence[int],
    top: int = 5,
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Nodes that discriminate each cluster, i.e. automatic cluster names.

    Score = this cluster's mean activity minus the mean over all other
    clusters. The top positive and top negative nodes are what a figure caption
    should quote; `frame.attrs["between_cluster_spread"]` ranks nodes globally.
    """
    profiles = cluster_activity(attractors, labels, weights=weights)
    if profiles.shape[0] < 2:
        raise ValueError("need at least two clusters to contrast")
    counts = pd.Series(np.asarray(labels, dtype=int)).value_counts()

    rows = []
    for level in profiles.index:
        contrast = (
            profiles.loc[level] - profiles.drop(index=level).mean(axis=0)
        ).sort_values()
        up = contrast.tail(int(top))[::-1]
        down = contrast.head(int(top))
        rows.append(
            {
                "cluster": level,
                "n_attractors": int(counts.get(level, 0)),
                "mass": float(profiles.attrs["mass"].get(level, float("nan"))),
                "on": ", ".join(
                    f"{node} (+{value:.2f})" for node, value in up.items() if value > 0
                ),
                "off": ", ".join(
                    f"{node} ({value:.2f})" for node, value in down.items() if value < 0
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("cluster")
    frame.attrs["between_cluster_spread"] = profiles.var(axis=0).sort_values(
        ascending=False
    )
    return frame


def cluster_groups(
    attractors: pd.DataFrame,
    labels: Sequence[int],
    drop_noise: bool = False,
) -> Dict[str, pd.DataFrame]:
    """{"cluster 0": sub-frame, ...}, ready for `plot_attractor_panels`."""
    frame = pd.DataFrame(attractors)
    labels = np.asarray(labels, dtype=int)
    out: Dict[str, pd.DataFrame] = {}
    for level in sorted({int(v) for v in labels}):
        if drop_noise and level == NOISE:
            continue
        name = "noise" if level == NOISE else f"cluster {level}"
        out[name] = frame.iloc[np.flatnonzero(labels == level)]
    return out