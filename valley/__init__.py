"""valley — attractor landscape distances for Boolean models.

Proof-of-concept package: given a .bnet model and a table of reachable
attractors, build and compare candidate pairwise distance matrices.

Typical use
-----------
>>> import pandas as pd
>>> import valley
>>> model = valley.read_bnet("model.bnet")
>>> attractors = pd.read_csv("attractors.csv", index_col=0)
>>> matrices = valley.build_distance_matrices(attractors, model)
>>> valley.compare_matrices(matrices)

A one-call convenience wrapper is also available:

>>> report = valley.run_comparison("model.bnet", attractors)
>>> report["summary"]
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

__version__ = "0.1.0.dev0"
__author__ = "Saran Pankaew"

# --- model input and influence graph ---------------------------------------
from .bnet import (
    BooleanModel,
    parse_regulators,
    read_bnet,
    signed_adjacency,
    state_matrix,
)

# --- node-level geometry ----------------------------------------------------
from .geometry import (
    GraphGeometry,
    graph_from_weights,
    laplacian,
    shortest_path_cost,
    weight_matrix,
)

# --- distance definitions ---------------------------------------------------
from . import distances
from .distances import (
    diffusion_features,
    hamming,
    heat_kernel,
    module_coarse_grained,
    quadratic_form_distance,
    reference_axis,
    set_distance,
    wasserstein,
    weighted_hamming,
)

# --- comparison and validation ---------------------------------------------
from . import compare
from .compare import (
    build_distance_matrices,
    centrality_weights,
    compare_matrices,
    cross_correlation,
    mantel,
    scc_modules,
    separation_scores,
    sweep_t,
    upper,
)

# --- clustering -------------------------------------------------------------
from . import clustering
from .clustering import (
    Clustering,
    bootstrap_stability,
    build_clusterings,
    choose_k,
    cluster_activity,
    cluster_attractors,
    cluster_groups,
    cluster_signatures,
    consensus_clustering,
    consensus_matrix,
    kmedoids,
    linkage_matrix,
    medoids_of,
    score_clusterings,
)

# --- embedding --------------------------------------------------------------
from . import embedding
from .embedding import (
    Embedding,
    build_embeddings,
    classical_mds,
    compare_embeddings,
    elevation,
    heat_kernel_pca,
    nonmetric_mds,
    score_embeddings,
    phate_embedding,
)

# --- plotting ---------------------------------------------------------------
from . import plotting
from .plotting import (
    average_activity,
    influence_graph,
    layout_positions,
    plot_attractor_on_network,
    plot_attractor_panels,
    plot_embedding,
    plot_embedding_grid,
    plot_influence_graph,
    plot_landscape,
)

__all__ = [
    "__version__",
    # submodules
    "distances",
    "compare",
    # model / graph
    "BooleanModel",
    "read_bnet",
    "parse_regulators",
    "signed_adjacency",
    "state_matrix",
    # geometry
    "GraphGeometry",
    "weight_matrix",
    "laplacian",
    "graph_from_weights",
    "shortest_path_cost",
    # distances
    "quadratic_form_distance",
    "hamming",
    "weighted_hamming",
    "heat_kernel",
    "diffusion_features",
    "module_coarse_grained",
    "reference_axis",
    "wasserstein",
    "set_distance",
    # comparison
    "build_distance_matrices",
    "compare_matrices",
    "cross_correlation",
    "separation_scores",
    "sweep_t",
    "mantel",
    "upper",
    "centrality_weights",
    "scc_modules",
    # clustering
    "clustering",
    "Clustering",
    "cluster_attractors",
    "build_clusterings",
    "score_clusterings",
    "choose_k",
    "linkage_matrix",
    "kmedoids",
    "medoids_of",
    "consensus_matrix",
    "consensus_clustering",
    "bootstrap_stability",
    "cluster_activity",
    "cluster_signatures",
    "cluster_groups",
    # embedding
    "embedding",
    "Embedding",
    "classical_mds",
    "nonmetric_mds",
    "heat_kernel_pca",
    "build_embeddings",
    "score_embeddings",
    "compare_embeddings",
    "elevation",
    "phate_embedding",
    # plotting
    "plotting",
    "plot_embedding",
    "plot_embedding_grid",
    "plot_landscape",
    "influence_graph",
    "layout_positions",
    "plot_influence_graph",
    "average_activity",
    "plot_attractor_on_network",
    "plot_attractor_panels",
    # convenience
    "load_geometry",
    "run_comparison",
]


def load_geometry(
    model: BooleanModel,
    symmetrisation: str = "absolute",
    laplacian_kind: str = "combinatorial",
) -> GraphGeometry:
    """Build the node-level GraphGeometry directly from a BooleanModel."""
    A, P = signed_adjacency(model)
    W = weight_matrix(A, P, mode=symmetrisation)
    return GraphGeometry(W, kind=laplacian_kind)


def run_comparison(
    bnet_path: str,
    attractors,
    t_values: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    include_wasserstein: bool = False,
) -> Dict[str, object]:
    """One-call proof of concept: model + attractors -> all comparison tables.

    Returns a dict with keys:
      model, geometry, matrices, summary, cross_correlation, t_sweep
      and separation (only when `labels` is given).
    """
    model = read_bnet(bnet_path)
    geometry = load_geometry(model)
    matrices = build_distance_matrices(
        attractors,
        model,
        t_values=t_values,
        include_wasserstein=include_wasserstein,
    )
    report: Dict[str, object] = {
        "model": model,
        "geometry": geometry,
        "matrices": matrices,
        "summary": compare_matrices(matrices),
        "cross_correlation": cross_correlation(matrices),
        "t_sweep": sweep_t(attractors, model, t_values=t_values),
    }
    if labels is not None:
        report["separation"] = separation_scores(matrices, labels)
    return report