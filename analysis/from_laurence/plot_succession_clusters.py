"""Colour terminal succession-diagram nodes by attractor cluster."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd


def terminal_cluster_labels(
    diagram: nx.DiGraph,
    attractors: pd.DataFrame,
    clusters: Sequence[int],
) -> pd.DataFrame:
    """Match each terminal node's fixed state to compatible attractors.

    The succession diagram stores fixed variables in each node's ``space``
    attribute.  Attractor entries equal to 0.5 are wildcards and therefore
    match either fixed value.  A terminal is assigned a cluster only when all
    matching attractors have the same label.
    """
    labels = np.asarray(clusters)
    if len(labels) != len(attractors):
        raise ValueError("clusters must contain one label per attractor")

    states = attractors.copy()
    states.columns = states.columns.map(str)
    records = []

    for node, data in diagram.nodes(data=True):
        if diagram.out_degree(node) != 0:
            continue

        space: Mapping[str, object] = data.get("space", {})
        matches = np.ones(len(states), dtype=bool)
        for variable, value in space.items():
            variable = str(variable)
            if variable not in states:
                matches &= False
                break
            activity = states[variable].to_numpy(dtype=float)
            fixed_value = float(value)
            matches &= np.isclose(activity, fixed_value) | np.isclose(activity, 0.5)

        matched_labels = np.unique(labels[matches])
        cluster = int(matched_labels[0]) if len(matched_labels) == 1 else None
        records.append(
            {
                "node": node,
                "cluster": cluster,
                "n_matches": int(matches.sum()),
                "status": (
                    "assigned"
                    if cluster is not None
                    else "unmatched"
                    if not matches.any()
                    else "mixed"
                ),
            }
        )

    return pd.DataFrame.from_records(records).set_index("node")


def plot_succession_clusters(
    diagram: nx.DiGraph,
    attractors: pd.DataFrame,
    clusters: Sequence[int],
    *,
    ax: plt.Axes | None = None,
    pos: Mapping | None = None,
    cmap: str = "tab10",
    seed: int = 42,
) -> tuple[plt.Axes, pd.DataFrame]:
    """Draw a succession diagram with terminal nodes filled by cluster.

    Returns the axes and a terminal-node mapping table with ``cluster``,
    ``n_matches``, and ``status`` columns.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 10))
    if pos is None:
        pos = nx.kamada_kawai_layout(diagram)

    mapping = terminal_cluster_labels(diagram, attractors, clusters)
    terminal_nodes = set(mapping.index)
    assigned = mapping[mapping["status"] == "assigned"]
    cluster_ids = sorted(assigned["cluster"].unique())
    colour_map = plt.get_cmap(cmap)
    cluster_colours = {
        cluster: colour_map(index % colour_map.N)
        for index, cluster in enumerate(cluster_ids)
    }

    node_colours = []
    for node in diagram.nodes:
        if node not in terminal_nodes:
            node_colours.append("#d9d9d9")
        elif mapping.loc[node, "status"] == "assigned":
            node_colours.append(cluster_colours[mapping.loc[node, "cluster"]])
        elif mapping.loc[node, "status"] == "mixed":
            node_colours.append("#fdae61")
        else:
            node_colours.append("#ffffff")

    nx.draw_networkx(
        diagram,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color=node_colours,
        node_size=700,
        font_size=8,
        arrows=True,
        edgecolors="#4d4d4d",
        linewidths=0.7,
    )
    legend = [
        Patch(facecolor=cluster_colours[cluster], edgecolor="#4d4d4d", label=f"cluster {cluster}")
        for cluster in cluster_ids
    ]
    if (mapping["status"] == "mixed").any():
        legend.append(Patch(facecolor="#fdae61", edgecolor="#4d4d4d", label="mixed clusters"))
    if (mapping["status"] == "unmatched").any():
        legend.append(Patch(facecolor="#ffffff", edgecolor="#4d4d4d", label="unmatched"))
    if legend:
        ax.legend(handles=legend, title="terminal attractors", loc="best")
    ax.set_axis_off()
    return ax, mapping