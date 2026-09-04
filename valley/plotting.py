"""Figures for the attractor landscape.

Three sections, in order:

1. Embedding plots       -- scatter of the 2D / 3D projections, with the
                            potential U(s) available as the third axis.
2. Influence graph plots -- networkx drawing of the model structure, with the
                            layout and every node / edge size under control.
3. Attractors on network -- node colour set by the activity of one attractor,
                            or by an aggregate activity over a set of them:
                            plain mean, probability-weighted mean, median, or
                            the fraction of attractors holding the node on.

Conventions used throughout: every function draws on a matplotlib Axes, accepts
`ax` so panels can be composed by the caller, returns that Axes, and forwards
unrecognised keyword arguments to the underlying matplotlib / networkx call.
Defaults are therefore a starting point, never a constraint.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .bnet import BooleanModel, signed_adjacency
from .embedding import Embedding

ActivityInput = Union[pd.Series, pd.DataFrame, np.ndarray, Mapping, Sequence]

_SIGN_COLOURS = {1.0: "#238b45", -1.0: "#cb181d", 0.0: "#969696"}
_SIGN_LABELS = {1.0: "activation", -1.0: "inhibition", 0.0: "dual / unsigned"}
_LAYOUTS = {
    "spring": nx.spring_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout,
    "shell": nx.shell_layout,
    "spectral": nx.spectral_layout,
    "spiral": nx.spiral_layout,
    "random": nx.random_layout,
}
_SEEDED_LAYOUTS = {"spring", "random"}


# --- shared helpers ---------------------------------------------------------


def _new_axes(
    ax: Optional[plt.Axes] = None,
    figsize: Sequence[float] = (6.0, 6.0),
    projection: Optional[str] = None,
) -> plt.Axes:
    """Return the given Axes, or open a new figure with one."""
    if ax is not None:
        return ax
    figure = plt.figure(figsize=tuple(figsize))
    return figure.add_subplot(111, projection=projection)


def _coords_of(embedding) -> np.ndarray:
    coords = (
        embedding.coords
        if isinstance(embedding, Embedding)
        else np.asarray(embedding, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("need at least two embedding dimensions to plot")
    return coords


def _scatter(ax, xy: np.ndarray, z: Optional[np.ndarray] = None, **kwargs):
    if z is None:
        return ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
    return ax.scatter(xy[:, 0], xy[:, 1], z, **kwargs)


def _cluster_names(clusters: Sequence) -> np.ndarray:
    """Cluster labels as strings, with -1 rendered as unassigned noise."""
    values = np.asarray(clusters)
    if values.dtype.kind in "iu":
        return np.array(
            ["noise" if int(v) < 0 else f"cluster {int(v)}" for v in values]
        )
    return values.astype(str)


def _draw_cluster_overlays(
    ax,
    coords: np.ndarray,
    clusters: Sequence,
    hulls: bool = True,
    annotate: bool = True,
    font_size: int = 8,
    hull_alpha: float = 0.12,
) -> None:
    """Shade each cluster's convex hull and label it at its centroid.

    Purely cosmetic: the partition is computed on the distance matrix in
    `clustering.py` and only displayed here, so these overlays never define a
    cluster. 2D only; degenerate or noise groups are skipped with a message.
    """
    values = _cluster_names(clusters)
    for level in pd.unique(values):
        if str(level) == "noise":
            continue
        points = coords[values == level][:, :2]
        if hulls and points.shape[0] >= 3:
            try:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(points)
                ax.fill(
                    points[hull.vertices, 0],
                    points[hull.vertices, 1],
                    alpha=hull_alpha,
                    zorder=0,
                )
            except Exception as exc:  # collinear or degenerate cluster
                print(f"[skip] hull:{level} -> {exc}")
        if annotate:
            centroid = points.mean(axis=0)
            ax.annotate(
                str(level),
                (centroid[0], centroid[1]),
                fontsize=font_size + 1,
                ha="center",
                va="center",
                alpha=0.55,
                zorder=1,
            )


def _weighted_mean(frame: pd.DataFrame, weights: Sequence[float]) -> pd.Series:
    """Weighted column means, e.g. with MaBoSS attractor probabilities."""
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != frame.shape[0]:
        raise ValueError("weights length must match the number of states")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must not sum to zero")
    return pd.Series(
        (frame.to_numpy(dtype=float) * w[:, None]).sum(axis=0) / total,
        index=frame.columns,
    )


# === 1. embedding plots =====================================================


def plot_embedding(
    embedding,
    labels: Optional[Sequence[str]] = None,
    colour_by: Optional[Sequence] = None,
    colour_label: Optional[str] = None,
    clusters: Optional[Sequence] = None,
    cluster_hulls: bool = True,
    annotate_clusters: bool = True,
    height: Optional[Sequence[float]] = None,
    projection: str = "auto",
    ax: Optional[plt.Axes] = None,
    figsize: Sequence[float] = (6.0, 6.0),
    dot_size: Union[float, Sequence[float]] = 60.0,
    colour: str = "#4c72b0",
    cmap: str = "viridis",
    alpha: float = 0.9,
    edgecolor: str = "white",
    linewidth: float = 0.5,
    annotate: bool = False,
    font_size: int = 8,
    legend: bool = True,
    colorbar: bool = True,
    title: Optional[str] = None,
    grid: bool = True,
    **scatter_kwargs,
) -> plt.Axes:
    """Scatter one embedding, in 2D or 3D.

    Parameters worth knowing
    ------------------------
    embedding : Embedding or (n, k) array of coordinates.
    colour_by : per-attractor values. Numeric input gives a colour map plus a
        colour bar; anything else is treated as categorical and gets a legend.
    clusters : per-attractor labels from `clustering.cluster_attractors`.
        Shorthand for a categorical `colour_by` that also shades each cluster's
        convex hull and labels it at its centroid; -1 is drawn as unassigned
        noise without a hull. Mutually exclusive with `colour_by`. The partition
        is computed on the distance matrix and only displayed here, so the
        overlays never define a cluster.
    height : per-attractor elevation, e.g. `elevation(probabilities)`. When
        given, it becomes the z axis and the plot is a landscape.
    projection : "2d", "3d", or "auto" (3D when `height` is given or when the
        embedding has three components).
    dot_size : scalar or one size per attractor.

    Extra keyword arguments go straight to `ax.scatter`, so anything
    matplotlib accepts (marker, norm, vmin, vmax, zorder, ...) works here.
    """
    coords = _coords_of(embedding)
    if projection == "auto":
        projection = "3d" if (height is not None or coords.shape[1] >= 3) else "2d"
    if projection not in {"2d", "3d"}:
        raise ValueError("projection must be '2d', '3d' or 'auto'")

    z = None
    z_label = None
    if projection == "3d":
        if height is not None:
            z = np.asarray(height, dtype=float)
            z_label = "U"
        elif coords.shape[1] >= 3:
            z = coords[:, 2]
            z_label = "dim3"
        else:
            raise ValueError("3D plot needs `height` or a third component")

    ax = _new_axes(ax, figsize, projection="3d" if projection == "3d" else None)

    style = {
        "s": dot_size,
        "alpha": alpha,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
    }
    style.update(scatter_kwargs)

    if clusters is not None:
        if colour_by is not None:
            raise ValueError("pass either `clusters` or `colour_by`, not both")
        colour_by = _cluster_names(clusters)
        colour_label = colour_label or "cluster"

    values = None if colour_by is None else np.asarray(colour_by)
    if values is None:
        _scatter(ax, coords, z, color=colour, **style)
    elif values.dtype.kind in "fiub":
        handle = _scatter(ax, coords, z, c=values.astype(float), cmap=cmap, **style)
        if colorbar:
            ax.figure.colorbar(
                handle, ax=ax, shrink=0.75, label=colour_label or "value"
            )
    else:
        for level in pd.unique(values):
            mask = values == level
            _scatter(
                ax,
                coords[mask],
                None if z is None else z[mask],
                label=str(level),
                **style,
            )
        if legend:
            ax.legend(fontsize=font_size, frameon=False, title=colour_label)

    if annotate and labels is not None:
        for index, name in enumerate(labels):
            if z is None:
                ax.annotate(
                    str(name),
                    (coords[index, 0], coords[index, 1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=font_size,
                )
            else:
                ax.text(
                    coords[index, 0],
                    coords[index, 1],
                    z[index],
                    str(name),
                    fontsize=font_size,
                )

    if clusters is not None and coords.shape[1] >= 2 and projection == "2d":
        _draw_cluster_overlays(
            ax,
            coords,
            clusters,
            hulls=cluster_hulls,
            annotate=annotate_clusters,
            font_size=font_size,
        )

    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    if z_label is not None:
        ax.set_zlabel(z_label)
    if title is None and isinstance(embedding, Embedding):
        title = embedding.name
    if title:
        ax.set_title(title, fontsize=font_size + 3)
    if grid and projection == "2d":
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
    return ax


def plot_embedding_grid(
    embeddings: Dict[str, Embedding],
    ncols: int = 3,
    panel_size: Sequence[float] = (4.0, 4.0),
    scores: Optional[pd.DataFrame] = None,
    **plot_kwargs,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """One panel per embedding method, for a side-by-side comparison.

    Pass the `score_embeddings` table as `scores` to carry each method's stress
    in its panel title. Panels are 2D unless `projection` says otherwise.
    """
    plot_kwargs.setdefault("projection", "2d")
    names = list(embeddings)
    if not names:
        raise ValueError("no embeddings to plot")
    ncols = int(max(1, min(ncols, len(names))))
    nrows = int(np.ceil(len(names) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    flat = axes.ravel()
    for index, name in enumerate(names):
        title = name
        if scores is not None and name in scores.index and "stress" in scores:
            title = "{} (stress {:.3f})".format(
                name, float(scores.loc[name, "stress"])
            )
        plot_embedding(embeddings[name], ax=flat[index], title=title, **plot_kwargs)
    for spare in flat[len(names) :]:
        spare.set_axis_off()
    figure.tight_layout()
    return figure, {name: flat[i] for i, name in enumerate(names)}


def plot_landscape(
    embedding,
    height: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Sequence[float] = (7.0, 6.0),
    dot_size: float = 60.0,
    surface: bool = True,
    surface_alpha: float = 0.35,
    cmap: str = "viridis",
    **plot_kwargs,
) -> plt.Axes:
    """The landscape view: embedding on (x, y), potential U on z.

    Low U is a deep valley, i.e. a robust high-probability attractor. With
    `surface`, a Delaunay surface is drawn under the markers; it needs at least
    four non-collinear attractors and is skipped with a message otherwise.
    """
    z = np.asarray(height, dtype=float)
    ax = plot_embedding(
        embedding,
        labels=labels,
        height=z,
        projection="3d",
        ax=ax,
        figsize=figsize,
        dot_size=dot_size,
        cmap=cmap,
        **plot_kwargs,
    )
    if surface:
        coords = _coords_of(embedding)
        try:
            ax.plot_trisurf(
                coords[:, 0],
                coords[:, 1],
                z,
                cmap=cmap,
                alpha=surface_alpha,
                linewidth=0.0,
            )
        except Exception as exc:  # too few or degenerate points
            print(f"[skip] surface -> {exc}")
    return ax


# === 2. influence graph plots ===============================================


def influence_graph(model: BooleanModel, include_dual: bool = True) -> nx.DiGraph:
    """Directed influence graph of a BooleanModel, with a `sign` on each edge.

    sign = +1 activation, -1 inhibition, 0 dual or unresolved. Set
    `include_dual=False` to drop the edges whose sign could not be resolved.
    """
    A, P = signed_adjacency(model)
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes)
    for i, source in enumerate(model.nodes):
        for j, target in enumerate(model.nodes):
            if P[i, j] == 0:
                continue
            sign = float(A[i, j])
            if sign == 0.0 and not include_dual:
                continue
            G.add_edge(source, target, sign=sign, weight=1.0)
    return G


def layout_positions(
    G: nx.Graph,
    layout: Union[str, Mapping, callable] = "spring",
    seed: int = 0,
    **layout_kwargs,
) -> Dict[str, np.ndarray]:
    """Node positions from a named networkx layout.

    `layout` accepts a name (spring, kamada_kawai, circular, shell, spectral,
    spiral, random), a ready-made {node: (x, y)} mapping, or any callable
    taking the graph. Compute the positions once and reuse them across panels
    so that every figure of the same model is directly comparable.
    """
    if isinstance(layout, Mapping):
        return dict(layout)
    if callable(layout):
        return layout(G, **layout_kwargs)
    if layout not in _LAYOUTS:
        raise ValueError(
            f"unknown layout {layout!r}; choose from {sorted(_LAYOUTS)}"
        )
    if layout in _SEEDED_LAYOUTS:
        layout_kwargs.setdefault("seed", seed)
    return _LAYOUTS[layout](G, **layout_kwargs)


def _edge_widths(G: nx.Graph, edge_width) -> Union[float, list]:
    """Resolve edge_width into something draw_networkx_edges accepts.

    Scalar -> constant width. Mapping keyed by (u, v) -> per-edge width.
    String -> scale widths by that edge attribute. Sequence -> used as given.
    """
    if isinstance(edge_width, Mapping):
        return [float(edge_width.get((u, v), 1.0)) for u, v in G.edges()]
    if isinstance(edge_width, str):
        values = np.array(
            [float(data.get(edge_width, 1.0)) for _, _, data in G.edges(data=True)]
        )
        largest = float(values.max()) if values.size else 1.0
        return list(0.5 + 2.5 * values / (largest or 1.0))
    if np.ndim(edge_width) == 0:
        return float(edge_width)
    return list(np.asarray(edge_width, dtype=float))


def _sign_legend(ax, colours: Mapping[float, str], font_size: int) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=colours[sign], lw=2, label=_SIGN_LABELS[sign])
        for sign in (1.0, -1.0, 0.0)
        if sign in colours
    ]
    ax.legend(handles=handles, fontsize=font_size, frameon=False, loc="best")


def plot_influence_graph(
    model: Union[BooleanModel, nx.Graph],
    positions: Optional[Mapping] = None,
    layout: Union[str, Mapping, callable] = "spring",
    layout_kwargs: Optional[dict] = None,
    seed: int = 0,
    ax: Optional[plt.Axes] = None,
    figsize: Sequence[float] = (7.0, 7.0),
    node_size: Union[float, Sequence[float]] = 600.0,
    node_color: Union[str, Sequence] = "#f0f0f0",
    node_edgecolor: str = "#404040",
    node_linewidth: float = 1.0,
    edge_width: Union[float, str, Mapping, Sequence] = 1.2,
    edge_alpha: float = 0.85,
    arrowsize: int = 12,
    connectionstyle: str = "arc3,rad=0.08",
    sign_colours: Optional[Mapping[float, str]] = None,
    with_labels: bool = True,
    font_size: int = 9,
    sign_legend: bool = True,
    title: Optional[str] = None,
    **node_kwargs,
) -> plt.Axes:
    """Draw the influence graph with networkx.

    Accepts a BooleanModel or an already built graph. Edge colour encodes the
    sign (green activation, red inhibition, grey dual). Every visual parameter
    is exposed: `node_size` and `edge_width` take a scalar, a per-element
    sequence, or (for edges) an attribute name to scale by; `layout` takes a
    name, a callable, or a precomputed position mapping.

    Extra keyword arguments go to `draw_networkx_nodes`.
    """
    G = model if isinstance(model, nx.Graph) else influence_graph(model)
    if positions is None:
        positions = layout_positions(
            G, layout=layout, seed=seed, **(layout_kwargs or {})
        )
    ax = _new_axes(ax, figsize)

    colours = dict(_SIGN_COLOURS)
    colours.update(sign_colours or {})
    edge_colours = [
        colours.get(float(data.get("sign", 0.0)), colours[0.0])
        for _, _, data in G.edges(data=True)
    ]

    nx.draw_networkx_nodes(
        G,
        positions,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edgecolor,
        linewidths=node_linewidth,
        **node_kwargs,
    )
    nx.draw_networkx_edges(
        G,
        positions,
        ax=ax,
        width=_edge_widths(G, edge_width),
        edge_color=edge_colours,
        alpha=edge_alpha,
        arrows=True,
        arrowsize=arrowsize,
        connectionstyle=connectionstyle,
        node_size=node_size,
    )
    if with_labels:
        nx.draw_networkx_labels(G, positions, ax=ax, font_size=font_size)
    if sign_legend:
        _sign_legend(ax, colours, font_size)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=font_size + 3)
    return ax


# === 3. attractor states on the network =====================================


def _stack_states(attractors: ActivityInput) -> pd.DataFrame:
    """Normalise any accepted attractor input into a (n_states, n_nodes) frame."""
    if isinstance(attractors, pd.DataFrame):
        frame = attractors.copy()
    elif isinstance(attractors, pd.Series):
        frame = attractors.to_frame().T
    elif isinstance(attractors, Mapping):
        frame = pd.Series(dict(attractors), dtype=float).to_frame().T
    elif isinstance(attractors, np.ndarray):
        frame = pd.DataFrame(np.atleast_2d(np.asarray(attractors, dtype=float)))
    elif isinstance(attractors, (list, tuple)):
        if not len(attractors):
            raise ValueError("no attractor states given")
        if all(np.isscalar(item) for item in attractors):
            frame = pd.DataFrame([np.asarray(attractors, dtype=float)])
        else:
            frame = pd.concat(
                [_stack_states(item) for item in attractors], ignore_index=True
            )
    else:
        raise TypeError(f"cannot read attractor states from {type(attractors)!r}")
    frame.columns = [str(column) for column in frame.columns]
    return frame


def average_activity(
    attractors: ActivityInput,
    nodes: Optional[Sequence[str]] = None,
    model: Optional[BooleanModel] = None,
    fill_missing: float = 0.0,
    weights: Optional[Sequence[float]] = None,
    aggregate: str = "mean",
) -> pd.Series:
    """Mean node-activity vector over one or several attractor states.

    A single attractor (Series, dict, 1D array, single-row frame) passes
    through unchanged. Several attractors (DataFrame, 2D array, or a list
    mixing those) are averaged node by node into one activity vector in [0, 1],
    which is also the right summary for a cyclic attractor or a phenotype
    group. Node order follows `nodes` or `model.nodes` when either is given.

    `aggregate` chooses how several states are summarised: "mean" (default),
    "median", or "fraction_active", the share of states holding the node on.
    With `weights`, "mean" becomes a weighted mean, so passing MaBoSS attractor
    probabilities gives the model's stationary average node activity rather than
    an unweighted average over attractors. How the vector was obtained is
    recorded in `series.attrs`.
    """
    frame = _stack_states(attractors)
    if nodes is None and model is not None:
        nodes = list(model.nodes)
    if nodes is not None:
        names = [str(node) for node in nodes]
        positional = all(column.isdigit() for column in frame.columns)
        if positional and frame.shape[1] == len(names):
            frame.columns = names
        unknown = sorted(set(frame.columns) - set(names))
        if unknown:
            raise ValueError(f"columns not present in the node list: {unknown}")
        frame = frame.reindex(columns=names).fillna(fill_missing)
    values = frame.astype(float)
    if aggregate == "mean":
        series = (
            values.mean(axis=0)
            if weights is None
            else _weighted_mean(values, weights)
        )
    elif aggregate == "median":
        if weights is not None:
            raise ValueError("`weights` is only supported with aggregate='mean'")
        series = values.median(axis=0)
    elif aggregate == "fraction_active":
        if weights is None:
            series = (values > 0.5).astype(float).mean(axis=0)
        else:
            series = _weighted_mean((values > 0.5).astype(float), weights)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    series.name = "activity"
    series.attrs["n_states"] = int(values.shape[0])
    series.attrs["aggregate"] = aggregate
    series.attrs["weighted"] = weights is not None
    return series


def plot_attractor_on_network(
    attractors: ActivityInput,
    model: Union[BooleanModel, nx.Graph],
    nodes: Optional[Sequence[str]] = None,
    aggregate: str = "mean",
    weights: Optional[Sequence[float]] = None,
    positions: Optional[Mapping] = None,
    layout: Union[str, Mapping, callable] = "spring",
    layout_kwargs: Optional[dict] = None,
    seed: int = 0,
    ax: Optional[plt.Axes] = None,
    figsize: Sequence[float] = (7.0, 7.0),
    node_size: float = 600.0,
    size_by_activity: bool = False,
    node_size_range: Sequence[float] = (150.0, 900.0),
    cmap: str = "coolwarm",
    vmin: float = 0.0,
    vmax: float = 1.0,
    node_edgecolor: str = "#404040",
    node_linewidth: float = 1.0,
    edge_width: Union[float, str, Mapping, Sequence] = 1.0,
    edge_color: str = "#bdbdbd",
    edge_alpha: float = 0.6,
    colour_edges_by_sign: bool = False,
    sign_colours: Optional[Mapping[float, str]] = None,
    arrowsize: int = 10,
    connectionstyle: str = "arc3,rad=0.08",
    with_labels: bool = True,
    font_size: int = 9,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    title: Optional[str] = None,
    **node_kwargs,
) -> plt.Axes:
    """Colour the influence graph by attractor activity.

    `attractors` is either a single attractor state, drawn directly, or several
    states, which are aggregated into one activity vector first (see
    `average_activity`). `aggregate` selects that summary ("mean", "median",
    "fraction_active") and `weights` turns the mean into a weighted one, so
    passing MaBoSS attractor probabilities maps the model's stationary average
    node activity onto the influence graph. A cluster profile from
    `clustering.cluster_activity` can be passed directly as a single state.
    Node colour is the activity on `cmap` between `vmin` and `vmax`; with
    `size_by_activity`, node area also scales with it.

    Reuse `positions` across calls to keep several attractor figures
    comparable. Extra keyword arguments go to `draw_networkx_nodes`.
    """
    G = model if isinstance(model, nx.Graph) else influence_graph(model)
    if nodes is None:
        nodes = list(G.nodes()) if isinstance(model, nx.Graph) else list(model.nodes)
    activity = average_activity(
        attractors, nodes=nodes, weights=weights, aggregate=aggregate
    )
    n_states = int(activity.attrs.get("n_states", 1))
    weighted = bool(activity.attrs.get("weighted", False))
    order = [str(node) for node in G.nodes()]
    values = activity.reindex(order).to_numpy(dtype=float)

    if positions is None:
        positions = layout_positions(
            G, layout=layout, seed=seed, **(layout_kwargs or {})
        )
    ax = _new_axes(ax, figsize)

    if size_by_activity:
        span = float(vmax - vmin) or 1.0
        scaled = np.clip((values - vmin) / span, 0.0, 1.0)
        low, high = float(node_size_range[0]), float(node_size_range[1])
        sizes = low + (high - low) * scaled
    else:
        sizes = node_size

    if colour_edges_by_sign:
        colours = dict(_SIGN_COLOURS)
        colours.update(sign_colours or {})
        edge_colours = [
            colours.get(float(data.get("sign", 0.0)), colours[0.0])
            for _, _, data in G.edges(data=True)
        ]
    else:
        edge_colours = edge_color

    nx.draw_networkx_edges(
        G,
        positions,
        ax=ax,
        width=_edge_widths(G, edge_width),
        edge_color=edge_colours,
        alpha=edge_alpha,
        arrows=True,
        arrowsize=arrowsize,
        connectionstyle=connectionstyle,
        node_size=sizes,
    )
    collection = nx.draw_networkx_nodes(
        G,
        positions,
        ax=ax,
        node_size=sizes,
        node_color=values,
        cmap=plt.get_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        edgecolors=node_edgecolor,
        linewidths=node_linewidth,
        **node_kwargs,
    )
    if with_labels:
        nx.draw_networkx_labels(G, positions, ax=ax, font_size=font_size)
    if colorbar:
        if n_states == 1 and not weighted:
            default_label = "activity"
        elif aggregate == "fraction_active":
            default_label = "fraction active"
        elif weighted:
            default_label = "weighted mean activity"
        else:
            default_label = f"{aggregate} activity"
        ax.figure.colorbar(
            collection, ax=ax, shrink=0.75, label=colorbar_label or default_label
        )
    ax.set_axis_off()
    if title is None:
        if n_states == 1:
            title = "single attractor"
        else:
            summary = "weighted mean" if weighted else aggregate.replace("_", " ")
            title = f"{summary} activity over {n_states} attractors"
    ax.set_title(title, fontsize=font_size + 3)
    return ax


def plot_attractor_panels(
    attractor_sets: Mapping[str, ActivityInput],
    model: Union[BooleanModel, nx.Graph],
    ncols: int = 3,
    panel_size: Sequence[float] = (4.5, 4.5),
    positions: Optional[Mapping] = None,
    layout: Union[str, Mapping, callable] = "spring",
    layout_kwargs: Optional[dict] = None,
    seed: int = 0,
    **plot_kwargs,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """One network panel per attractor or per group of attractors.

    Values of `attractor_sets` follow the same rules as
    `plot_attractor_on_network`, so a panel can hold a single state or the mean
    of a group. A single layout is computed once and shared by every panel.
    """
    names = list(attractor_sets)
    if not names:
        raise ValueError("no attractor sets to plot")
    G = model if isinstance(model, nx.Graph) else influence_graph(model)
    if positions is None:
        positions = layout_positions(
            G, layout=layout, seed=seed, **(layout_kwargs or {})
        )
    ncols = int(max(1, min(ncols, len(names))))
    nrows = int(np.ceil(len(names) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    flat = axes.ravel()
    for index, name in enumerate(names):
        plot_attractor_on_network(
            attractor_sets[name],
            G,
            positions=positions,
            ax=flat[index],
            title=str(name),
            **plot_kwargs,
        )
    for spare in flat[len(names) :]:
        spare.set_axis_off()
    figure.tight_layout()
    return figure, {name: flat[i] for i, name in enumerate(names)}