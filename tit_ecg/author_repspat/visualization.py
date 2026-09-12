import pandas as pd
import networkx as nx

import numpy as np
from .data import _feature_array

def _spatial_xy(adata):
    coords = adata.obsm["spatial"]
    if hasattr(coords, "iloc"):
        return coords.iloc[:, 0], coords.iloc[:, 1]
    return coords[:, 0], coords[:, 1]

def plot_spatial_clusters(adata, label_key="labels", figsize=(4, 4),
                          point_size=10, alpha=1.0,
                          title="Spatial Plot of Cells with Cluster Colors",
                          show_legend=True):
    import matplotlib.pyplot as plt

    if label_key not in adata.obs:
        raise KeyError(f"'{label_key}' not found in adata.obs.")
    x, y = _spatial_xy(adata)
    labels = adata.obs[label_key]
    x, y, labels = np.asarray(x), np.asarray(y), np.asarray(labels)
    fig, ax = plt.subplots(figsize=figsize)

    for cid in np.unique(labels):
        m = labels == cid
        ax.scatter(x[m], y[m], s=point_size, alpha=alpha, label=f"Cluster {cid}")

    ax.set(xlabel="X coordinate", ylabel="Y coordinate", title=title)
    ax.grid(False)

    if show_legend:
        ax.legend(title="Clusters", markerscale=1.5,
                  bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    return fig, ax


def _is_binary_feature(column):
    values = column.dropna()
    return (
        not values.empty
        and (
            pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
        )
        and set(pd.unique(values)).issubset({0, 1})
    )


def _resolve_binary_features(feature_df, feature_columns):
    if feature_columns is None:
        columns = [
            column for column in feature_df
            if _is_binary_feature(feature_df[column])
        ]
    else:
        columns = list(feature_columns)
        missing = [column for column in columns if column not in feature_df]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        non_binary = [
            column for column in columns
            if not _is_binary_feature(feature_df[column])
        ]
        if non_binary:
            raise ValueError(f"Feature columns must be binary: {non_binary}")
    if not columns:
        raise ValueError("No binary feature columns found to plot.")
    return columns


def _resolve_numeric_features(feature_df, feature_columns):
    if feature_columns is None:
        columns = [
            column for column in feature_df
            if pd.api.types.is_numeric_dtype(feature_df[column])
        ]
    else:
        columns = list(feature_columns)
        missing = [column for column in columns if column not in feature_df]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        non_numeric = [
            column for column in columns
            if not pd.api.types.is_numeric_dtype(feature_df[column])
        ]
        if non_numeric:
            raise ValueError(f"Feature columns must be numeric: {non_numeric}")
    if not columns:
        raise ValueError("No numeric feature columns found to plot.")
    return columns


def _standardize_features(feature_df, feature_columns):
    selected = feature_df[feature_columns]
    scale = selected.std(ddof=0).replace(0, np.nan)
    return ((selected - selected.mean()) / scale).fillna(0.0)


def _plot_cluster_feature_figure(
    x, y, labels, feature_df, feature_columns, cluster_id,
    top_n, figsize, point_size, alpha,
):
    import matplotlib.pyplot as plt

    mask = labels == cluster_id
    cluster_size = int(mask.sum())
    cluster_features = feature_df.iloc[mask][feature_columns]
    presence = (
        cluster_features.mean()
        .sort_values(ascending=False, kind="stable")
        .iloc[:top_n]
    )
    counts = cluster_features[presence.index].sum().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    spatial_ax, feature_ax = axes

    spatial_ax.scatter(
        x[~mask], y[~mask], s=point_size, alpha=min(alpha, 0.35),
        color="lightgrey",
    )
    spatial_ax.scatter(
        x[mask], y[mask], s=point_size, alpha=alpha,
        color="tab:red",
    )
    spatial_ax.set(
        xlabel="X coordinate", ylabel="Y coordinate",
        title=f"Cluster {cluster_id} (n={cluster_size})",
    )
    spatial_ax.grid(False)

    plot_presence = presence.iloc[::-1]
    bars = feature_ax.barh(
        plot_presence.index, plot_presence.values * 100, color="tab:blue"
    )
    feature_ax.bar_label(
        bars,
        labels=[
            f"{counts[feature]}/{cluster_size} ({rate:.1%})"
            for feature, rate in plot_presence.items()
        ],
        padding=3,
    )
    feature_ax.set(
        xlabel="Cells with feature present (%)",
        title=f"Top {len(presence)} binary features",
        xlim=(0, 115),
    )
    feature_ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, axes


def _plot_cluster_feature_enrichment_figure(
    x, y, labels, feature_df, feature_columns, cluster_id,
    top_n, figsize, point_size, alpha,
):
    import matplotlib.pyplot as plt

    mask = labels == cluster_id
    cluster_size = int(mask.sum())
    standardized = _standardize_features(feature_df, feature_columns)
    cluster_mean = standardized.iloc[mask].mean()
    if (~mask).any():
        rest_mean = standardized.iloc[~mask].mean()
        scores = cluster_mean - rest_mean
    else:
        scores = cluster_mean
    scores = scores.sort_values(ascending=False, kind="stable").iloc[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    spatial_ax, feature_ax = axes

    spatial_ax.scatter(
        x[~mask], y[~mask], s=point_size, alpha=min(alpha, 0.35),
        color="lightgrey",
    )
    spatial_ax.scatter(
        x[mask], y[mask], s=point_size, alpha=alpha,
        color="tab:red",
    )
    spatial_ax.set(
        xlabel="X coordinate", ylabel="Y coordinate",
        title=f"Cluster {cluster_id} (n={cluster_size})",
    )
    spatial_ax.grid(False)

    plot_scores = scores.iloc[::-1]
    bars = feature_ax.barh(
        plot_scores.index, plot_scores.values, color="tab:green"
    )
    feature_ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    for bar, value in zip(bars, plot_scores.values):
        label_x = value if value >= 0 else 0
        feature_ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f" {value:.3g}", va="center", ha="left", fontsize=9,
        )
    x_min = min(plot_scores.min(), 0)
    x_max = max(plot_scores.max(), 0)
    feature_ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.2 + 0.05)
    feature_ax.set(
        xlabel="Mean z-score difference (cluster − rest)",
        title=f"Top {len(scores)} cluster-defining features",
    )
    feature_ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, axes


def plot_cluster_feature_presence(
    adata,
    top_n=5,
    feature_columns=None,
    figsize=(8, 3),
    point_size=2,
    alpha=1.0,
    label_key="labels",
):
    """Plot each cluster beside binary presence or continuous enrichment.

    Binary features are plotted as presence rates. If no binary features are
    found, or if any selected feature is continuous, numeric features are
    plotted by cluster-vs-rest standardized enrichment.
    Returns cluster IDs mapped to ``(figure, axes)`` tuples.
    """
    if label_key not in adata.obs:
        raise KeyError(f"'{label_key}' not found in adata.obs.")

    x_values, y_values = _spatial_xy(adata)
    x, y = np.asarray(x_values).reshape(-1), np.asarray(y_values).reshape(-1)
    labels = np.asarray(adata.obs[label_key]).reshape(-1)
    feature_df = pd.DataFrame(
        _feature_array(adata),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    if not (len(x) == len(y) == len(labels) == len(feature_df)):
        raise ValueError(
            "x, y, labels, and feature_df must contain the same number of rows."
        )
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0
    ):
        raise ValueError("top_n must be a positive integer or None.")

    if feature_columns is None:
        binary_columns = [
            column for column in feature_df
            if _is_binary_feature(feature_df[column])
        ]
        if binary_columns:
            feature_columns = binary_columns
            plotter = _plot_cluster_feature_figure
        else:
            feature_columns = _resolve_numeric_features(feature_df, feature_columns)
            plotter = _plot_cluster_feature_enrichment_figure
    else:
        feature_columns = list(feature_columns)
        missing = [column for column in feature_columns if column not in feature_df]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        if all(_is_binary_feature(feature_df[column]) for column in feature_columns):
            plotter = _plot_cluster_feature_figure
        else:
            feature_columns = _resolve_numeric_features(feature_df, feature_columns)
            plotter = _plot_cluster_feature_enrichment_figure

    cluster_ids = pd.unique(labels[~pd.isna(labels)])
    if len(cluster_ids) == 0:
        raise ValueError("labels must contain at least one cluster.")

    return {
        cluster_id: plotter(
            x, y, labels, feature_df, feature_columns, cluster_id,
            top_n, figsize, point_size, alpha,
        )
        for cluster_id in cluster_ids
    }


def pairwise_results_to_matrix(adata, plot=True, alpha=0.05):
    if hasattr(adata, "uns"):
        if "repspat_mmd_results" not in adata.uns:
            raise KeyError("'repspat_mmd_results' not found in adata.uns.")
        df = adata.uns["repspat_mmd_results"]
    else:
        df = adata

    # Link clusters that are NOT significantly different (adj_p >= alpha = similar spatial distributions)
    df = df.copy()
    df['link'] = (df['adj_p'] >= alpha).astype(int)
    
    # Nodes and edges
    all_nodes = pd.unique(df[['region_1','region_2']].values.ravel())
    edges = df[df['link'] == 1][['region_1','region_2','obs_mmd_sq']]
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    for _, row in edges.iterrows():
        G.add_edge(row['region_1'], row['region_2'], weight=row['obs_mmd_sq'])
    
    # Adjacency matrix
    result_matrix = nx.to_pandas_adjacency(G, weight='weight')
    
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        pos = nx.circular_layout(G)  # nodes in circle
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='white', edge_color='lightgrey',
                width=2, node_size=800, font_size=12, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: round(v,2) for k,v in edge_labels.items()})
        plt.show()

    return result_matrix
