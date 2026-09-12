import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering
from .data import _as_array, _feature_array

_ALLOWED_HAC_LINKAGES = {"ward", "single", "complete", "average"}


def _spatial_neighbors(*args, **kwargs):
    import squidpy as sq

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Calling `spatial_neighbors` is deprecated.*",
            category=FutureWarning,
        )
        return sq.gr.spatial_neighbors(*args, **kwargs)


def _validate_hac_linkage(linkage):
    normalized = str(linkage).lower()
    if normalized not in _ALLOWED_HAC_LINKAGES:
        allowed = sorted(_ALLOWED_HAC_LINKAGES)
        raise ValueError(f"linkage must be one of: {allowed}")
    return normalized


def _hac_model_and_input(
    adata,
    n_clusters,
    connectivity,
    linkage="ward",
    compute_distances=False,
):
    linkage = _validate_hac_linkage(linkage)
    X = _feature_array(adata)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage=linkage,
        connectivity=connectivity,
        compute_distances=compute_distances,
    )

    return model, X, linkage


def _caller_adata_name(adata):
    import inspect

    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    if caller is None:
        return "adata"

    for name, value in caller.f_locals.items():
        if value is adata:
            return name
    return "adata"


def spatial_silhouette_analysis(
    adata,
    n_neighbors_list=[6,8],
    n_clusters_range=range(4,9),
    linkage="ward",
):
    results = []

    # Distance matrix of features
    dist_matrix = _as_array(adata.obsp["repspat_distances"])
    for knn in n_neighbors_list:
        # Construct spatial neighbors graph
        _spatial_neighbors(
            adata,
            n_neighs=knn,
            coord_type="generic",
            delaunay=False
        )

        connectivity_sparse = adata.obsp["spatial_connectivities"].tocsr()
        adjacency = connectivity_sparse.toarray().astype(bool)

        for n_clusters in n_clusters_range:
            clustering, clustering_input, _ = _hac_model_and_input(
                adata,
                n_clusters=n_clusters,
                connectivity=connectivity_sparse,
                linkage=linkage,
            )

            cluster_labels = clustering.fit_predict(clustering_input)
            labels = np.asarray(cluster_labels)
            silhouettes = np.zeros(labels.shape[0], dtype=float)
            label_indices = {
                label: np.flatnonzero(labels == label)
                for label in np.unique(labels)
            }
            neighbor_labels = {}
            for label, indices in label_indices.items():
                connected = adjacency[indices].any(axis=0)
                neighbor_labels[label] = [
                    other
                    for other in np.unique(labels[connected])
                    if other != label
                ]

            for label, members in label_indices.items():
                if len(members) > 1:
                    intra = dist_matrix[np.ix_(members, members)]
                    a = (intra.sum(axis=1) - np.diag(intra)) / (len(members) - 1)
                else:
                    a = np.zeros(len(members), dtype=float)

                neighbors = neighbor_labels.get(label, [])
                if not neighbors:
                    continue

                b_candidates = [
                    dist_matrix[np.ix_(members, label_indices[neighbor])].mean(axis=1)
                    for neighbor in neighbors
                ]
                b = np.minimum.reduce(b_candidates)
                denom = np.maximum(a, b)
                silhouettes[members] = np.divide(
                    b - a,
                    denom,
                    out=np.zeros_like(denom, dtype=float),
                    where=denom != 0,
                )

            sil_scores = silhouettes
            avg_sil = np.mean(sil_scores)

            results.append({
                "n_neighbors": knn,
                "n_clusters": n_clusters,
                "avg_silhouette": avg_sil
            })

    results_df = pd.DataFrame(results)
    adata.uns["silhouette_scores"] = results_df
    print(
        "Silhouette scores are stored in "
        f'{_caller_adata_name(adata)}.uns["silhouette_scores"]'
    )
    return adata


def create_blocks(adata, knn: int, region_key: str = "labels",
                  block_key: str = "repspat_block_id"):
    """Create KMeans blocks within each region."""
    if region_key not in adata.obs:
        raise KeyError(f"'{region_key}' not found in adata.obs.")

    feature_df = pd.DataFrame(
        _feature_array(adata),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    block_ids = pd.Series(index=adata.obs_names, dtype="Int64")

    for region in adata.obs[region_key].unique():
        obs_names = adata.obs_names[adata.obs[region_key] == region]
        df = feature_df.loc[obs_names]
        num_blks = len(df) // knn
        num_blks = 1 if num_blks == 0 or num_blks >= len(df.drop_duplicates()) else num_blks
        if num_blks == 1:
            block_ids.loc[obs_names] = 1
        else:
            block_ids.loc[obs_names] = (
                KMeans(n_clusters=num_blks, n_init=10, random_state=0).fit(df).labels_ + 1
            )

    adata.obs[block_key] = block_ids
    adata.uns.setdefault("repspat", {})["block_key"] = block_key
    print(
        "Block IDs are stored in "
        f'{_caller_adata_name(adata)}.obs["{block_key}"]'
    )
    return adata


def spatial_constrained_hac(adata, n_clusters: int = 7, n_neighs: int = 8,
                            coord_type: str = "generic", delaunay: bool = False,
                            region_key: str = "labels",
                            linkage: str = "ward",
):
    _spatial_neighbors(
        adata,
        n_neighs=n_neighs,
        coord_type=coord_type,
        delaunay=delaunay,
    )

    connectivity = adata.obsp["spatial_connectivities"].tocsr()

    repspat_settings = adata.uns.get("repspat", {})
    model, X, resolved_linkage = _hac_model_and_input(
        adata,
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage=linkage,
        compute_distances=True,
    )
    labels = model.fit_predict(X) + 1
    adata.obs[region_key] = pd.Series(labels, index=adata.obs_names).astype("category")
    adata.uns.setdefault("repspat", {})["region_key"] = region_key
    adata.uns["repspat_hac"] = {
        "n_clusters": n_clusters,
        "n_neighs": n_neighs,
        "coord_type": coord_type,
        "delaunay": delaunay,
        "linkage": resolved_linkage,
        "layer": repspat_settings.get("layer"),
        "metric": repspat_settings.get("metric"),
    }

    adata_name = _caller_adata_name(adata)
    print(
        "HAC labels are stored in "
        f'{adata_name}.obs["{region_key}"]'
    )

    return adata
