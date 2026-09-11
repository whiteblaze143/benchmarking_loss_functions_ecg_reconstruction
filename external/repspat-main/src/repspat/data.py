from scipy.spatial.distance import pdist, squareform

def _as_array(matrix):
    """Return a dense array from dense, sparse, or matrix-like AnnData storage."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "A"):
        return matrix.A
    return matrix

def _feature_array(adata):
    layer = adata.uns.get("repspat", {}).get("layer")
    if layer is None:
        raise KeyError(
            "No RepSpat feature layer found. Run compute_distances with a layer first."
        )
    return _as_array(adata.layers[layer])

def compute_distances(
    adata,
    *,
    layer,
    metric,
    distance_key="repspat_distances",
):
    """Compute pairwise feature distances from a required AnnData layer."""
    if layer is None:
        raise ValueError("layer must be provided. Store features in adata.layers first.")
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in AnnData layers.")

    X = _as_array(adata.layers[layer])
    dist_values = squareform(pdist(X, metric=metric))
    adata.obsp[distance_key] = dist_values
    adata.uns.setdefault("repspat", {}).update({
        "layer": layer,
        "metric": metric,
        "distance_key": distance_key,
    })
    return adata
