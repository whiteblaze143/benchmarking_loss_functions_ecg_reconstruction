# repSpat

A Python package for detecting repeated spatial patterns in spatial omics data, tissue regions that are spatially separated but share similar underlying cell-type or biological profile distributions. Built on a nonparametric statistical inference framework that integrates constrained clustering with a block-permutation procedure using the maximum mean discrepancy (MMD) statistic, enabling formal hypothesis testing for repeated spatial domains. Works natively with AnnData .h5ad files, the standard format for single-cell and spatial omics workflows.

## Install

```bash
pip install repspat
```

## Usage

```python
import scanpy as sc
from repspat import compute_distances, create_blocks, multiple_comparison
from repspat import spatial_silhouette_analysis, spatial_constrained_hac
from repspat import (
    pairwise_results_to_matrix,
    plot_cluster_feature_presence,
    plot_spatial_clusters,
)
```

### Load a sample

```python
adata = sc.read_h5ad("03_TNBC_2018_spe.h5ad")
adata = adata[adata.obs["sample_id"] == "Sample_04"].copy()

adata = compute_distances(
    adata,
    layer="binary",
    metric="jaccard",
)

{
    "feature_layer": adata.layers["binary"].shape,
    "coords_mat": adata.obsm["spatial"].shape,
    "dist_matrix": adata.obsp["repspat_distances"].shape,
    "adata": adata.shape,
}
# {
#   'feature_layer': (5381, 36),
#   'coords_mat': (5381, 2),
#   'dist_matrix': (5381, 5381),
#   'adata': (5381, 36)
# }
```

If you want to use continuous features, store them in a layer and pass that
layer explicitly:

```python
adata.layers["continuous"] = adata.X.copy()
adata = compute_distances(
    adata,
    layer="continuous",
    metric="euclidean",
)
```

### Find the right number of clusters

Runs spatially-aware silhouette analysis over a range of k and neighbor configs:

```python
adata = spatial_silhouette_analysis(
    adata,
    n_neighbors_list=[6, 8],
    n_clusters_range=range(4, 9),
)
adata.uns["silhouette_scores"]
#    n_neighbors  n_clusters  avg_silhouette
# 0            6           4        0.136721
# 5            8           4        0.161823
```

### Cluster cells

```python
adata = spatial_constrained_hac(
    adata,
    n_clusters=7,
    n_neighs=8,
    linkage="ward",
)
# labels are stored in adata.obs["labels"]
```

### Plot

```python
plot_spatial_clusters(adata)
```

### Plot each cluster's top features

Creates one two-panel figure per cluster: a spatial highlight and the cluster's
top features. Binary features are shown as presence rates; continuous numeric
features are shown as cluster-vs-rest standardized enrichment scores.

```python
cluster_figures = plot_cluster_feature_presence(adata, top_n=10)
```

### Run MMD tests between clusters

```python
adata = create_blocks(adata, knn=8)
adata = multiple_comparison(adata, kernel="IMQ")

# pairs that are not significantly different
results = adata.uns["repspat_mmd_results"]
print(results[results["adj_p"] >= 0.05])
```

### Similarity network

```python
matrix = pairwise_results_to_matrix(adata)
```

See `example.ipynb` for a full walkthrough on a TNBC dataset.

## API

### `compute_distances(adata, *, layer, metric, distance_key="repspat_distances")`
Computes pairwise distances from the required AnnData layer, stores them in `adata.obsp[distance_key]`, records the active feature layer in `adata.uns["repspat"]["layer"]`, and returns the same `adata`.

### `spatial_silhouette_analysis(adata, n_neighbors_list, n_clusters_range, linkage="ward")`
Stores a DataFrame of silhouette scores across all `(n_neighbors, n_clusters)` combinations in `adata.uns["silhouette_scores"]` and returns the same `adata`. Use this to pick clustering params. `linkage` can be `"ward"`, `"single"`, `"complete"`, or `"average"`.

### `spatial_constrained_hac(adata, n_clusters, n_neighs, coord_type, delaunay, linkage="ward")`
HAC with a spatial connectivity constraint. Stores labels in `adata.obs["labels"]`, prints the storage location, and returns the same `adata`. Clustering uses the active feature layer recorded by `compute_distances`; `metric` controls the stored distance matrix used by silhouette and downstream tests. Set `linkage` to `"single"`, `"complete"`, or `"average"` when needed.

### `create_blocks(adata, knn)`
Splits regions into KMeans blocks for block permutation testing, stores block IDs in `adata.obs["repspat_block_id"]`, prints that location, and returns the same `adata`.

### `two_sample_mmd(sample1_idx, sample2_idx, dist_matrix, patient_data, kernel, kernel_param, nperm)`
MMD² between two groups with a permutation null. Returns observed statistic, null distribution, and p-value.

### `multiple_comparison(adata, kernel, kernel_param, nperm, adj_p)`
Pairwise MMD across all cluster pairs. Reads regions, block IDs, and distances from AnnData. Stores results in `adata.uns["repspat_mmd_results"]`, prints that location, and returns the same `adata`. `adj_p` can be `"BH"`, `"bonferroni"`, or `"holm"`.

### `plot_spatial_clusters(adata)`
Scatter plot colored by `adata.obs["labels"]`.

### `plot_cluster_feature_presence(adata, top_n)`
Creates a spatial highlight and top feature chart for every cluster. Uses binary-feature prevalence when binary features are available, otherwise falls back to mean z-score difference (cluster − rest) for numeric features.

### `pairwise_results_to_matrix(adata, plot, alpha=0.05)`
Builds an adjacency matrix and network graph from `adata.uns["repspat_mmd_results"]`. Edges connect clusters whose adjusted p-value is at least `alpha` (not significantly different).

## Requirements

- Python >= 3.9
- numpy, pandas, scipy, scikit-learn, statsmodels, scanpy, squidpy >= 1.2, matplotlib, networkx

## License

MIT
