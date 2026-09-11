"""Label-free functional-atlas audit of frozen primary M3 scientific objects."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_adjacency(registry: pd.DataFrame) -> set[tuple[int, int]]:
    locations = {
        (int(row.ix), int(row.iy), int(row.iz)): int(row.domain_id)
        for row in registry.itertuples(index=False)
    }
    edges: set[tuple[int, int]] = set()
    offsets = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)
    ]
    for (x, y, z), domain in locations.items():
        for dx, dy, dz in offsets:
            neighbor = locations.get((x + dx, y + dy, z + dz))
            if neighbor is not None and neighbor != domain:
                edges.add(tuple(sorted((domain, neighbor))))
    return edges


def classical_mds(squared_distances: np.ndarray, dimensions: int = 3):
    n = len(squared_distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared_distances @ centering
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    positive = np.maximum(eigenvalues[:dimensions], 0.0)
    coordinates = eigenvectors[:, :dimensions] * np.sqrt(positive)
    return coordinates, eigenvalues


def main() -> None:
    parser = argparse.ArgumentParser(description="M3I label-free functional atlas audit")
    parser.add_argument("--frozen-dir", default="refine-logs/qvcg/m3_adapted_repspat_primary_frozen")
    parser.add_argument("--domain-stats", default="refine-logs/qvcg/VCG_DOMAIN_STATS.parquet")
    parser.add_argument("--domain-registry", default="refine-logs/qvcg/VCG_SPATIAL_DOMAIN_REGISTRY.parquet")
    parser.add_argument("--out-dir", default="refine-logs/qvcg/m3i_functional_atlas")
    args = parser.parse_args()

    frozen = Path(args.frozen_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((frozen / "M3_FREEZE_MANIFEST.json").read_text())
    for name, expected in manifest["sha256"].items():
        observed = sha256(frozen / name)
        if observed != expected:
            raise RuntimeError(f"Frozen M3 checksum mismatch for {name}")

    pairs = pd.read_parquet(frozen / "VCG_REPSPAT_PAIR_TESTS.parquet").copy()
    mmd = np.load(frozen / "VCG_MMD_MATRIX.npy")
    stats = pd.read_parquet(args.domain_stats).sort_values("domain_id")
    registry = pd.read_parquet(args.domain_registry)
    domains = stats.domain_id.to_numpy(dtype=int)
    if len(pairs) != 2016 or mmd.shape != (64, 64) or len(domains) != 64:
        raise RuntimeError("M3I requires exactly 64 domains and 2,016 primary pairs")

    centroids = stats[["centroid_vx", "centroid_vy", "centroid_vz"]].to_numpy(float)
    centroid_by_domain = dict(zip(domains, centroids))
    pairs["physical_distance"] = [
        float(np.linalg.norm(centroid_by_domain[int(i)] - centroid_by_domain[int(j)]))
        for i, j in pairs[["domain_1", "domain_2"]].itertuples(index=False, name=None)
    ]
    adjacency = physical_adjacency(registry)
    pairs["physically_adjacent"] = [
        tuple(sorted((int(i), int(j)))) in adjacency
        for i, j in pairs[["domain_1", "domain_2"]].itertuples(index=False, name=None)
    ]
    distant_threshold = float(pairs.physical_distance.quantile(0.75))
    pairs["physically_distant_q75"] = pairs.physical_distance >= distant_threshold
    pairs.to_parquet(out / "M3I_PAIR_GEOMETRY.parquet", index=False)

    graph = nx.Graph()
    graph.add_nodes_from(domains.tolist())
    edge_rows = pairs[pairs.similarity_edge]
    graph.add_edges_from(edge_rows[["domain_1", "domain_2"]].itertuples(index=False, name=None))
    degrees = dict(graph.degree())
    components = list(nx.connected_components(graph))
    path_lengths = [
        distance for source, lengths in nx.all_pairs_shortest_path_length(graph)
        for target, distance in lengths.items() if source < target
    ]
    maximal_cliques = list(nx.find_cliques(graph))
    clique_sizes = [len(clique) for clique in maximal_cliques]
    degree_table = pd.DataFrame({
        "domain_id": domains,
        "degree": [degrees[int(domain)] for domain in domains],
        "clustering_coefficient": [nx.clustering(graph, int(domain)) for domain in domains],
    })
    degree_table.to_parquet(out / "M3I_DOMAIN_GRAPH_METRICS.parquet", index=False)

    distant_nonreject = pairs[pairs.physically_distant_q75 & pairs.similarity_edge].copy()
    distant_nonreject.sort_values(["physical_distance", "mmd2_imq_obs"], ascending=[False, True]).to_parquet(
        out / "M3I_DISTANT_NONREJECTED_PAIRS.parquet", index=False
    )
    adjacent_reject = pairs[pairs.physically_adjacent & pairs.bh_reject_imq].copy()
    adjacent_reject.sort_values(["physical_distance", "mmd2_imq_obs"], ascending=[True, False]).to_parquet(
        out / "M3I_ADJACENT_REJECTED_PAIRS.parquet", index=False
    )

    correlation = spearmanr(pairs.physical_distance, pairs.mmd2_imq_obs)
    coordinates, eigenvalues = classical_mds(np.maximum(mmd, 0.0), dimensions=3)
    positive_sum = float(eigenvalues[eigenvalues > 0].sum())
    negative_sum = float(-eigenvalues[eigenvalues < 0].sum())
    embedding = pd.DataFrame({
        "domain_id": domains,
        "mds_1": coordinates[:, 0], "mds_2": coordinates[:, 1], "mds_3": coordinates[:, 2],
        "centroid_vx": centroids[:, 0], "centroid_vy": centroids[:, 1], "centroid_vz": centroids[:, 2],
    })
    embedding.to_parquet(out / "M3I_MDS_EMBEDDING.parquet", index=False)
    np.save(out / "M3I_MDS_EIGENVALUES.npy", eigenvalues)

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = np.where(pairs.bh_reject_imq, "#d1495b", "#2878b5")
    axis.scatter(pairs.physical_distance, pairs.mmd2_imq_obs, c=colors, s=16, alpha=0.65, linewidths=0)
    axis.axvline(distant_threshold, color="0.3", linestyle="--", linewidth=1, label="physical-distance Q75")
    axis.set(xlabel="Physical VCG centroid distance", ylabel="Observed IMQ MMD²",
             title="Frozen M3 physical versus functional distance")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "M3I_PHYSICAL_VS_MMD.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    scatter = axes[0].scatter(centroids[:, 0], centroids[:, 1], c=coordinates[:, 0], cmap="viridis", s=38)
    axes[0].set(xlabel="VCG centroid x", ylabel="VCG centroid y", title="Physical domains colored by MDS-1")
    fig.colorbar(scatter, ax=axes[0], label="Functional MDS-1")
    axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=domains, cmap="turbo", s=38)
    axes[1].set(xlabel="MDS-1", ylabel="MDS-2", title="Functional MMD geometry (domain IDs only)")
    fig.tight_layout()
    fig.savefig(out / "M3I_FUNCTIONAL_ATLAS_MDS.png", dpi=200)
    plt.close(fig)

    topology = {
        "num_nodes": graph.number_of_nodes(), "num_edges": graph.number_of_edges(),
        "edge_density": float(nx.density(graph)),
        "degree_min": int(min(degrees.values())), "degree_median": float(np.median(list(degrees.values()))),
        "degree_mean": float(np.mean(list(degrees.values()))), "degree_max": int(max(degrees.values())),
        "num_connected_components": len(components),
        "component_sizes": sorted([len(component) for component in components], reverse=True),
        "average_clustering_coefficient": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "diameter": int(nx.diameter(graph)) if nx.is_connected(graph) else None,
        "average_shortest_path_length": float(nx.average_shortest_path_length(graph)) if nx.is_connected(graph) else None,
        "shortest_path_length_counts": {str(k): int(v) for k, v in sorted(Counter(path_lengths).items())},
        "clique_number": int(max(clique_sizes)),
        "num_maximal_cliques": len(maximal_cliques),
        "maximal_clique_size_counts": {str(k): int(v) for k, v in sorted(Counter(clique_sizes).items())},
    }
    summary = {
        "run_id": "M3I_FUNCTIONAL_ATLAS_AUDIT", "status": "COMPLETE",
        "scientific_object": "frozen M3 pairwise MMD geometry and BH non-rejection graph",
        "categorical_labels_created": False,
        "threshold_definitions": {
            "physical_adjacency": "26-neighbor contact between occupied voxels of distinct domains",
            "physically_distant": "pairwise centroid distance >= empirical Q75",
            "physical_distance_q75": distant_threshold,
        },
        "topology": topology,
        "physical_functional_association": {
            "spearman_rho": float(correlation.statistic), "spearman_p_value": float(correlation.pvalue),
        },
        "pair_counts": {
            "physically_adjacent_pairs": int(pairs.physically_adjacent.sum()),
            "physically_adjacent_rejected_pairs": len(adjacent_reject),
            "physically_distant_pairs": int(pairs.physically_distant_q75.sum()),
            "physically_distant_nonrejected_pairs": len(distant_nonreject),
        },
        "mmd_by_bh_decision": {
            "nonrejected_median": float(pairs.loc[pairs.similarity_edge, "mmd2_imq_obs"].median()),
            "rejected_median": float(pairs.loc[pairs.bh_reject_imq, "mmd2_imq_obs"].median()),
        },
        "mds": {
            "input_distance": "sqrt(max(MMD_squared, 0)); classical MDS receives squared distance matrix MMD_squared",
            "positive_eigenvalue_sum": positive_sum,
            "negative_eigenvalue_magnitude_sum": negative_sum,
            "negative_to_positive_eigenmass": negative_sum / positive_sum if positive_sum else None,
            "top_eigenvalues": eigenvalues[:10].tolist(),
        },
    }
    (out / "M3I_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
