"""Full released-repSpat implementation sensitivity for Q-VCG M3.

Outputs are isolated under ``m3r_reference`` and never overwrite primary M3.
The only scientific adaptation retained is neighborhood/block parameter m=10.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile
import threading

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from rep_stat_ecg.scripts.discover_repeated_motifs import assign_domains
from rep_stat_ecg.src.motifs.mmd import (
    benjamini_hochberg,
    biased_mmd2_batch_from_block_sums,
    build_reference_repspat_blocks,
    generate_reference_repspat_assignments,
    mmd2_from_block_sums,
    precompute_block_kernel_sums,
)

PAIR_KEYS = {
    "domain_i", "domain_j", "n_i", "n_j", "n_blocks_i", "n_blocks_j",
    "mmd2_obs", "n_perm", "n_exceed", "p_perm", "seed",
}


def atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        np.savez(tmp, **values)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_pair(path: Path, d1: int, d2: int, n_perm: int, seed: int) -> dict | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            if not PAIR_KEYS.issubset(values.files):
                return None
            row = {key: values[key].item() for key in PAIR_KEYS}
        observed = (int(row["domain_i"]), int(row["domain_j"]),
                    int(row["n_perm"]), int(row["seed"]))
        return row if observed == (d1, d2, n_perm, seed) else None
    except (OSError, ValueError, EOFError):
        return None


def save_block_cache(path: Path, blocks: list[np.ndarray], n_rows: int, m: int) -> None:
    points = np.concatenate(blocks, axis=0)
    sizes = np.asarray([len(block) for block in blocks], dtype=np.int64)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        np.savez(tmp, points=points, sizes=sizes, n_rows=n_rows, m=m,
                 implementation="released_KMeans_n_init10_random_state0_floor")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_block_cache(path: Path, n_rows: int, m: int) -> list[np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            if int(values["n_rows"]) != n_rows or int(values["m"]) != m:
                return None
            if str(values["implementation"]) != "released_KMeans_n_init10_random_state0_floor":
                return None
            points = values["points"]
            sizes = values["sizes"].astype(np.int64)
        if len(points) != n_rows or int(sizes.sum()) != n_rows:
            return None
        ends = np.cumsum(sizes)
        starts = np.r_[0, ends[:-1]]
        return [points[start:end] for start, end in zip(starts, ends)]
    except (OSError, ValueError, KeyError, EOFError):
        return None
def graph_summary(pair_table: pd.DataFrame, domains: list[int]) -> tuple[nx.Graph, dict]:
    graph = nx.Graph()
    graph.add_nodes_from(domains)
    edges = pair_table.loc[pair_table.similarity_edge, ["domain_1", "domain_2"]]
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
    non_cliques = [
        index for index, component in enumerate(components)
        if graph.subgraph(component).number_of_edges() != len(component) * (len(component) - 1) // 2
    ]
    return graph, {
        "num_similarity_edges": graph.number_of_edges(),
        "num_connected_components": len(components),
        "largest_component_size": max(map(len, components)),
        "component_sizes": [len(component) for component in components],
        "non_clique_component_ids": non_cliques,
        "num_non_clique_components": len(non_cliques),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M3R released-repSpat sensitivity")
    parser.add_argument("--micro-table", default="refine-logs/qvcg/VCG_MICROSTATE_TABLE.parquet")
    parser.add_argument("--domain-stats", default="refine-logs/qvcg/VCG_DOMAIN_STATS.parquet")
    parser.add_argument("--domain-registry", default="refine-logs/qvcg/VCG_SPATIAL_DOMAIN_REGISTRY.parquet")
    parser.add_argument("--std-json", default="refine-logs/qvcg/VCG_COORDINATE_STANDARDIZER.json")
    parser.add_argument("--primary-pairs", default="refine-logs/qvcg/VCG_REPSPAT_PAIR_TESTS.parquet")
    parser.add_argument("--out-dir", default="refine-logs/qvcg/m3r_reference")
    parser.add_argument("--neighborhood-size", type=int, default=10)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42000,
                        help="Deterministic execution seed; not a scientific parameter")
    parser.add_argument("--pair-workers", type=int, default=6)
    parser.add_argument("--large-pair-block-threshold", type=int, default=9000,
                        help="Serialize memory-heavy pairs; scheduling only, no statistical effect")
    args = parser.parse_args()

    if args.neighborhood_size != 10 or args.n_perm != 200 or args.alpha != 0.05:
        raise ValueError("M3R contract requires m=10, 200 permutations, and BH alpha=0.05")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "pair_checkpoints"
    stats = pd.read_parquet(args.domain_stats)
    registry = pd.read_parquet(args.domain_registry)
    microstates = assign_domains(pd.read_parquet(args.micro_table), registry, args.std_json)
    domains = sorted(map(int, stats.domain_id.unique()))
    expected = list(combinations(domains, 2))

    complete, missing = [], []
    for index, (d1, d2) in enumerate(expected):
        pair_seed = args.seed + index
        path = checkpoints / f"pair_{d1:03d}_{d2:03d}.npz"
        cached = load_pair(path, d1, d2, args.n_perm, pair_seed)
        (complete if cached is not None else missing).append(
            cached if cached is not None else (index, d1, d2, path)
        )
    print(f"Validated {len(complete)}/{len(expected)} M3R checkpoints; {len(missing)} missing", flush=True)

    active = sorted({domain for _, d1, d2, _ in missing for domain in (d1, d2)})
    blocks: dict[int, list[np.ndarray]] = {}
    block_cache_dir = out / "block_cache"
    block_cache_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        for domain in tqdm(active, desc="Reference KMeans blocks"):
            xi = microstates.loc[
                microstates.domain_id == domain, ["s", "rho", "kappa"]
            ].to_numpy(dtype=np.float64)
            cache_path = block_cache_dir / f"domain_{domain:03d}.npz"
            cached_blocks = load_block_cache(cache_path, len(xi), args.neighborhood_size)
            if cached_blocks is None:
                cached_blocks = build_reference_repspat_blocks(xi, args.neighborhood_size)
                save_block_cache(cache_path, cached_blocks, len(xi), args.neighborhood_size)
            blocks[domain] = cached_blocks

    large_pair_semaphore = threading.Semaphore(1)

    def compute_pair(task: tuple[int, int, int, Path]) -> dict:
        index, d1, d2, path = task
        pair_seed = args.seed + index
        is_large = len(blocks[d1]) + len(blocks[d2]) >= args.large_pair_block_threshold
        lock = large_pair_semaphore if is_large else nullcontext()
        with lock:
            sums, sizes, diagonal = precompute_block_kernel_sums(blocks[d1], blocks[d2], "IMQ", 1.0)
            q_i = len(blocks[d1])
            observed_mask = np.zeros(len(sizes), dtype=bool)
            observed_mask[:q_i] = True
            observed = mmd2_from_block_sums(
                sums, sizes, observed_mask, ~observed_mask, diagonal, clip_zero=False
            )
            target = int(min(sizes[:q_i].sum(), sizes[q_i:].sum()))
            assignments = generate_reference_repspat_assignments(
                sizes, target, args.n_perm, np.random.RandomState(pair_seed)
            )
            null = biased_mmd2_batch_from_block_sums(
                sums, sizes, assignments, clip_zero=False
            )
            # Released package convention is exactly mean(null >= observed), with
            # no +1 correction and no numerical tie tolerance.
            exceed = int(np.count_nonzero(null >= observed))
            row = {
                "domain_i": d1, "domain_j": d2,
                "n_i": int(sizes[:q_i].sum()), "n_j": int(sizes[q_i:].sum()),
                "n_blocks_i": q_i, "n_blocks_j": len(blocks[d2]),
                "mmd2_obs": observed, "n_perm": args.n_perm,
                "n_exceed": exceed, "p_perm": exceed / args.n_perm,
                "seed": pair_seed,
            }
            atomic_npz(path, **row)
            return row

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, args.pair_workers)) as pool:
            futures = [pool.submit(compute_pair, task) for task in missing]
            for future in tqdm(as_completed(futures), total=len(futures), desc="M3R pair tests"):
                future.result()

    rows = []
    for index, (d1, d2) in enumerate(expected):
        row = load_pair(
            checkpoints / f"pair_{d1:03d}_{d2:03d}.npz",
            d1, d2, args.n_perm, args.seed + index,
        )
        if row is None:
            raise RuntimeError(f"M3R incomplete at pair {(d1, d2)}")
        rows.append(row)
    rows.sort(key=lambda row: (row["domain_i"], row["domain_j"]))
    q_values = benjamini_hochberg(np.asarray([row["p_perm"] for row in rows]))
    pair_table = pd.DataFrame([
        {
            "domain_1": int(row["domain_i"]), "domain_2": int(row["domain_j"]),
            "mmd2_imq_obs": float(row["mmd2_obs"]),
            "p_perm_imq": float(row["p_perm"]), "q_bh_imq": float(q_value),
            "bh_reject_imq": bool(q_value < args.alpha),
            "similarity_edge": bool(q_value >= args.alpha), **row,
        }
        for row, q_value in zip(rows, q_values)
    ])
    pair_table.to_parquet(out / "M3R_PAIR_TESTS.parquet", index=False)
    matrix = np.zeros((len(domains), len(domains)), dtype=np.float64)
    for row in pair_table.itertuples(index=False):
        matrix[row.domain_1, row.domain_2] = matrix[row.domain_2, row.domain_1] = row.mmd2_imq_obs
    np.save(out / "M3R_MMD_MATRIX.npy", matrix)

    graph, topology = graph_summary(pair_table, domains)
    component_rows = []
    domain_component_rows = []
    for component_id, component in enumerate(sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))):
        nodes = sorted(component)
        subgraph = graph.subgraph(nodes)
        is_clique = subgraph.number_of_edges() == len(nodes) * (len(nodes) - 1) // 2
        component_rows.append({
            "component_id": component_id, "domain_ids": json.dumps(nodes),
            "num_domains": len(nodes), "is_clique": is_clique,
            "status": "AMBIGUOUS_NON_CLIQUE" if not is_clique else "FULLY_CONNECTED",
        })
        domain_component_rows.extend(
            {"domain_id": domain, "component_id": component_id}
            for domain in nodes
        )
    pd.DataFrame(component_rows).to_parquet(out / "M3R_COMPONENTS.parquet", index=False)
    pd.DataFrame(domain_component_rows).sort_values("domain_id").to_parquet(
        out / "M3R_DOMAIN_TO_COMPONENT.parquet", index=False
    )

    primary = pd.read_parquet(args.primary_pairs).sort_values(["domain_1", "domain_2"])
    reference = pair_table.sort_values(["domain_1", "domain_2"])
    if not np.array_equal(primary[["domain_1", "domain_2"]].to_numpy(),
                          reference[["domain_1", "domain_2"]].to_numpy()):
        raise RuntimeError("Primary and M3R pair identities differ")
    primary_edges = primary.similarity_edge.to_numpy(dtype=bool)
    reference_edges = reference.similarity_edge.to_numpy(dtype=bool)
    union = np.count_nonzero(primary_edges | reference_edges)
    comparison = {
        "edge_jaccard": float(np.count_nonzero(primary_edges & reference_edges) / union) if union else 1.0,
        "raw_rejection_agreement": float(np.mean(
            (primary.p_perm_imq.to_numpy() < args.alpha) ==
            (reference.p_perm_imq.to_numpy() < args.alpha)
        )),
        "bh_decision_agreement": float(np.mean(
            primary.bh_reject_imq.to_numpy(dtype=bool) ==
            reference.bh_reject_imq.to_numpy(dtype=bool)
        )),
        "p_value_spearman_rho": float(spearmanr(
            primary.p_perm_imq, reference.p_perm_imq
        ).statistic),
        "primary_num_edges": int(primary_edges.sum()),
        "m3r_num_edges": int(reference_edges.sum()),
        "primary_num_rejections": int(primary.bh_reject_imq.sum()),
        "m3r_num_rejections": int(reference.bh_reject_imq.sum()),
        **topology,
    }
    summary = {
        "run_id": "M3R_REFERENCE_IMPLEMENTATION_SENSITIVITY",
        "status": "COMPLETE",
        "reference_details": {
            "block_count": "floor(n/m)", "m": args.neighborhood_size,
            "m_is_qvcg_adaptation": True,
            "clustering": "sklearn.KMeans(n_init=10, random_state=0)",
            "kernel": "IMQ", "c": 1.0, "mmd_estimator": "biased Eq. 6",
            "n_permutations": args.n_perm, "p_value": "n_exceed / B",
            "multiple_testing": "global BH", "alpha": args.alpha,
            "edge": "BH non-rejection",
        },
        **topology,
    }
    (out / "M3R_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "M3R_VS_M3_COMPARISON.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"summary": summary, "comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
