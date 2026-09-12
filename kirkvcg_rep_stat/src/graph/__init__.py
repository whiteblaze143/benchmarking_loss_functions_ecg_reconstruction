"""Graph and clique reassignment module for VCG repSpat."""
from .clique_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_cliques_to_reps,
)

__all__ = [
    "build_similarity_graph",
    "extract_maximal_cliques",
    "reassign_cliques_to_reps",
]
