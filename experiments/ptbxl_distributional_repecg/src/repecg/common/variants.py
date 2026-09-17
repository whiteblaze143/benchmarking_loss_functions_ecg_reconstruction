from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class ExperimentVariant:
    representation: str = "full"
    mechanism: str = "full"
    head: str = "native"
    capacity: str = "native"
    control: str = "none"

PAPER01_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "mean_distance_recurrence": ExperimentVariant(mechanism="mean_distance"),
    "phase_content_permuted": ExperimentVariant(mechanism="phase_content_permutation"),
    "cyclic_relabel_sham": ExperimentVariant(control="cyclic_relabel_sham"),
}

PAPER02_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "linear_kernel": ExperimentVariant(representation="linear"),
    "no_circular": ExperimentVariant(mechanism="no_circular")
}

PAPER03_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "order_scrambled": ExperimentVariant(representation="order_scrambled"),
    "time_reversed": ExperimentVariant(representation="time_reversed"),
    "monotone_warp_sham": ExperimentVariant(representation="monotone_warp_sham"),
    "unordered_kme": ExperimentVariant(representation="unordered_kme"),
}

PAPER04_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "time_shuffled": ExperimentVariant(mechanism="time_shuffle_local")
}

PAPER05_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "occupancy_only": ExperimentVariant(representation="occupancy_only", mechanism="occupancy_only"),
    "chronology_shuffled": ExperimentVariant(representation="chronology_shuffled", mechanism="chronology_shuffle"),
    "identity_order_sham": ExperimentVariant(representation="identity_order_sham", control="identity_order_sham"),
}

PAPER06_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "full_signal": ExperimentVariant(representation="full_signal"),
    "macro_component": ExperimentVariant(representation="macro_component"),
    "residual_marginal": ExperimentVariant(representation="residual_marginal"),
    "macro_residual_marginals": ExperimentVariant(representation="macro_residual_marginals"),
    "shuffled_residual": ExperimentVariant(representation="shuffled_residual", mechanism="shuffle_residual_given_z"),
    "joint_pair_sham": ExperimentVariant(representation="joint_pair_sham", control="joint_pair_sham"),
}

PAPER07_VARIANTS = {
    "continuous_primary": ExperimentVariant(mechanism="continuous_primary"),
    "categorical_primary": ExperimentVariant(mechanism="categorical_primary"),
    "continuous_auxiliary": ExperimentVariant(mechanism="continuous_auxiliary"),
    "categorical_auxiliary": ExperimentVariant(mechanism="categorical_auxiliary"),
}

PAPER08_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "kmeans_tokens": ExperimentVariant(mechanism="kmeans_dictionary")
}

PAPER09_VARIANTS = {
    "full": ExperimentVariant(),
    "diagnosis_only": ExperimentVariant(mechanism="diagnosis_only"),
    "mismatched_q": ExperimentVariant(control="mismatch_q_waveform")
}

PAPER10_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "erm": ExperimentVariant(mechanism="erm"),
    "irm": ExperimentVariant(mechanism="irm"),
    "coral": ExperimentVariant(mechanism="coral"),
    "causirl": ExperimentVariant(mechanism="causirl")
}

PAPER11_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "shuffled_futures": ExperimentVariant(mechanism="shuffle_history_futures")
}

PAPER12_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "unconditional_z": ExperimentVariant(mechanism="unconditional_state")
}

PAPER13_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "no_propagation": ExperimentVariant(mechanism="no_structural_propagation"),
    "wrong_phase_surgery": ExperimentVariant(control="wrong_phase_surgery")
}

PAPER14_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "mmd_penalty_only": ExperimentVariant(mechanism="mmd_penalty_only")
}

PAPER15_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "capacity_matched_shared": ExperimentVariant(mechanism="shared_mechanism", capacity="matched")
}

def get_variants_for_paper(paper_id: int) -> Dict[str, ExperimentVariant]:
    name = f"PAPER{paper_id:02d}_VARIANTS"
    return globals()[name]
