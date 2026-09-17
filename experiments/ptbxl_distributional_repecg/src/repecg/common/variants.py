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
    "mean_distance_recurrence": ExperimentVariant(mechanism="mean_distance")
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
    "order_scrambled": ExperimentVariant(mechanism="scramble_local_order"),
    "level1": ExperimentVariant(representation="signature_level1"),
    "time_reversed": ExperimentVariant(control="time_reverse")
}

PAPER04_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "time_shuffled": ExperimentVariant(mechanism="time_shuffle_local")
}

PAPER05_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "occupancy_only": ExperimentVariant(mechanism="occupancy_only")
}

PAPER06_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "shuffled_residual": ExperimentVariant(mechanism="shuffle_residual_given_z")
}

PAPER07_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "learned_lead_id": ExperimentVariant(mechanism="categorical_lead_id")
}

PAPER08_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "kmeans_tokens": ExperimentVariant(mechanism="kmeans_dictionary")
}

PAPER09_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
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
