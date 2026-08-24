from scripts.analyze_ecgaim_ludb_oracle import (
    decode_mask,
    dominates,
    factorial_contrasts,
    pareto_flags,
)


def test_decode_seven_factor_mask() -> None:
    assert decode_mask("1110114") == {
        "mse": 1, "correlation": 1, "derivative": 1, "vcg": 0,
        "energy_distance": 1, "lead_consistency": 1, "mmd_kernel": 4,
    }


def test_pareto_requires_weak_all_and_strict_one() -> None:
    endpoints = (("corr", 1), ("error", -1))
    best = {"corr": 0.9, "error": 0.1}
    worse = {"corr": 0.8, "error": 0.2}
    tradeoff = {"corr": 0.95, "error": 0.3}
    assert dominates(best, worse, endpoints)
    assert not dominates(best, tradeoff, endpoints)
    assert pareto_flags([best, worse, tradeoff], endpoints) == [True, False, True]


def test_factorial_interaction_is_difference_in_differences() -> None:
    rows = []
    for corr in (0, 1):
        for vcg in (0, 1):
            rows.append({
                "factorial_mask": f"1{corr}0{vcg}000",
                "signal_pearson_p05": 0.5 + 0.1 * corr + 0.2 * vcg + 0.3 * corr * vcg,
            })
    result = factorial_contrasts(rows)
    interaction = next(
        row for row in result
        if row["contrast_type"] == "binary_interaction"
        and row["contrast"] == "correlation_x_vcg"
        and row["endpoint"] == "signal_pearson_p05"
    )
    assert abs(interaction["mean_improvement"] - 0.3) < 1e-12
