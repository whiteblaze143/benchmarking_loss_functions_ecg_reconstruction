import numpy as np

from scripts.evaluate_comprehensive_registry import subgroup_metrics


def test_subgroup_metrics_use_ptbxl_sex_encoding_and_common_tasks():
    n = 60
    labels = np.tile(np.array([[0, 1], [1, 0]]), (n // 2, 1))
    probabilities = np.full((n, 2), 0.5, dtype=float)
    sexes = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    # female=1 is deliberately near-perfect; male=0 is deliberately inverted.
    probabilities[sexes == 1] = 0.1 + 0.8 * labels[sexes == 1]
    probabilities[sexes == 0] = 0.9 - 0.8 * labels[sexes == 0]
    ages = np.r_[np.full(20, 30), np.full(20, 50), np.full(20, 70)]

    result = subgroup_metrics(
        probabilities,
        labels,
        ages,
        sexes,
        ["task_a", "task_b"],
    )

    assert result["female"]["n"] == 30
    assert result["male"]["n"] == 30
    assert result["female"]["macro_auroc"] > result["male"]["macro_auroc"]
    assert result["gaps"]["sex_encoding"] == "PTB-XL: male=0, female=1"
    assert result["gaps"]["gender_comparability"]["n_common_tasks"] == 2
