import numpy as np
import torch

from scripts.evaluate_comprehensive_registry import normalize_compiled_state_dict
import pytest

from scripts.clinical_metrics import (
    binary_task_metrics,
    multilabel_classification_metrics,
    probability_fidelity_metrics,
)


def test_binary_metrics_include_threshold_calibration_and_support():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])
    result = binary_task_metrics(probabilities, labels)
    assert result["status"] == "ok"
    assert result["support_positive"] == 2
    assert result["support_negative"] == 2
    assert result["threshold"] == 0.5
    assert result["threshold_source"] == "fixed_predefined"
    assert result["tp"] == result["tn"] == result["fp"] == result["fn"] == 1
    for key in ("auroc", "average_precision", "f1", "precision", "recall", "brier", "ece"):
        assert result[key] is not None


def test_single_class_task_is_explicitly_undefined_without_nan():
    result = binary_task_metrics(np.array([0.1, 0.2]), np.array([0, 0]))
    assert result["status"] == "single_class"
    assert result["auroc"] is None
    assert result["average_precision"] is None
    assert result["support_positive"] == 0
    assert result["support_negative"] == 2


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0.0, 0.9]),
        np.array([0.0, 1.9]),
        np.array([-0.2, 1.0]),
    ],
)
def test_binary_metrics_reject_fractional_or_out_of_range_labels(labels):
    with pytest.raises(ValueError, match="only 0/1"):
        binary_task_metrics(np.array([0.2, 0.8]), labels)


def test_multilabel_metrics_store_every_named_task_and_micro_macro():
    labels = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    probabilities = np.array([[0.1, 0.8], [0.9, 0.2], [0.8, 0.7], [0.2, 0.1]])
    result = multilabel_classification_metrics(probabilities, labels, ["A", "B"])
    assert result["n_tasks"] == 2
    assert result["n_tasks_with_both_classes"] == 2
    assert set(result["per_task"]) == {"A", "B"}
    assert result["macro_auroc"] == result["macro"]["auroc"]
    assert result["micro"]["status"] == "ok"


def test_probability_fidelity_is_labeled_proxy_not_ground_truth():
    reference = np.array([[0.1, 0.8], [0.9, 0.2]])
    reconstructed = reference.copy()
    result = probability_fidelity_metrics(reconstructed, reference, ["A", "B"])
    assert result["ground_truth_status"] == "unavailable"
    assert result["evaluation_type"].endswith("_proxy")
    assert result["macro"]["probability_mae"] == 0.0
    assert result["macro"]["threshold_agreement"] == 1.0


def test_compiled_state_prefix_is_normalized_without_relaxing_keys():
    tensor = torch.ones(2)
    state = normalize_compiled_state_dict({"_orig_mod.layer.weight": tensor})
    assert list(state) == ["layer.weight"]
    assert state["layer.weight"] is tensor
