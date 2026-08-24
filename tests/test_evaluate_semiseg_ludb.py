import numpy as np

from scripts.evaluate_semiseg_ludb import (
    AuthorPreprocessor,
    boundary_arrays,
    components,
    evaluation_view,
    monotonic_match,
    notebook_interval_values,
)


def test_components_and_boundaries_are_inclusive():
    mask = np.zeros(100, dtype=int)
    mask[10:20], mask[30:40], mask[50:70] = 1, 2, 3
    assert components(mask)[1] == [(10, 19)]
    bounds = boundary_arrays(mask)
    assert bounds["P_onset"].tolist() == [10]
    assert bounds["P_offset"].tolist() == [19]
    assert bounds["QRS_offset"].tolist() == [39]
    assert bounds["T_offset"].tolist() == [69]


def test_match_is_one_to_one_and_minimum_error():
    assert monotonic_match([100, 110], [105], 10) in ([(0, 0)], [(1, 0)])
    assert monotonic_match([100, 200], [90, 101, 201], 20) == [(0, 1), (1, 2)]


def test_notebook_interval_reproduction_retains_zero_convention():
    mask = np.zeros(2500, dtype=int)
    mask[100:120], mask[150:180], mask[220:260] = 1, 2, 3
    values = notebook_interval_values(mask)
    assert values["PR"] == 200.0
    assert values["QRS"] == 116.0
    assert values["QT"] == 436.0


def test_author_preprocessor_shapes_and_modes():
    time = np.arange(5000) / 500
    signal = np.sin(2 * np.pi * time)
    label = np.zeros(5000, dtype=int); label[100:400] = 1; label[500:800] = 2; label[900:1200] = 3
    process = AuthorPreprocessor()
    no_padding, native_label = process(signal, label, "none")
    notebook, _ = process(signal, label, "notebook")
    assert no_padding.shape == (1, 2500)
    assert native_label.shape == (2500,)
    assert np.isfinite(no_padding).all() and np.isfinite(notebook).all()
    assert not np.array_equal(no_padding, notebook)


def test_annotated_window_excludes_edges_without_changing_reference_clock():
    reference = np.zeros(20, dtype=int)
    reference[5:8], reference[10:13] = 1, 2
    predicted = reference.copy()
    predicted[1:3], predicted[17:19] = 3, 3
    ref_iou, pred_iou, pred_boundaries = evaluation_view(reference, predicted, "annotated")
    assert np.array_equal(ref_iou, reference[5:13])
    assert np.array_equal(pred_iou, predicted[5:13])
    assert np.flatnonzero(pred_boundaries).tolist() == [5, 6, 7, 10, 11, 12]
