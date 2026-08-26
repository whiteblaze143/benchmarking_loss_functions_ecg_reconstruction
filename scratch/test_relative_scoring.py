import numpy as np

def monotonic_match_indices(reference: np.ndarray, predicted: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    # Mocking the real function for testing
    from scripts.evaluate_ecgaim_ludb_blinded_daemon import monotonic_match_indices
    return monotonic_match_indices(reference, predicted, tolerance)

def test_relative_scoring():
    reference = np.array([100, 200, 300, 400])
    orig_pred = np.array([102, 215, 301, 450]) # 102 matches 100, 301 matches 300 (tol=10)
    model_pred = np.array([98, 205, 405]) # 98 matches 100, 205 matches 200, 405 matches 400

    tolerance = 10
    
    orig_pairs = monotonic_match_indices(reference, orig_pred, tolerance)
    model_pairs = monotonic_match_indices(reference, model_pred, tolerance)
    
    orig_matched_refs = set(ref_idx for ref_idx, pred_idx in orig_pairs)
    model_matched_refs = set(ref_idx for ref_idx, pred_idx in model_pairs)
    
    preserved = orig_matched_refs & model_matched_refs
    recovered = model_matched_refs - orig_matched_refs
    lost = orig_matched_refs - model_matched_refs
    
    print(f"Orig pairs: {orig_pairs} -> matched refs: {orig_matched_refs}")
    print(f"Model pairs: {model_pairs} -> matched refs: {model_matched_refs}")
    print(f"Preserved: {len(preserved)}, Recovered: {len(recovered)}, Lost: {len(lost)}")
    
    orig_fp = len(orig_pred) - len(orig_pairs)
    model_fp = len(model_pred) - len(model_pairs)
    delta_fp = model_fp - orig_fp
    print(f"Orig FP: {orig_fp}, Model FP: {model_fp}, Delta FP: {delta_fp}")
    
test_relative_scoring()
