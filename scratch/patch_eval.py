with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "r") as f:
    code = f.read()

import re

# We will modify score_relative_boundaries to only run monotonic_match_indices for orig_pred if it's not cached.
# Actually we can just cache `orig_tp20`, `orig_errors` inside `ORIGINAL_MATCHES`!

new_func = """def score_relative_boundaries(
    records: list[dict[str, Any]], 
    model_predictions: dict[tuple[int, int], dict[str, np.ndarray]],
    lead_indices: Iterable[int],
    is_original: bool = False
) -> list[dict[str, Any]]:
    \"\"\"Calculates per-lead relative metrics (Preserved, Recovered, Delta F1, Delta FP) for reconstructed signals.\"\"\"
    rows = []
    tolerance20 = round(20 * TARGET_FS / 1000)
    tolerance150 = round(150 * TARGET_FS / 1000)
    
    for lead_index in lead_indices:
        lead_name = f"lead_{lead_index}"
        for boundary in BOUNDARIES:
            ref_total = 0
            model_pred_total = 0
            orig_pred_total = 0
            
            orig_tp20 = 0
            model_tp20 = 0
            
            orig_errors = []
            model_errors = []
            
            preserved_total = 0
            recovered_total = 0
            lost_total = 0
            
            for record_index, record in enumerate(records):
                reference = record["boundaries"][lead_index][boundary]
                ref_total += len(reference)
                
                model_pred = model_predictions[(record_index, lead_index)][boundary]
                model_pred_total += len(model_pred)
                
                # Model matching
                model_pairs20 = monotonic_match_indices(reference, model_pred, tolerance20)
                model_pairs150 = monotonic_match_indices(reference, model_pred, tolerance150)
                model_matched_refs = set(ref_idx for ref_idx, pred_idx in model_pairs20)
                model_tp20 += len(model_pairs20)
                model_errors.extend(
                    float(model_pred[right] - reference[left]) * 1000 / TARGET_FS
                    for left, right in model_pairs150
                )
                
                # Original matching
                cache_key = (record_index, lead_index, boundary)
                if is_original:
                    orig_matched_refs = model_matched_refs
                    orig_pred = model_pred
                    orig_pred_total += len(orig_pred)
                    orig_tp20 += len(model_pairs20)
                    orig_errors_rec = [float(orig_pred[right] - reference[left]) * 1000 / TARGET_FS for left, right in model_pairs150]
                    orig_errors.extend(orig_errors_rec)
                    
                    ORIGINAL_MATCHES[cache_key] = {
                        "matched_refs": orig_matched_refs,
                        "pred_total": len(orig_pred),
                        "tp20": len(model_pairs20),
                        "errors": orig_errors_rec,
                    }
                else:
                    cached = ORIGINAL_MATCHES.get(cache_key)
                    if cached:
                        orig_matched_refs = cached["matched_refs"]
                        orig_pred_total += cached["pred_total"]
                        orig_tp20 += cached["tp20"]
                        orig_errors.extend(cached["errors"])
                    else:
                        orig_matched_refs = set()
                
                # Calculate relative recovery sets
                preserved = orig_matched_refs & model_matched_refs
                recovered = model_matched_refs - orig_matched_refs
                lost = orig_matched_refs - model_matched_refs
                
                preserved_total += len(preserved)
                recovered_total += len(recovered)
                lost_total += len(lost)
            
            orig_denom = ref_total + orig_pred_total
            model_denom = ref_total + model_pred_total
            orig_f1 = 2 * orig_tp20 / orig_denom if orig_denom else None
            model_f1 = 2 * model_tp20 / model_denom if model_denom else None
            
            delta_f1 = (model_f1 - orig_f1) if (orig_f1 is not None and model_f1 is not None) else None
            retention_pct = (model_f1 / orig_f1 * 100) if (orig_f1 and model_f1 is not None) else None
            
            orig_err_arr = np.asarray(orig_errors, dtype=float)
            model_err_arr = np.asarray(model_errors, dtype=float)
            orig_mae = float(np.abs(orig_err_arr).mean()) if len(orig_err_arr) else None
            model_mae = float(np.abs(model_err_arr).mean()) if len(model_err_arr) else None
            delta_mae = (model_mae - orig_mae) if (orig_mae is not None and model_mae is not None) else None
            
            orig_fp = orig_pred_total - orig_tp20
            model_fp = model_pred_total - model_tp20
            delta_fp = model_fp - orig_fp
            
            rows.append({
                "lead_name": lead_name,
                "boundary": boundary,
                "reference_events": ref_total,
                "orig_predicted": orig_pred_total,
                "model_predicted": model_pred_total,
                "orig_tp20": orig_tp20,
                "model_tp20": model_tp20,
                "orig_f1_20ms": orig_f1,
                "model_f1_20ms": model_f1,
                "delta_f1_20ms": delta_f1,
                "retention_pct": retention_pct,
                "orig_mae_ms": orig_mae,
                "model_mae_ms": model_mae,
                "delta_mae_ms": delta_mae,
                "orig_fp": orig_fp,
                "model_fp": model_fp,
                "delta_fp": delta_fp,
                "preserved_events": preserved_total,
                "recovered_events": recovered_total,
                "lost_events": lost_total,
            })
            
    return rows"""

pattern = re.compile(r"def score_relative_boundaries\([\s\S]*?    return rows", re.MULTILINE)
code = pattern.sub(new_func, code)

# ALSO update the ORIGINAL_MATCHES type hint
code = code.replace("ORIGINAL_MATCHES: dict[tuple[int, int, str], set[int]] = {}", "ORIGINAL_MATCHES: dict[tuple[int, int, str], dict] = {}")

with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "w") as f:
    f.write(code)
