#!/usr/bin/env python3
"""Build an evidence-tiered interpretation of all ECG reconstruction results so far.

The report deliberately separates corrected ``missing_leads_v2`` results from
legacy evaluator output.  Legacy signal-reconstruction rows remain useful when
restricted to the nine genuinely missing leads, but legacy QRS/delineation rows
are not used because they were computed from copied V2 and are therefore
degenerate.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
QUEUE_STATE = ROOT / "refine-logs/queue_3arch/queue_state.json"
JOB_LOG_DIR = ROOT / "refine-logs/queue_3arch/jobs"
DB_PATH = ROOT / "results/clinical_biomarkers_multids/clinical_metrics.db"
V2_CSV = (
    ROOT
    / "results/clinical_biomarkers_multids/clinical_metrics_summary_missing_leads_v2.csv"
)
ARCH_COMPLETENESS = (
    ROOT / "results/clinical_biomarkers_multids/architecture_completeness.json"
)
REPORT_PATH = ROOT / "results/analysis/results_so_far_detailed_2026-08-12.md"

OBSERVED_LEADS = ("I", "II", "V2")
MISSING_LEADS = ("III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6")
EPOCH_RE = re.compile(
    r"Epoch (\d+) \| Val Loss: ([+-]?[0-9.eE]+) "
    r"\| Val Missing Pearson: ([+-]?[0-9.eE]+)"
)
MASK_RE = re.compile(r"(\d{7})(?=_s\d+$)")


def f(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.1f}%"


def pval(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    numeric = float(value)
    if numeric == 0:
        return "<1e-300"
    if numeric < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.4f}"


def mask_from_id(model_id: str) -> str:
    match = MASK_RE.search(model_id)
    return match.group(1) if match else "unknown"


def architecture_from_id(model_id: str) -> str:
    if "msvae" in model_id:
        return "MSVAE"
    if "ecg_aim" in model_id:
        return "ECG-AIM"
    return "U-Net"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return output


def parse_training(queue_jobs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for job in queue_jobs:
        if not job["id"].startswith("msvae_") or job.get("status") != "completed":
            continue
        mask = mask_from_id(job["id"])
        log_path = JOB_LOG_DIR / f"{job['id']}.log"
        epochs = []
        if log_path.exists():
            epochs = [
                (int(epoch), float(loss), float(correlation))
                for epoch, loss, correlation in EPOCH_RE.findall(
                    log_path.read_text(errors="replace")
                )
            ]
        if not epochs:
            continue
        best = max(epochs, key=lambda item: item[2])
        final = epochs[-1]
        minutes = np.nan
        if job.get("started") and job.get("completed"):
            started = datetime.fromisoformat(job["started"].replace("Z", "+00:00"))
            completed = datetime.fromisoformat(job["completed"].replace("Z", "+00:00"))
            minutes = (completed - started).total_seconds() / 60
        rows.append(
            {
                "model_id": job["id"],
                "mask": mask,
                "corr": int(mask[1]),
                "deriv": int(mask[2]),
                "vcg": int(mask[3]),
                "ed": int(mask[4]),
                "lead": int(mask[5]),
                "mmd": int(mask[6]),
                "epoch1_r": epochs[0][2],
                "best_r": best[2],
                "best_epoch": best[0],
                "final_r": final[2],
                "selection_drop": best[2] - final[2],
                "final_loss": final[1],
                "minutes": minutes,
                "epochs": len(epochs),
            }
        )
    return pd.DataFrame(rows).sort_values("mask").reset_index(drop=True)


def missing_signal_aggregates(legacy: pd.DataFrame) -> pd.DataFrame:
    targets = [f"Signal_Lead_{lead}" for lead in MISSING_LEADS]
    selected = legacy[legacy["target"].isin(targets)].copy()
    aggregate = (
        selected.groupby(["dataset", "model_id"], as_index=False)
        .agg(
            missing_mae=("mae", "mean"),
            missing_pearson=("pearson_r", "mean"),
            missing_r2=("r2", "mean"),
            n_missing_leads=("target", "nunique"),
        )
    )
    aggregate["architecture"] = aggregate["model_id"].map(architecture_from_id)
    aggregate["mask"] = aggregate["model_id"].map(mask_from_id)
    return aggregate


def paired_binary_effects(
    frame: pd.DataFrame, metric: str, higher_is_better: bool = True
) -> list[dict]:
    indexed = frame.set_index("mask")
    output = []
    for factor, index in (("Correlation", 1), ("Derivative", 2), ("VCG", 3), ("Energy distance", 4), ("Lead consistency", 5)):
        differences = []
        for mask, row in indexed.iterrows():
            if mask[index] != "0":
                continue
            alternate = mask[:index] + "1" + mask[index + 1 :]
            if alternate in indexed.index:
                differences.append(float(indexed.loc[alternate, metric] - row[metric]))
        if differences:
            output.append(
                {
                    "factor": factor,
                    "pairs": len(differences),
                    "mean": statistics.mean(differences),
                    "median": statistics.median(differences),
                    "beneficial": sum(
                        value > 0 if higher_is_better else value < 0
                        for value in differences
                    ),
                    "min": min(differences),
                    "max": max(differences),
                }
            )
    for variant in range(1, 5):
        differences = []
        for mask, row in indexed.iterrows():
            if mask[-1] != "0":
                continue
            alternate = mask[:-1] + str(variant)
            if alternate in indexed.index:
                differences.append(float(indexed.loc[alternate, metric] - row[metric]))
        if differences:
            output.append(
                {
                    "factor": f"MMD-{variant} vs none",
                    "pairs": len(differences),
                    "mean": statistics.mean(differences),
                    "median": statistics.median(differences),
                    "beneficial": sum(
                        value > 0 if higher_is_better else value < 0
                        for value in differences
                    ),
                    "min": min(differences),
                    "max": max(differences),
                }
            )
    return output


def target_pivot(legacy: pd.DataFrame, dataset: str, target: str, value: str) -> pd.Series:
    selected = legacy[(legacy["dataset"] == dataset) & (legacy["target"] == target)]
    return selected.set_index("model_id")[value]


def main() -> None:
    queue = json.loads(QUEUE_STATE.read_text())
    jobs = queue["jobs"]
    queue_counts = Counter(job.get("status", "unknown") for job in jobs)
    queue_arch_counts = {
        architecture: Counter(
            job.get("status", "unknown")
            for job in jobs
            if job["id"].startswith(f"{architecture}_")
        )
        for architecture in ("msvae", "ecg_aim")
    }
    architecture_gate = json.loads(ARCH_COMPLETENESS.read_text())

    with sqlite3.connect(DB_PATH) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        metrics = pd.read_sql_query("SELECT * FROM clinical_metrics", connection)
        paired = pd.read_sql_query("SELECT * FROM paired_inference", connection)
    legacy = metrics[metrics["evaluation_version"] == "legacy_v1"].copy()
    v2 = metrics[metrics["evaluation_version"] == "missing_leads_v2"].copy()
    paired_v2 = paired[paired["evaluation_version"] == "missing_leads_v2"].copy()
    training = parse_training(jobs)
    signal = missing_signal_aggregates(legacy)

    v2_csv = pd.read_csv(V2_CSV)
    v2_csv_duplicates = int(v2_csv.duplicated().sum())
    v2_csv_unique = int(len(v2_csv.drop_duplicates()))

    report: list[str] = []
    add = report.append
    extend = report.extend
    add("# Detailed interpretation of ECG reconstruction results so far")
    add("")
    add(f"**Snapshot:** {datetime.now().astimezone().isoformat(timespec='seconds')}")
    add("")
    add(
        "This report separates *what the data currently demonstrates* from what is "
        "still incomplete or invalid. The corrected evaluator is authoritative for "
        "clinical claims. Legacy rows are retained only for exploratory signal-level "
        "analysis restricted to the nine reconstructed leads."
    )
    add("")

    add("## Executive verdict")
    add("")
    add(
        "1. **No architecture-level conclusion is currently permitted.** The registry "
        f"contains {architecture_gate['registered_models']['unet']}/160 U-Nets, "
        f"{architecture_gate['registered_models']['msvae']}/160 MSVAEs, and "
        f"{architecture_gate['registered_models']['ecg_aim']}/160 ECG-AIM models. "
        "The formal architecture gate is therefore closed."
    )
    add(
        "2. **The strongest repeatable U-Net loss finding is that correlation loss helps "
        "and energy distance hurts.** Across all 160 U-Net masks, correlation improved "
        "mean missing-lead Pearson on PTB-XL, EchoNext, and Sunnybrook; energy distance "
        "reduced it for every one of 80 matched pairs on each cohort."
    )
    if not training.empty:
        winner = training.loc[training["best_r"].idxmax()]
        add(
            f"3. **The best fully logged MSVAE training run so far is `{winner.model_id}`** "
            f"with best validation missing-lead Pearson {winner.best_r:.4f} at epoch "
            f"{int(winner.best_epoch)}. This is a training/validation result, not yet a "
            "corrected clinical or external-cohort result."
        )
    add(
        "4. **The one corrected PTB-XL model shows clinically meaningful degradation.** "
        "For `f_1000000_s42`, reconstruction lowers paired ECGFounder macro AUROC by "
        "0.0349 and macro AUPRC by 0.0774, while worsening Brier score and calibration "
        "error. All four patient-cluster bootstrap intervals exclude zero."
    )
    add(
        "5. **The corrected evaluator is not advancing.** It crashes on NaN LVH scores "
        "after completing the first PTB-XL model, then the watcher restarts it. The CSV "
        f"contains {len(v2_csv):,} rows but only {v2_csv_unique} unique rows "
        f"({v2_csv_duplicates:,} duplicates). SQLite correctly retains only the latest "
        "unique rows, and is the source used below."
    )
    add("")

    add("## Evidence and completeness audit")
    add("")
    extend(
        markdown_table(
            ["Layer", "Current coverage", "Usable for", "Not usable for"],
            [
                [
                    "MSVAE training logs",
                    f"{len(training)} fully parsed; {queue_arch_counts['msvae'].get('running', 0)} running; "
                    f"{queue_arch_counts['msvae'].get('failed', 0)} failed",
                    "Within-MSVAE validation ranking and convergence",
                    "Architecture or clinical superiority",
                ],
                [
                    "Corrected V2 evaluator",
                    f"{v2.model_id.nunique()} model, {v2.dataset.nunique()} dataset, "
                    f"{len(v2)} metric rows, {len(paired_v2)} paired rows",
                    "Paired PTB-XL ECGFounder, QRS, and LVH agreement for one U-Net",
                    "Model ranking, EchoNext, external generalization, architecture claims",
                ],
                [
                    "Legacy evaluator",
                    f"{legacy.model_id.nunique()} models across "
                    f"{legacy.dataset.nunique()} datasets",
                    "Exploratory nine-missing-lead signal and ST metrics",
                    "QRS/delineation endpoints and formal clinical claims",
                ],
                [
                    "ECG-AIM",
                    f"{queue_arch_counts['ecg_aim'].get('completed', 0)}/160 complete",
                    "Nothing yet",
                    "Any performance statement",
                ],
            ],
        )
    )
    add("")
    add(f"SQLite integrity check: **`{quick_check}`**.")
    add("")
    add(
        f"Queue state at report generation: {queue_counts.get('completed', 0)} completed, "
        f"{queue_counts.get('running', 0)} running, {queue_counts.get('failed', 0)} "
        f"failed, and {queue_counts.get('pending', 0)} pending out of {len(jobs)} jobs."
    )
    add("")

    add("## Experimental design and mask decoding")
    add("")
    add(
        "A mask is `MSE-Corr-Deriv-VCG-ED-Lead-MMD`. MSE is always enabled. "
        "Correlation, first-derivative L1, Kors vectorcardiogram, energy distance, and "
        "Goldberger lead-consistency losses are binary. The final digit selects no MMD "
        "(0), global adaptive RBF (1), anatomical Laplacian (2), anatomical multiscale "
        "IMQ (3), or temporal K-means multiscale IMQ (4)."
    )
    add("")
    extend(
        markdown_table(
            ["Position", "Symbol", "Meaning"],
            [
                ["1", "MSE", "Anchor reconstruction loss; always 1"],
                ["2", "Corr", "Pearson correlation loss"],
                ["3", "Deriv", "First-difference L1 loss"],
                ["4", "VCG", "Kors VCG angle and magnitude loss"],
                ["5", "ED", "Empirical energy distance"],
                ["6", "Lead", "Goldberger limb-lead consistency"],
                ["7", "MMD", "Kernel variant 0–4"],
            ],
        )
    )
    add("")
    add(
        "All current MSVAE runs use seed 42. Consequently, differences among masks do "
        "not yet include seed-to-seed uncertainty. Final composite losses are not "
        "comparable between masks because enabling an additional term changes the scale "
        "and meaning of the optimized objective; validation Pearson is the comparable "
        "training endpoint."
    )
    add("")

    add("## MSVAE training results")
    add("")
    if training.empty:
        add("No complete MSVAE training logs were parsable.")
    else:
        add(
            f"Across {len(training)} fully logged runs, best validation missing-lead "
            f"Pearson is mean {training.best_r.mean():.4f}, median "
            f"{training.best_r.median():.4f}, SD {training.best_r.std(ddof=1):.4f}, and "
            f"range {training.best_r.min():.4f}–{training.best_r.max():.4f}. "
            f"The median runtime is {training.minutes.median():.1f} minutes. "
            f"{int((training.best_epoch == 10).sum())}/{len(training)} runs achieve their "
            "best Pearson at epoch 10, so many configurations are still improving at the "
            "fixed training horizon."
        )
        add("")
        add(
            f"Checkpoint selection matters: {int((training.selection_drop > 0.01).sum())} "
            "runs lose more than 0.01 Pearson from their best epoch to epoch 10. The most "
            f"unstable run drops {training.selection_drop.max():.4f}. Rankings below use "
            "the best validation checkpoint rather than the last epoch."
        )
        add("")
        add("### Best and worst fully logged MSVAE configurations")
        add("")
        ranked = pd.concat(
            [training.nlargest(12, "best_r"), training.nsmallest(10, "best_r")]
        ).drop_duplicates("model_id")
        ranked = ranked.sort_values("best_r", ascending=False)
        extend(
            markdown_table(
                [
                    "Model",
                    "Best r",
                    "Best epoch",
                    "Epoch-1 r",
                    "Final r",
                    "Best→final drop",
                    "Minutes",
                ],
                [
                    [
                        row.model_id,
                        f(row.best_r),
                        int(row.best_epoch),
                        f(row.epoch1_r),
                        f(row.final_r),
                        f(row.selection_drop),
                        f(row.minutes, 1),
                    ]
                    for row in ranked.itertuples()
                ],
            )
        )
        add("")
        add(
            "The top run, `1100000`, is MSE plus correlation with no other auxiliary "
            "term. The next-best runs mostly add either MMD-3, VCG, or derivative loss "
            "without energy distance. This pattern is consistent with the U-Net external "
            "signal results, where the same `110000x` family dominates."
        )
        add("")
        add(
            "The severe failures cluster around energy distance: `1000110`, `1010103`, "
            "and `1001100` reach only 0.1095, 0.1279, and 0.3046 best Pearson. Some MMD "
            "variants rescue those combinations, producing very large positive pairwise "
            "effects, but that is recovery from an unstable baseline—not evidence that "
            "MMD is generally beneficial."
        )
        add("")
        add("### Matched MSVAE loss-component effects")
        add("")
        training_effects = paired_binary_effects(training, "best_r", True)
        extend(
            markdown_table(
                ["Change", "Matched pairs", "Mean Δr", "Median Δr", "Beneficial", "Range"],
                [
                    [
                        item["factor"],
                        item["pairs"],
                        f(item["mean"], 5),
                        f(item["median"], 5),
                        f"{item['beneficial']}/{item['pairs']}",
                        f"{f(item['min'], 4)} to {f(item['max'], 4)}",
                    ]
                    for item in training_effects
                ],
            )
        )
        add("")
        add(
            "These MSVAE effects are interim because the completed subset is not the full "
            "factorial and contains only one seed. The robust directional finding is "
            "energy distance: mean matched Δr is strongly negative and only 2/27 pairs "
            "improve. VCG is usually protective, especially in energy-distance failures. "
            "Correlation and derivative losses have positive medians but their means are "
            "pulled negative by single catastrophic interactions."
        )
        add("")
    failed = [job for job in jobs if job["id"].startswith("msvae_") and job.get("status") == "failed"]
    add("### Failed and provenance-limited MSVAE jobs")
    add("")
    if failed:
        extend(
            markdown_table(
                ["Job", "Recorded reason", "Interpretation"],
                [
                    [
                        job["id"],
                        job.get("error", "unknown"),
                        "CUDA/Triton hardware-access failure; not evidence that the loss mask is invalid",
                    ]
                    for job in failed
                ],
            )
        )
    add("")
    add(
        "The failed logs report `Invalid access of peer GPU memory over nvlink or a hardware "
        "error`. They should be retried after GPU health is established. The administratively "
        "completed `msvae_f_1111112_s42` checkpoint has no matching per-job log in the current "
        "log directory and is excluded from the quantitative training table; its compatibility "
        "audit also lacks the current source/data contract."
    )
    add("")

    add("## Corrected V2 PTB-XL evaluation: `f_1000000_s42`")
    add("")
    add(
        "This is the only model/dataset pair currently evaluated with independent "
        "delineation over the nine missing leads. It contains 2,198 ECG records from "
        "1,904 patients for ECGFounder and 2,197 records from 1,903 patients for QRS. "
        "All paired intervals below use 500 patient-cluster bootstrap replicates."
    )
    add("")
    add("### Paired original-versus-reconstructed clinical inference")
    add("")
    paired_order = {"auroc": 0, "auprc": 1, "brier": 2, "ece": 3, "mae": 4, "bias": 5}
    paired_sorted = paired_v2.assign(
        order=paired_v2["metric"].map(paired_order).fillna(99)
    ).sort_values(["endpoint", "order"])
    extend(
        markdown_table(
            ["Endpoint", "Metric", "Original/reference", "Reconstruction", "Δ", "95% CI", "p", "Patients"],
            [
                [
                    row.endpoint,
                    row.metric,
                    f(row.reference_value, 5),
                    f(row.reconstruction_value, 5),
                    f(row.delta, 5),
                    f"[{f(row.ci_low, 5)}, {f(row.ci_high, 5)}]",
                    pval(row.p_value),
                    int(row.n_patients),
                ]
                for row in paired_sorted.itertuples()
            ],
        )
    )
    add("")
    add(
        "Reconstruction reduces ECGFounder macro AUROC from 0.8841 to 0.8492 "
        "(Δ −0.0349, 95% CI −0.0439 to −0.0261) and macro AUPRC from 0.4769 to "
        "0.3995 (Δ −0.0774, 95% CI −0.0996 to −0.0523). Brier score rises by "
        "0.00048 and expected calibration error rises by 0.00500; because lower is "
        "better for both, calibration also worsens. These are paired degradation "
        "estimates, not merely differences between two independent summaries."
    )
    add("")
    add(
        "The QRS estimate is systematically short by 14.31 ms, with MAE 16.84 ms and "
        "patient-bootstrap MAE interval 16.38–17.35 ms. LVH voltage is systematically "
        "attenuated by 0.397 mV, with MAE 0.559 mV. Both biases exclude zero by a wide "
        "margin, indicating directional amplitude/timing distortion rather than only "
        "random reconstruction noise."
    )
    add("")
    qrs = v2[v2["target"] == "QRS_Overall"]
    if not qrs.empty:
        row = qrs.iloc[0]
        add("### Corrected missing-lead QRS endpoint")
        add("")
        extend(
            markdown_table(
                ["MAE ms", "Pearson", "R²", "Bias ms", "Limits of agreement ms", "AUROC", "AUPRC", "Sensitivity", "Specificity", "Adjusted OR (95% CI), p"],
                [[
                    f(row.mae),
                    f(row.pearson_r),
                    f(row.r2),
                    f(row.bland_bias),
                    f"{f(row.loa_low)} to {f(row.loa_high)}",
                    f(row.auroc),
                    f(row.auprc),
                    f(row.sens),
                    f(row.spec),
                    f"{f(row.adj_or)} ({f(row.adj_or_ci_low)}–{f(row.adj_or_ci_high)}), p={f(row.pval_logistic)}",
                ]],
            )
        )
        add("")
        add(
            "The continuous QRS agreement is only moderate (r=0.532, R²=0.283), with "
            "wide limits of agreement from −43.51 to +14.88 ms. The near-perfect AUPRC "
            "and 99.3% sensitivity should not be read as overall clinical fidelity: "
            "specificity is only 33.3%, and the adjusted association with conduction "
            "disease is non-significant (OR 0.646, 95% CI 0.245–1.700, p=0.376). The "
            "threshold result is therefore prevalence-sensitive and poorly discriminates "
            "the negative class despite a high ranking metric."
        )
        add("")

    add("### ECGFounder task-level performance on reconstructed ECGs")
    add("")
    tasks = v2[
        v2["target"].str.startswith("ECGFounder_")
        & (v2["target"] != "ECGFounder_Macro_150")
    ].sort_values("auroc", ascending=False)
    add(
        f"Among {len(tasks)} reported tasks, {int((tasks.auroc >= 0.9).sum())} have "
        f"AUROC ≥0.90, but {int((tasks.auprc < 0.1).sum())} have AUPRC <0.10, "
        f"{int((tasks.f1 == 0).sum())} have F1=0, and "
        f"{int((tasks.sens < 0.2).sum())} have sensitivity <0.20 at the fixed 0.5 "
        "threshold. Median AUROC is high, but median sensitivity is only "
        f"{tasks.sens.median():.3f}. This is classic rare-label behavior: ranking can "
        "look strong while thresholded detection remains clinically weak."
    )
    add("")
    extend(
        markdown_table(
            ["Task", "AUROC (CI)", "AUPRC (CI)", "F1", "Sensitivity", "Specificity", "PPV", "NPV"],
            [
                [
                    row.target.replace("ECGFounder_", ""),
                    f"{f(row.auroc)} ({f(row.auroc_ci_low)}–{f(row.auroc_ci_high)})",
                    f"{f(row.auprc)} ({f(row.auprc_ci_low)}–{f(row.auprc_ci_high)})",
                    f(row.f1),
                    f(row.sens),
                    f(row.spec),
                    f(row.ppv),
                    f(row.npv),
                ]
                for row in tasks.itertuples()
            ],
        )
    )
    add("")
    add(
        "Task-level intervals above use the evaluator's 50-replicate, record-level "
        "bootstrap rather than "
        "the 500-replicate patient-cluster bootstrap used for macro paired deltas. They "
        "are useful descriptively but do not yet satisfy the requested patient-level "
        "inference standard for every task."
    )
    add("")

    add("## Legacy signal-level results: exploratory only")
    add("")
    add(
        "The legacy evaluator directly compares reconstructed and target waveforms per "
        "lead. Those rows remain informative after removing the copied observed leads "
        "I, II, and V2. Its QRS, boundary, and morphology rows are invalid: zero QRS and "
        "boundary error and Dice ≈1.0 for every U-Net reveal that the old code measured "
        "the copied V2 rather than reconstructed leads. No claim below uses those rows."
    )
    add("")
    summary_rows = []
    for (dataset, architecture), group in signal.groupby(["dataset", "architecture"]):
        summary_rows.append(
            [
                dataset,
                architecture,
                len(group),
                f(group.missing_pearson.mean()),
                f(group.missing_pearson.median()),
                f"{f(group.missing_pearson.min())}–{f(group.missing_pearson.max())}",
                f(group.missing_mae.mean()),
                f(group.missing_mae.median()),
            ]
        )
    extend(
        markdown_table(
            ["Dataset", "Architecture", "Models", "Mean r", "Median r", "r range", "Mean MAE", "Median MAE"],
            summary_rows,
        )
    )
    add("")
    add(
        "MAE is dataset-scale dependent and should not be compared directly between "
        "PTB-XL, EchoNext, Sunnybrook, and LUDB. Pearson rank patterns are much more "
        "stable across cohorts. The early 19-model MSVAE subset appears stronger than "
        "U-Net on PTB-XL and EchoNext, but it is selectively sampled, includes models "
        "that later failed inference-readiness retries, and was evaluated only with the "
        "legacy semantics. It is hypothesis-generating, not an architecture result."
    )
    add("")

    add("### Full 160-mask U-Net factorial effects on missing-lead signal Pearson")
    add("")
    unet_signal = signal[signal["architecture"] == "U-Net"]
    factor_rows = []
    for dataset in ("ptb_xl", "echonext", "sunnybrook"):
        group = unet_signal[unet_signal["dataset"] == dataset]
        for item in paired_binary_effects(group, "missing_pearson", True):
            factor_rows.append(
                [
                    dataset,
                    item["factor"],
                    item["pairs"],
                    f(item["mean"], 5),
                    f(item["median"], 5),
                    f"{item['beneficial']}/{item['pairs']}",
                ]
            )
    extend(
        markdown_table(
            ["Dataset", "Change", "Pairs", "Mean Δr", "Median Δr", "Beneficial"],
            factor_rows,
        )
    )
    add("")
    add(
        "Correlation is the cleanest positive factor: it improves all 80 matched U-Net "
        "pairs on each of PTB-XL, EchoNext, and Sunnybrook, by mean Δr +0.150, +0.130, "
        "and +0.140. Energy distance is the cleanest negative factor: it harms all 80 "
        "pairs on every cohort, by mean Δr −0.243, −0.231, and −0.209. Derivative loss "
        "is also consistently negative for PTB-XL and Sunnybrook and mostly negative on "
        "EchoNext. VCG and lead consistency have small, interaction-dependent effects. "
        "MMD-3 is closest to neutral; other MMD variants usually reduce Pearson, although "
        "some improve MAE slightly."
    )
    add("")

    add("### Cross-dataset stability and strongest U-Nets")
    add("")
    wide_r = unet_signal.pivot(index="model_id", columns="dataset", values="missing_pearson")
    correlation_rows = []
    for left, right in (("ptb_xl", "echonext"), ("ptb_xl", "sunnybrook"), ("echonext", "sunnybrook")):
        available = wide_r[[left, right]].dropna()
        rho, p_value = spearmanr(available[left], available[right])
        correlation_rows.append([left, right, len(available), f(rho), pval(p_value)])
    extend(markdown_table(["Dataset A", "Dataset B", "Models", "Spearman ρ", "p"], correlation_rows))
    add("")
    add(
        "Rank correlation is exceptionally high (ρ=0.960–0.991). Within the U-Net "
        "family, loss-mask ranking therefore transfers across these three cohorts rather "
        "than being a PTB-XL-only artifact. The `110000x` family—MSE plus correlation, "
        "optionally with MMD—is consistently strongest."
    )
    add("")
    top_rows = []
    for dataset in ("ptb_xl", "echonext", "sunnybrook"):
        group = unet_signal[unet_signal["dataset"] == dataset].nlargest(10, "missing_pearson")
        for rank, row in enumerate(group.itertuples(), 1):
            top_rows.append([dataset, rank, row.model_id, f(row.missing_pearson), f(row.missing_mae), f(row.missing_r2)])
    extend(
        markdown_table(
            ["Dataset", "Rank", "Model", "Missing-lead r", "MAE", "Mean R²"],
            top_rows,
        )
    )
    add("")

    add("### Sunnybrook dedicated signal endpoints")
    add("")
    sunny = legacy[(legacy["dataset"] == "sunnybrook") & legacy["model_id"].str.startswith("f_")]
    sunny_pivot = sunny.pivot(index="model_id", columns="target", values="mae")
    sunny_rows = []
    for metric, direction in (
        ("Signal_Missing_Leads_Pearson", "higher"),
        ("Signal_Missing_Leads_MSE", "lower"),
        ("Signal_Missing_Leads_SNR_dB", "higher"),
        ("Signal_Missing_Leads_DTW", "lower"),
    ):
        values = sunny_pivot[metric]
        best = values.nlargest(3) if direction == "higher" else values.nsmallest(3)
        sunny_rows.append(
            [
                metric,
                direction,
                f(values.mean()),
                f(values.min()),
                f(values.max()),
                "; ".join(f"{model}={f(value)}" for model, value in best.items()),
            ]
        )
    extend(markdown_table(["Metric", "Better", "Mean", "Min", "Max", "Top three"], sunny_rows))
    add("")
    add(
        "`f_1100000_s42`, `f_1100003_s42`, and `f_1100004_s42` dominate Pearson, "
        "MSE, and SNR. DTW favors `f_1100011_s42` and `f_1100012_s42`, showing that "
        "temporal alignment and pointwise/morphologic fidelity are related but not "
        "identical objectives."
    )
    add("")

    add("### Legacy ST-segment and ECGFounder observations")
    add("")
    st_targets = [f"ST_Lead_{lead}" for lead in MISSING_LEADS]
    st = (
        legacy[(legacy["dataset"] == "ptb_xl") & legacy["target"].isin(st_targets)]
        .groupby("model_id", as_index=False)
        .agg(st_mae=("mae", "mean"), st_r=("pearson_r", "mean"))
    )
    st["architecture"] = st["model_id"].map(architecture_from_id)
    st_rows = []
    for architecture, group in st.groupby("architecture"):
        for rank, row in enumerate(group.nlargest(8, "st_r").itertuples(), 1):
            st_rows.append([architecture, rank, row.model_id, f(row.st_r), f(row.st_mae)])
    extend(markdown_table(["Architecture", "Rank", "Model", "Mean missing-lead ST r", "Mean ST MAE"], st_rows))
    add("")
    macro = legacy[(legacy["dataset"] == "ptb_xl") & (legacy["target"] == "ECGFounder_Macro_150")].copy()
    macro["architecture"] = macro["model_id"].map(architecture_from_id)
    macro_rows = []
    for architecture, group in macro.groupby("architecture"):
        macro_rows.append(
            [
                architecture,
                len(group),
                f(group.auroc.mean()),
                f(group.auroc.median()),
                f(group.auroc.max()),
                f(group.auprc.mean()),
                f(group.auprc.max()),
                group.loc[group.auroc.idxmax(), "model_id"],
            ]
        )
    extend(
        markdown_table(
            ["Architecture", "Models", "Mean AUROC", "Median AUROC", "Best AUROC", "Mean AUPRC", "Best AUPRC", "Best-AUROC model"],
            macro_rows,
        )
    )
    add("")
    add(
        "The legacy ECGFounder table lacks paired original-ECG predictions and patient-"
        "cluster deltas, so the apparent MSVAE advantage cannot be promoted to an "
        "architecture claim. It is nevertheless consistent with the signal and ST "
        "rankings: the early `factorial_msvae_1000013_s42` model leads all three."
    )
    add("")

    add("## Dataset-by-dataset status")
    add("")
    extend(
        markdown_table(
            ["Dataset", "Corrected V2 status", "Legacy status", "Interpretation now"],
            [
                ["PTB-XL", "1 U-Net, partial post-processing", "160 U-Net + 19 early MSVAE", "Only one-model paired clinical result is claimable"],
                ["EchoNext", "No rows", "Signal/QRS rows only; no classifier comparison", "Actual original-vs-reconstructed EchoNext classifier result is still missing"],
                ["Sunnybrook", "No rows", "160 U-Nets", "Signal endpoints exploratory; old delineation endpoints invalid"],
                ["LUDB", "No rows", "7 U-Nets", "Too incomplete for mask or architecture claims"],
                ["ISP", "No rows", "No rows", "Not evaluated"],
                ["Zhejiang", "No rows", "No integrated rows", "Not evaluated in this database"],
            ],
        )
    )
    add("")

    add("## What can and cannot be claimed")
    add("")
    add("### Supported now")
    add("")
    add(
        "- In the complete 160-mask U-Net factorial, correlation loss robustly improves "
        "missing-lead waveform correlation across PTB-XL, EchoNext, and Sunnybrook."
    )
    add(
        "- In the same U-Net factorial, energy distance robustly harms missing-lead "
        "correlation across all three cohorts."
    )
    add(
        "- For corrected PTB-XL `f_1000000_s42`, reconstruction significantly degrades "
        "paired ECGFounder macro discrimination and calibration."
    )
    add(
        "- For that model, reconstructed missing leads systematically shorten QRS and "
        "attenuate Sokolow-Lyon voltage."
    )
    add("")
    add("### Not supported yet")
    add("")
    add("- MSVAE versus U-Net versus ECG-AIM superiority.")
    add("- Best loss mask across architectures.")
    add("- Any corrected EchoNext classifier preservation claim.")
    add("- Corrected external-cohort delineation or morphology preservation.")
    add("- Seed-robust factorial effects for MSVAE or ECG-AIM.")
    add("- Clinical equivalence or non-inferiority of reconstructed ECGs.")
    add("")

    add("## Required next actions, in priority order")
    add("")
    add(
        "1. **Stop the evaluator retry loop and filter non-finite LVH values before all "
        "classification and bootstrap calls.** The completion sentinel is never written "
        "because the crash occurs before `ST_Lead_V6`, causing repeated 70-minute passes."
    )
    add(
        "2. **Rebuild the V2 CSV from SQLite rather than appending per retry.** SQLite is "
        "deduplicated by primary key; the CSV is not."
    )
    add(
        "3. **Resume corrected evaluation and verify the first model reaches its final "
        "sentinel, then advances to model 2.** Do not infer progress from process uptime."
    )
    add(
        "4. **Add the actual EchoNext classifier comparison** with original-versus-"
        "reconstructed predictions, patient-level paired AUROC/AUPRC, Brier, ECE, and "
        "bootstrap deltas. The current EchoNext rows are only signal/QRS summaries."
    )
    add(
        "5. **Complete all MSVAE and ECG-AIM masks and rerun the corrected evaluator** "
        "before any architecture statement."
    )
    add(
        "6. **Add multiple seeds for shortlisted masks.** The current seed-42 MSVAE "
        "ranking has no reproducibility interval."
    )
    add(
        "7. **Investigate GPU ECC/hardware failures before retrying the three failed "
        "MSVAE jobs.** Their failures are infrastructural, not interpretable outcomes."
    )
    add("")

    add("## Appendix A: every fully logged MSVAE training model")
    add("")
    if not training.empty:
        extend(
            markdown_table(
                ["Mask", "Corr", "Deriv", "VCG", "ED", "Lead", "MMD", "Epoch-1 r", "Best r", "Best epoch", "Final r", "Drop", "Minutes"],
                [
                    [
                        row.mask,
                        row.corr,
                        row.deriv,
                        row.vcg,
                        row.ed,
                        row.lead,
                        row.mmd,
                        f(row.epoch1_r),
                        f(row.best_r),
                        int(row.best_epoch),
                        f(row.final_r),
                        f(row.selection_drop),
                        f(row.minutes, 1),
                    ]
                    for row in training.itertuples()
                ],
            )
        )
    add("")

    add("## Appendix B: every U-Net model on legacy signal-valid axes")
    add("")
    unet_models = sorted(
        model for model in legacy.model_id.unique() if architecture_from_id(model) == "U-Net"
    )
    signal_lookup = signal.set_index(["dataset", "model_id"])
    st_lookup = st.set_index("model_id")
    macro_lookup = macro.set_index("model_id")
    appendix_rows = []
    for model in unet_models:
        values = []
        for dataset in ("ptb_xl", "echonext", "sunnybrook"):
            key = (dataset, model)
            if key in signal_lookup.index:
                values.extend(
                    [
                        f(signal_lookup.loc[key, "missing_pearson"]),
                        f(signal_lookup.loc[key, "missing_mae"]),
                    ]
                )
            else:
                values.extend(["—", "—"])
        appendix_rows.append(
            [
                model,
                mask_from_id(model),
                *values,
                f(st_lookup.loc[model, "st_r"]) if model in st_lookup.index else "—",
                f(macro_lookup.loc[model, "auroc"]) if model in macro_lookup.index else "—",
                f(macro_lookup.loc[model, "auprc"]) if model in macro_lookup.index else "—",
            ]
        )
    extend(
        markdown_table(
            ["Model", "Mask", "PTB r", "PTB MAE", "EchoNext r", "EchoNext MAE", "Sunnybrook r", "Sunnybrook MAE", "PTB ST r", "Legacy macro AUROC", "Legacy macro AUPRC"],
            appendix_rows,
        )
    )
    add("")

    add("## Appendix C: every early MSVAE model in the legacy evaluator")
    add("")
    msvae_models = sorted(
        model for model in legacy.model_id.unique() if architecture_from_id(model) == "MSVAE"
    )
    appendix_rows = []
    for model in msvae_models:
        values = []
        for dataset in ("ptb_xl", "echonext"):
            key = (dataset, model)
            if key in signal_lookup.index:
                values.extend(
                    [
                        f(signal_lookup.loc[key, "missing_pearson"]),
                        f(signal_lookup.loc[key, "missing_mae"]),
                        f(signal_lookup.loc[key, "missing_r2"]),
                    ]
                )
            else:
                values.extend(["—", "—", "—"])
        appendix_rows.append(
            [
                model,
                mask_from_id(model),
                *values,
                f(st_lookup.loc[model, "st_r"]) if model in st_lookup.index else "—",
                f(macro_lookup.loc[model, "auroc"]) if model in macro_lookup.index else "—",
                f(macro_lookup.loc[model, "auprc"]) if model in macro_lookup.index else "—",
            ]
        )
    extend(
        markdown_table(
            ["Model", "Mask", "PTB r", "PTB MAE", "PTB R²", "EchoNext r", "EchoNext MAE", "EchoNext R²", "PTB ST r", "Legacy macro AUROC", "Legacy macro AUPRC"],
            appendix_rows,
        )
    )
    add("")

    add("## Source artifacts")
    add("")
    for path, description in (
        (QUEUE_STATE, "Live 320-job queue state"),
        (JOB_LOG_DIR, "Per-job MSVAE training logs"),
        (DB_PATH, "Authoritative clinical metrics and paired-inference SQLite store"),
        (V2_CSV, "Append-only V2 CSV; audited for duplicates, not used as authority"),
        (ARCH_COMPLETENESS, "Architecture claim gate"),
        (ROOT / "scripts/common_loss.py", "Factorial mask and loss definitions"),
        (ROOT / "scripts/evaluate_clinical_biomarkers_multids.py", "Evaluator semantics and failure location"),
    ):
        add(f"- `{path.relative_to(ROOT)}` — {description}")
    add("")
    add(
        "Generated by `scripts/build_results_so_far_report.py` using the project virtual "
        "environment. Rebuild after queue or evaluator changes to refresh all tables."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(REPORT_PATH)
    print(f"lines={len(report)} bytes={REPORT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
