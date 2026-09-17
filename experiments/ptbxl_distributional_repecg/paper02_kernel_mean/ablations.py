from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
OUTPUT = EXPERIMENT / "outputs/paper02_kernel_mean"


def main() -> None:
    evaluation = OUTPUT / "development_evaluation"
    metrics = pd.read_csv(evaluation / "metrics.csv")
    required = {"kernel", "moments", "gaussian", "linear"}
    if set(metrics.variant) != required:
        raise RuntimeError(f"expected variants {sorted(required)}, found {sorted(metrics.variant)}")
    kernel = metrics.set_index("variant").loc["kernel"]
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "variant": row.variant,
                "macro_auroc": row.macro_auroc,
                "delta_kernel_minus_variant_macro_auroc": float(kernel.macro_auroc - row.macro_auroc),
                "macro_auprc": row.macro_auprc,
                "micro_auroc": row.micro_auroc,
                "brier": row.brier,
                "ece": row.ece,
                "validation_threshold_macro_f1": row.validation_threshold_macro_f1,
            }
        )
    pd.DataFrame(rows).to_csv(evaluation / "ablations.csv", index=False)
    gate = json.loads((evaluation / "claim_gate.json").read_text())
    print(json.dumps({"rows": len(rows), "claim_gate": gate["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
