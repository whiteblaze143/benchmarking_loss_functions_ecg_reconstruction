import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED = {
    "unet": "1101",
    "msvae": "1110",
    "ecgaim": "1100",
}
MODEL_IDS = [
    "unet__e1c1m0d1__s42",
    "msvae__e1c1m1d0__s42",
    "ecgaim__e1c1m0d0__s42",
]


def run_figure(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, script, *arguments],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
        },
        check=True,
    )


def assert_triplet(base: Path, required_inputs: list[Path]) -> None:
    provenance = json.loads(base.with_suffix(".provenance.json").read_text())
    assert set(provenance["outputs"]) == {"pdf", "svg", "png"}
    assert all(Path(item["path"]).is_file() for item in provenance["outputs"].values())
    assert {str(path) for path in required_inputs}.issubset(provenance["inputs"])


def test_echonext_stress_figure_includes_all_17_conditions(tmp_path):
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps({"masks": SELECTED}))
    conditions = [
        *(f"gaussian_{snr}db" for snr in (24, 12, 6, 0)),
        "baseline_drift",
        *(
            f"nstdb_{noise}_{snr}db"
            for noise in ("bw", "em", "ma")
            for snr in (24, 12, 6, 0)
        ),
    ]
    rows = [
        {
            "classifier": "EchoNext_Mini_12_SHD",
            "model_id": model_id,
            "condition": condition,
            "auroc": 0.8 - 0.001 * index,
        }
        for model_id in MODEL_IDS
        for index, condition in enumerate(conditions)
    ]
    database = tmp_path / "clinical.parquet"
    pd.DataFrame(rows).to_parquet(database, index=False)
    output = tmp_path / "stress"

    run_figure(
        "scripts/figures/figure_echonext_stress.py",
        "--database",
        str(database),
        "--selection",
        str(selection),
        "--output",
        str(output),
    )

    assert_triplet(output, [database, selection])
    metadata = json.loads(output.with_suffix(".provenance.json").read_text())["metadata"]
    assert "baseline drift" in metadata["conditions"]


def test_smartwatch_protocol_figure_includes_square_wave(tmp_path):
    rows = []
    for device in (
        "applewatch_serie8",
        "fitbitsense2",
        "samsunggalaxy6",
        "withingsscanwatch",
    ):
        for experiment in ("heart_rate", "r_wave_amplitude", "st_offset"):
            rows.append(
                {
                    "device": device,
                    "experiment_type": experiment,
                    "watch_vs_simulator_mae": 1.0,
                        "watch_vs_philips_max_aligned_cross_correlation": None,
                }
            )
        rows.append(
            {
                "device": device,
                "experiment_type": "square_wave",
                "watch_vs_simulator_mae": None,
                    "watch_vs_philips_max_aligned_cross_correlation": 0.9,
            }
        )
    database = tmp_path / "protocol.parquet"
    pd.DataFrame(rows).to_parquet(database, index=False)
    output = tmp_path / "protocol"

    run_figure(
        "scripts/figures/figure_smartwatch_protocol.py",
        "--database",
        str(database),
        "--output",
        str(output),
    )

    assert_triplet(output, [database])
    metadata = json.loads(output.with_suffix(".provenance.json").read_text())["metadata"]
    assert "square_wave_endpoint" in metadata
