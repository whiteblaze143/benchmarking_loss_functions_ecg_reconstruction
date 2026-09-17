from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_key TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    paper_id INTEGER NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    variant TEXT NOT NULL,
    seed INTEGER NOT NULL,
    learning_rate REAL,
    weight_decay REAL,
    checkpoint_path TEXT,
    source_path TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('complete', 'failed', 'ineligible')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS metrics (
    run_key TEXT NOT NULL REFERENCES runs(run_key),
    metric_name TEXT NOT NULL,
    class_name TEXT NOT NULL DEFAULT '',
    metric_value REAL NOT NULL,
    PRIMARY KEY (run_key, metric_name, class_name)
);
CREATE INDEX IF NOT EXISTS metrics_lookup ON metrics(metric_name, class_name);
CREATE TABLE IF NOT EXISTS expected_evaluations (
    run_type TEXT NOT NULL,
    paper_id INTEGER NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    variant TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'failed', 'ineligible')),
    reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_type, paper_id, dataset, split, variant, seed)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(SCHEMA)
    return connection


def _metric_rows(metrics: dict[str, Any]) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for name, value in metrics.items():
        if isinstance(value, list):
            if len(value) != len(CLASSES):
                raise ValueError(f"metric {name!r}: expected {len(CLASSES)} class values, found {len(value)}")
            rows.extend((name, class_name, float(item)) for class_name, item in zip(CLASSES, value))
        else:
            rows.append((name, "", float(value)))
    return rows


def _insert_run(connection: sqlite3.Connection, row: dict[str, Any], metrics: dict[str, Any]) -> None:
    existing = connection.execute(
        "SELECT payload_sha256 FROM runs WHERE run_key = ?", (row["run_key"],)
    ).fetchone()
    if existing:
        if existing[0] != row["payload_sha256"]:
            raise RuntimeError(f"conflicting payload for existing run key {row['run_key']}")
        return
    connection.execute(
        """INSERT INTO runs (
            run_key, run_type, paper_id, dataset, split, variant, seed,
            learning_rate, weight_decay, checkpoint_path, source_path,
            payload_sha256, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        tuple(row[column] for column in (
            "run_key", "run_type", "paper_id", "dataset", "split", "variant", "seed",
            "learning_rate", "weight_decay", "checkpoint_path", "source_path",
            "payload_sha256", "status",
        )),
    )
    connection.executemany(
        "INSERT INTO metrics (run_key, metric_name, class_name, metric_value) VALUES (?, ?, ?, ?)",
        [(row["run_key"], name, class_name, value) for name, class_name, value in _metric_rows(metrics)],
    )


def ingest_paper(
    database: Path,
    paper_id: int,
    output: Path,
    dataset: str,
    split: str,
    run_type: str,
) -> dict[str, int]:
    manifest = json.loads((output / "manifest.json").read_text())
    seed = int(manifest["seed"])
    variants = list(manifest["variants"])
    cell_paths = sorted((output / "cells").glob("*/summary.json"))
    evaluation_paths = sorted((output / "development_evaluation").glob("metrics_*.json"))
    if len(cell_paths) != 9 * len(variants):
        raise RuntimeError(f"expected {9 * len(variants)} cell summaries, found {len(cell_paths)}")
    if len(evaluation_paths) != len(variants):
        raise RuntimeError(f"expected {len(variants)} evaluation summaries, found {len(evaluation_paths)}")

    with connect(database) as connection:
        for path in cell_paths:
            payload_bytes = path.read_bytes()
            payload = json.loads(payload_bytes)
            variant = payload["variant"]
            learning_rate = float(payload["learning_rate"])
            weight_decay = float(payload["weight_decay"])
            run_key = (
                f"{run_type}:paper{paper_id:02d}:{dataset}:{split}:{variant}:seed{seed}:"
                f"lr{learning_rate:.17g}:wd{weight_decay:.17g}"
            )
            _insert_run(connection, {
                "run_key": run_key,
                "run_type": run_type + "_selection_cell",
                "paper_id": paper_id,
                "dataset": dataset,
                "split": split,
                "variant": variant,
                "seed": seed,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "checkpoint_path": str(path.parent / "checkpoint.pt"),
                "source_path": str(path),
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "status": "complete",
            }, payload["metrics"])
        for path in evaluation_paths:
            payload_bytes = path.read_bytes()
            payload = json.loads(payload_bytes)
            variant = payload["variant"]
            run_key = f"{run_type}:paper{paper_id:02d}:{dataset}:{split}:{variant}:seed{seed}:evaluation"
            _insert_run(connection, {
                "run_key": run_key,
                "run_type": run_type + "_evaluation",
                "paper_id": paper_id,
                "dataset": dataset,
                "split": split,
                "variant": variant,
                "seed": seed,
                "learning_rate": None,
                "weight_decay": None,
                "checkpoint_path": str(output / f"{variant}_best.pt"),
                "source_path": str(path),
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "status": "complete",
            }, payload["metrics"])
            connection.execute(
                """UPDATE expected_evaluations SET status = 'complete', reason = ''
                   WHERE run_type = ? AND paper_id = ? AND dataset = ? AND split = ?
                     AND variant = ? AND seed = ?""",
                (run_type, paper_id, dataset, split, variant, seed),
            )

        run_count = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE run_type LIKE ? AND paper_id = ? AND dataset = ? AND split = ?",
            (run_type + "%", paper_id, dataset, split),
        ).fetchone()[0]
        metric_count = connection.execute(
            """SELECT COUNT(*) FROM metrics m JOIN runs r USING (run_key)
               WHERE r.run_type LIKE ? AND r.paper_id = ? AND r.dataset = ? AND r.split = ?""",
            (run_type + "%", paper_id, dataset, split),
        ).fetchone()[0]
    expected_runs = 10 * len(variants)
    if run_count != expected_runs:
        raise RuntimeError(f"database coverage mismatch: expected {expected_runs} runs, found {run_count}")
    return {"runs": run_count, "metrics": metric_count}


def register_expected_evaluations(
    database: Path,
    *,
    run_type: str,
    paper_id: int,
    datasets: list[str],
    split: str,
    variants: list[str],
    seed: int,
) -> int:
    rows = [
        (run_type, paper_id, dataset, split, variant, seed, "pending", "")
        for dataset in datasets
        for variant in variants
    ]
    with connect(database) as connection:
        connection.executemany(
            """INSERT OR IGNORE INTO expected_evaluations
               (run_type, paper_id, dataset, split, variant, seed, status, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def coverage(database: Path, run_type: str) -> dict[str, int]:
    with connect(database) as connection:
        return {
            status: count
            for status, count in connection.execute(
                """SELECT status, COUNT(*) FROM expected_evaluations
                   WHERE run_type = ? GROUP BY status""",
                (run_type,),
            )
        }
