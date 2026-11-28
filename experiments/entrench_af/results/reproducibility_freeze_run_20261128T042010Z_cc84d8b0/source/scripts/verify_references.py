#!/usr/bin/env python3
"""Verify that every DOI cited in a Markdown report is in config/references.yaml and not blocklisted."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from common import (
    CONFIG_DIR,
    compute_definitions_hash,
    get_git_hash,
    resolve_run_dir,
    stamp_dataframe,
    utc_timestamp_iso,
)

DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+")
TRAILING_PUNCT = ".,;:)]}>\"'`"


def _clean_doi(raw: str) -> str:
    s = raw.strip()
    while s and s[-1] in TRAILING_PUNCT:
        s = s[:-1]
    return s


def verify_markdown_references(
    md_path: str | Path,
    refs_yaml_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> pd.DataFrame:
    md_path = Path(md_path)
    if refs_yaml_path is None:
        refs_yaml_path = CONFIG_DIR / "references.yaml"
    refs_yaml_path = Path(refs_yaml_path)

    with open(refs_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    allowed = {str(r["doi"]).strip().lower(): r for r in cfg.get("references", [])}
    blocklisted = {str(d).strip().lower() for d in cfg.get("blocklisted_dois", [])}

    text = md_path.read_text(encoding="utf-8")
    raw_matches = DOI_PATTERN.findall(text)
    cleaned = [_clean_doi(m) for m in raw_matches]

    seen_order: list[str] = []
    counts: dict[str, int] = {}
    for doi in cleaned:
        key = doi.lower()
        if key not in counts:
            seen_order.append(doi)
            counts[key] = 0
        counts[key] += 1

    rows = []
    bad_blocklisted = []
    bad_unknown = []
    for doi in seen_order:
        key = doi.lower()
        is_block = key in blocklisted
        is_allowed = key in allowed
        if is_block:
            bad_blocklisted.append(doi)
        if not is_allowed:
            bad_unknown.append(doi)
        ref_meta = allowed.get(key, {})
        rows.append(
            {
                "doi": doi,
                "doi_lower": key,
                "count_in_report": counts[key],
                "in_references_yaml": is_allowed,
                "is_blocklisted": is_block,
                "ref_key": ref_meta.get("key", ""),
                "authors": ref_meta.get("authors", ""),
                "year": ref_meta.get("year", ""),
                "status": "BLOCKLISTED" if is_block else ("OK" if is_allowed else "UNKNOWN"),
            }
        )

    audit_df = pd.DataFrame(rows)
    rdir = resolve_run_dir(run_dir)
    out_csv = rdir / "stage0" / "reference_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df = stamp_dataframe(
        audit_df,
        git_hash=get_git_hash(),
        definitions_hash=compute_definitions_hash(),
        run_timestamp=utc_timestamp_iso(),
    )
    audit_df.to_csv(out_csv, index=False)

    assert not bad_blocklisted, f"Blocklisted/fabricated DOIs found in {md_path}: {bad_blocklisted}"
    assert not bad_unknown, f"Unverified DOIs not in {refs_yaml_path}: {bad_unknown}"
    return audit_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Markdown report DOIs against config/references.yaml")
    parser.add_argument("md_path", type=str, help="Path to Markdown report")
    parser.add_argument("--refs", type=str, default=str(CONFIG_DIR / "references.yaml"), help="Path to references.yaml")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (defaults to LATEST_RUN.txt)")
    args = parser.parse_args()

    df = verify_markdown_references(args.md_path, args.refs, args.run_dir)
    print(f"[verify_references] OK | verified {len(df)} unique DOIs in {args.md_path}")


if __name__ == "__main__":
    main()
